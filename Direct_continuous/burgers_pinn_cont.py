import torch
import torch.nn as nn
import numpy as np
import scipy.io
from scipy.stats import qmc  # Sustituto moderno de pyDOE
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float32)
# ==========================================
# 1. DEFINICIÓN DEL MODELO FÍSICO (El Motor)
# ==========================================

class PhysicsInformedNN(nn.Module):
    def __init__(self, layers):
        super(PhysicsInformedNN, self).__init__()
        self.linears = nn.ModuleList([nn.Linear(layers[i], layers[i+1]) for i in range(len(layers)-1)])
        self.activation = nn.Tanh()

    def forward(self, x, t):
        inputs = torch.cat([x, t], dim=1)
        for i in range(len(self.linears) - 1):
            inputs = self.activation(self.linears[i](inputs))
        u = self.linears[-1](inputs)
        return u

    def net_f(self, x, t):
        u = self.forward(x, t)
        
        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), retain_graph=True, create_graph=True)[0]
        
        nu = 0.01 / np.pi
        f = u_t + u * u_x - nu * u_xx
        return f

def loss_function(model, x_u, t_u, u_train, x_f, t_f):
    u_pred = model(x_u, t_u)
    mse_u = torch.mean((u_pred - u_train) ** 2)
    f_pred = model.net_f(x_f, t_f)
    mse_f = torch.mean(f_pred ** 2)
    return mse_u + mse_f

# Configuración de la red y optimizador
layers = [2, 20, 20, 20, 20, 20, 20, 20, 20, 1]
model = PhysicsInformedNN(layers)
optimizer = torch.optim.LBFGS(
    model.parameters(), lr=1.0, max_iter=10000, max_eval=10000, history_size=50,
    tolerance_grad=1e-5, tolerance_change=1.0 * np.finfo(float).eps, line_search_fn="strong_wolfe"
)

# ==========================================
# 2. PROCESAMIENTO DE DATOS (El Combustible)
# ==========================================

data = scipy.io.loadmat(r'C:\Users\ematm\OneDrive\Desktop\TFG\Trabajo TFG propuesto\data\burgers_shock.mat') 

x = data['x']  # Forma: (256, 1)
t = data['t']  # Forma: (100, 1)
Exact = np.real(data['usol'])  # Forma: (256, 100) - filas x, columnas t

# IMPORTANTE: Usamos indexing='ij' para que la malla tenga exactamente 
# la misma orientación y orden que la matriz Exact (256 x 100)
X, T = np.meshgrid(x.flatten(), t.flatten(), indexing='ij')
X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
u_star = Exact.flatten()[:,None]

# 1. Condición inicial (t = 0, x variable en [-1, 1])
xx1 = np.hstack((x, np.zeros_like(x)))
uu1 = Exact[:, 0:1]

# 2. Frontera izquierda (x = -1, t variable en [0, 1])
xx2 = np.hstack((-1.0 * np.ones_like(t), t))
uu2 = Exact[0:1, :].T

# 3. Frontera derecha (x = 1, t variable en [0, 1])
xx3 = np.hstack((1.0 * np.ones_like(t), t))
uu3 = Exact[-1:, :].T

# Juntamos todos los datos de frontera/iniciales
X_u_train = np.vstack([xx1, xx2, xx3])
u_train = np.vstack([uu1, uu2, uu3])

# Seleccionamos aleatoriamente Nu = 100 puntos para entrenar
idx = np.random.choice(X_u_train.shape[0], 100, replace=False)
X_u_train = X_u_train[idx, :]
u_train = u_train[idx, :]

# Generación de puntos de colocación (MSE_f) mediante Hipercubo Latino
lb = X_star.min(0)
ub = X_star.max(0)

sampler = qmc.LatinHypercube(d=2)
sample = sampler.random(n=10000)
X_f_train = lb + (ub - lb) * sample
X_f_train = np.vstack((X_f_train, X_u_train))

# Conversión a tensores de PyTorch
x_u = torch.tensor(X_u_train[:, 0:1], dtype=torch.float32)
t_u = torch.tensor(X_u_train[:, 1:2], dtype=torch.float32)
u_train_tensor = torch.tensor(u_train, dtype=torch.float32)

x_f = torch.tensor(X_f_train[:, 0:1], dtype=torch.float32, requires_grad=True)
t_f = torch.tensor(X_f_train[:, 1:2], dtype=torch.float32, requires_grad=True)


# ==========================================
# 3. ENTRENAMIENTO (El Acelerador)
# ==========================================

iter_count = 0

def closure():
    global iter_count
    optimizer.zero_grad()
    loss = loss_function(model, x_u, t_u, u_train_tensor, x_f, t_f)
    loss.backward()
    
    iter_count += 1
    if iter_count % 100 == 0:
        print(f'Iteración: {iter_count}, Pérdida: {loss.item():.6f}')
        
    return loss

print("Iniciando entrenamiento...")
optimizer.step(closure)
print("Entrenamiento finalizado.")

# ==========================================
# 4. EVALUACIÓN Y VISUALIZACIÓN (Resultados)
# ==========================================

# Evaluamos el modelo en toda la malla de referencia (X_star)
model.eval()
with torch.no_grad():
    x_star_tensor = torch.tensor(X_star[:, 0:1], dtype=torch.float32)
    t_star_tensor = torch.tensor(X_star[:, 1:2], dtype=torch.float32)
    u_pred = model(x_star_tensor, t_star_tensor).detach().numpy()

# Reconstruimos la matriz de predicción con la forma original de la solución exacta
# (Asumiendo que Exact tiene forma (256, 100))
U_pred = u_pred.reshape(Exact.shape)

# Cálculo del error relativo L2
error_l2 = np.linalg.norm(Exact - U_pred, 2) / np.linalg.norm(Exact, 2)
print(f'Error relativo L2: {error_l2:.2e}')

# --- Generación de Gráficas (Estilo Figura A.6 del artículo) ---
fig, ax = plt.subplots(figsize=(7, 4))

# 1. Mapa de calor de la predicción espaciotemporal
h = ax.imshow(U_pred, interpolation='nearest', cmap='rainbow', 
              extent=[t.min(), t.max(), x.min(), x.max()], 
              origin='lower', aspect='auto')
fig.colorbar(h)

# Superponemos las marcas de los puntos de entrenamiento iniciales y de contorno (X_u_train)
ax.scatter(X_u_train[:, 1], X_u_train[:, 0], color='black', marker='x', s=20, label='Datos (100 puntos)')
ax.set_xlabel('$t$')
ax.set_ylabel('$x$')
ax.set_title(f'Predicción PINN - Ecuación de Burgers (L2 Error: {error_l2:.2e})', fontsize=10)
ax.legend(loc='upper right', frameon=True)
plt.tight_layout()
plt.show()

# 2. Cortes transversales (Snapshots en t = 0.25, 0.50, 0.75)
fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
time_snapshots = [0.25, 0.50, 0.75]

# Encontramos los índices temporales más cercanos en el vector t
t_flat = t.flatten()
for i, t_val in enumerate(time_snapshots):
    idx_t = np.argmin(np.abs(t_flat - t_val))
    
    axes[i].plot(x, Exact[:, idx_t], 'b-', linewidth=2, label='Exacto')
    axes[i].plot(x, U_pred[:, idx_t], 'r--', linewidth=2, label='Predicción PINN')
    axes[i].set_xlabel('$x$')
    axes[i].set_ylabel('$u(t,x)$')
    axes[i].set_title(f'$t = {t_val}$')
    axes[i].legend(loc='upper right')
    axes[i].set_xlim([-1, 1])
    axes[i].set_ylim([-1.1, 1.1])

plt.tight_layout()
plt.show()