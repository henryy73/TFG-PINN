import torch
import torch.nn as nn
import numpy as np
import scipy.io
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float32)

# ==========================================
# 1. DEFINICIÓN DEL MODELO FÍSICO (Inverso)
# ==========================================
class PINNDiscovery(nn.Module):
    def __init__(self, layers):
        super(PINNDiscovery, self).__init__()
        self.linears = nn.ModuleList([nn.Linear(layers[i], layers[i+1]) for i in range(len(layers)-1)])
        self.activation = nn.Tanh()

        # Declaramos lambda_1 y lambda_2 como parámetros entrenables de la red
        # Se inicializan en 0.0. El optimizador descubrirá sus verdaderos valores.
        self.lambda_1 = nn.Parameter(torch.tensor([0.0], dtype=torch.float32))
        self.lambda_2 = nn.Parameter(torch.tensor([-6.0], dtype=torch.float32)) # Usamos -6 y aplicamos exp() luego para asegurar que la difusividad sea > 0

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
        
        # Operador de la ecuación de Burgers con los parámetros desconocidos
        # Ecuación: f = u_t + lambda_1 * u * u_x - lambda_2 * u_xx
        l2 = torch.exp(self.lambda_2) # Truco de Raissi et al. para forzar l2 > 0
        f = u_t + self.lambda_1 * u * u_x - l2 * u_xx
        return f

def loss_function(model, x, t, u_train):
    # En el problema inverso, evaluamos MSE_u y MSE_f en los mismos puntos
    u_pred = model(x, t)
    mse_u = torch.mean((u_pred - u_train) ** 2)
    
    f_pred = model.net_f(x, t)
    mse_f = torch.mean(f_pred ** 2)
    
    return mse_u + mse_f

# 9 capas con 20 neuronas (Arquitectura del paper original)
layers = [2, 20, 20, 20, 20, 20, 20, 20, 20, 1]
model = PINNDiscovery(layers)

optimizer = torch.optim.LBFGS(
    model.parameters(), lr=1.0, max_iter=10000, max_eval=10000, history_size=50,
    tolerance_grad=1e-5, tolerance_change=1.0 * np.finfo(float).eps, line_search_fn="strong_wolfe"
)

# ==========================================
# 2. PROCESAMIENTO DE DATOS (2000 puntos aleatorios)
# ==========================================
# Asegúrate de ajustar la ruta al archivo .mat de tu ordenador
data = scipy.io.loadmat(r'C:\Users\ematm\OneDrive\Desktop\TFG\Trabajo TFG propuesto\data\burgers_shock.mat') 

x = data['x']
t = data['t']
Exact = np.real(data['usol'])

X, T = np.meshgrid(x.flatten(), t.flatten(), indexing='ij')
X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
u_star = Exact.flatten()[:,None]

# Seleccionamos N = 2000 puntos aleatorios de todo el dominio espaciotemporal
N_train = 2000
np.random.seed(1234) # Fijar semilla para reproducibilidad
idx = np.random.choice(X_star.shape[0], N_train, replace=False)

x_train_np = X_star[idx, 0:1]
t_train_np = X_star[idx, 1:2]
u_train_np = u_star[idx, :]

# Inyección de ruido (0.0 para "clean data", 0.01 para "1% noise")
noise_level = 0.01 
if noise_level > 0:
    noise = noise_level * np.std(u_train_np) * np.random.randn(*u_train_np.shape)
    u_train_np = u_train_np + noise

# Conversión a tensores de PyTorch
x_train = torch.tensor(x_train_np, dtype=torch.float32, requires_grad=True)
t_train = torch.tensor(t_train_np, dtype=torch.float32, requires_grad=True)
u_train_tensor = torch.tensor(u_train_np, dtype=torch.float32)

# ==========================================
# 3. ENTRENAMIENTO
# ==========================================
iter_count = 0

def closure():
    global iter_count
    optimizer.zero_grad()
    loss = loss_function(model, x_train, t_train, u_train_tensor)
    loss.backward()
    
    iter_count += 1
    if iter_count % 100 == 0:
        l1_val = model.lambda_1.item()
        l2_val = torch.exp(model.lambda_2).item()
        print(f'Iter: {iter_count}, Loss: {loss.item():.5e}, l1: {l1_val:.5f}, l2: {l2_val:.6f}')
        
    return loss

print(f"Iniciando descubrimiento (Noise: {noise_level*100}%)...")
optimizer.step(closure)
print("Entrenamiento finalizado.")

# ==========================================
# 4. EVALUACIÓN DE PARÁMETROS DESCUBIERTOS
# ==========================================
lambda_1_pred = model.lambda_1.item()
lambda_2_pred = torch.exp(model.lambda_2).item()

lambda_1_exact = 1.0
lambda_2_exact = 0.01 / np.pi # Aprox 0.0031831

error_lambda_1 = np.abs(lambda_1_pred - lambda_1_exact) / lambda_1_exact * 100
error_lambda_2 = np.abs(lambda_2_pred - lambda_2_exact) / lambda_2_exact * 100

print(f"\n--- Resultados del Descubrimiento ---")
print(f"Lambda 1 exacto: {lambda_1_exact:.6f}  | Predicho: {lambda_1_pred:.6f}  | Error: {error_lambda_1:.3f}%")
print(f"Lambda 2 exacto: {lambda_2_exact:.6f} | Predicho: {lambda_2_pred:.6f} | Error: {error_lambda_2:.3f}%")

# ==========================================
# 5. VISUALIZACIÓN
# ==========================================
model.eval()
with torch.no_grad():
    x_star_tensor = torch.tensor(X_star[:, 0:1], dtype=torch.float32)
    t_star_tensor = torch.tensor(X_star[:, 1:2], dtype=torch.float32)
    u_pred = model(x_star_tensor, t_star_tensor).detach().numpy()

U_pred = u_pred.reshape(Exact.shape)

fig, ax = plt.subplots(figsize=(8, 4))
h = ax.imshow(U_pred, interpolation='nearest', cmap='rainbow', 
              extent=[t.min(), t.max(), x.min(), x.max()], 
              origin='lower', aspect='auto')
fig.colorbar(h)
ax.scatter(t_train_np, x_train_np, color='black', marker='x', s=10, label=f'Datos ({N_train} puntos)')
ax.set_xlabel('$t$')
ax.set_ylabel('$x$')
ax.set_title(f'Predicción PINN Inversa (Ruido: {noise_level*100}%)', fontsize=12)
ax.legend(loc='upper right', frameon=True)
plt.tight_layout()
plt.show()

