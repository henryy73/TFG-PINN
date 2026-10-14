import torch
import torch.nn as nn
import numpy as np
import scipy.io
import matplotlib.pyplot as plt

# Usamos doble precisión (float64) para coincidir con la implementación del artículo
torch.set_default_dtype(torch.float64)

# ==========================================
# CUADRO DE BUTCHER (Gauss-Legendre IRK)
# ==========================================

def get_implicit_rk_coeffs(q):
    """
    Calcula la matriz A (q x q) y los vectores b, c (q) del cuadro de Butcher
    para el método implícito de Gauss-Legendre de q etapas en el intervalo [0, 1].
    """
    nodes, weights = np.polynomial.legendre.leggauss(q)
    c = (nodes + 1.0) / 2.0
    b = weights / 2.0

    A = np.zeros((q, q))
    for i in range(q):
        ci = c[i]
        tau_quad = (ci / 2.0) * (nodes + 1.0)
        w_quad = (ci / 2.0) * weights

        for j in range(q):
            c_other = np.delete(c, j)
            num = np.prod(tau_quad[:, None] - c_other[None, :], axis=1)
            denom = np.prod(c[j] - c_other)
            L_j_vals = num / denom
            A[i, j] = np.sum(w_quad * L_j_vals)

    return (torch.tensor(A, dtype=torch.float64), 
            torch.tensor(b, dtype=torch.float64), 
            torch.tensor(c, dtype=torch.float64))

# ==========================================
# 1. MODELO FÍSICO (Sección A.2 - Ec. A.5 a A.8)
# ==========================================

class DiscretePhysicsInformedNN(nn.Module):
    def __init__(self, layers, q, dt, A, b):
        super(DiscretePhysicsInformedNN, self).__init__()
        self.q = q
        self.dt = dt
        self.register_buffer('A', A)  # Matriz a_ij (q x q)
        self.register_buffer('b', b)  # Pesos b_j (q)
        
        # Red neuronal: 4 capas ocultas x 50 neuronas, activación Tanh
        self.linears = nn.ModuleList([nn.Linear(layers[i], layers[i+1]) for i in range(len(layers)-1)])
        self.activation = nn.Tanh()

    def forward(self, x):
        inputs = x
        for i in range(len(self.linears) - 1):
            inputs = self.activation(self.linears[i](inputs))
        outputs = self.linears[-1](inputs)
        return outputs  # [u^{n+c_1}, ..., u^{n+c_q}, u^{n+1}]

    def net_u_n(self, x):
        U = self.forward(x)
        U_stages = U[:, :self.q]       # (N, q)
        u_next = U[:, self.q:self.q+1]  # (N, 1)
        
        nu = 0.01 / np.pi
        
        # Evaluamos N[u^{n+c_j}] = u * u_x - nu * u_xx para cada etapa j
        N_u = []
        for j in range(self.q):
            u_j = U_stages[:, j:j+1]
            u_j_x = torch.autograd.grad(u_j, x, grad_outputs=torch.ones_like(u_j), retain_graph=True, create_graph=True)[0]
            u_j_xx = torch.autograd.grad(u_j_x, x, grad_outputs=torch.ones_like(u_j_x), retain_graph=True, create_graph=True)[0]
            
            n_j = u_j * u_j_x - nu * u_j_xx
            N_u.append(n_j)
            
        N_u_tensor = torch.cat(N_u, dim=1)
        
        # Ecuaciones (A.5) y (A.6) del artículo: Reconstrucción en t^n
        u_i_n = U_stages + self.dt * torch.matmul(N_u_tensor, self.A.T)
        u_q_plus_1_n = u_next + self.dt * torch.matmul(N_u_tensor, self.b.unsqueeze(1))
        
        U_n = torch.cat([u_i_n, u_q_plus_1_n], dim=1)
        return U_n

def loss_function(model, x_n, u_n_train, x_b_left, x_b_right):
    # Ecuación (A.9) del artículo: SSE = SSE_n + SSE_b
    U_n_pred = model.net_u_n(x_n)
    sse_n = torch.sum((U_n_pred - u_n_train) ** 2)
    
    U_b_l = model(x_b_left)
    U_b_r = model(x_b_right)
    sse_b = torch.sum(U_b_l ** 2) + torch.sum(U_b_r ** 2)
    
    return sse_n + sse_b

# ==========================================
# 2. DATOS DE ENTRENAMIENTO
# ==========================================

data = scipy.io.loadmat(r'C:\Users\ematm\OneDrive\Desktop\TFG\Trabajo TFG propuesto\data\burgers_shock.mat') 

x_raw = data['x']  # (256, 1)
t_raw = data['t']  # (100, 1)
Exact = np.real(data['usol'])  # (256, 100)

idx_t0 = np.argmin(np.abs(t_raw.flatten() - 0.1))
idx_t1 = np.argmin(np.abs(t_raw.flatten() - 0.9))

t_n = t_raw[idx_t0, 0]
t_n1 = t_raw[idx_t1, 0]
dt = t_n1 - t_n  # dt = 0.8

N_n = 250
idx_x = np.random.choice(x_raw.shape[0], N_n, replace=False)
x_n_train = x_raw[idx_x, :]
u_n_train = Exact[idx_x, idx_t0:idx_t0+1]

# q = 32 etapas Runge-Kutta
q = 32
A, b, c = get_implicit_rk_coeffs(q)

x_n_tensor = torch.tensor(x_n_train, dtype=torch.float64, requires_grad=True)
u_n_tensor = torch.tensor(u_n_train, dtype=torch.float64)

x_b_left = torch.tensor([[-1.0]], dtype=torch.float64)
x_b_right = torch.tensor([[1.0]], dtype=torch.float64)

# Arquitectura descrita en el artículo: [1, 50, 50, 50, 50, q + 1]
layers = [1, 50, 50, 50, 50, q + 1]
model = DiscretePhysicsInformedNN(layers, q, dt, A, b)

# ==========================================
# 3. ENTRENAMIENTO L-BFGS (Sección 3.1 / A.1)
# ==========================================

optimizer = torch.optim.LBFGS(
    model.parameters(), 
    lr=1.0, 
    max_iter=10000, 
    max_eval=10000, 
    history_size=50,
    tolerance_grad=1e-8, 
    tolerance_change=1.0 * np.finfo(float).eps, 
    line_search_fn="strong_wolfe"
)

iter_count = 0

def closure():
    global iter_count
    optimizer.zero_grad()
    loss = loss_function(model, x_n_tensor, u_n_tensor, x_b_left, x_b_right)
    loss.backward()
    
    iter_count += 1
    if iter_count % 100 == 0:
        print(f'L-BFGS Iteración: {iter_count}, SSE Loss: {loss.item():.6f}')
        
    return loss

print(f"Iniciando entrenamiento L-BFGS (q={q} etapas, dt={dt:.2f})...")
optimizer.step(closure)
print("Entrenamiento finalizado.")

# ==========================================
# 4. EVALUACIÓN Y FIGURA A.7
# ==========================================

model.eval()
x_star_tensor = torch.tensor(x_raw, dtype=torch.float64)

with torch.no_grad():
    U_pred_all = model(x_star_tensor).numpy()
    u_pred_t1 = U_pred_all[:, -1:]

u_exact_t1 = Exact[:, idx_t1:idx_t1+1]

error_l2 = np.linalg.norm(u_exact_t1 - u_pred_t1, 2) / np.linalg.norm(u_exact_t1, 2)
print(f'\nError relativo L2 en t = {t_n1:.2f}: {error_l2:.2e}')

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].plot(x_n_train, u_n_train, 'rx', label=f'Datos ($N_n = {N_n}$)')
axes[0].plot(x_raw, Exact[:, idx_t0], 'b-', linewidth=1.5, label='Exacto')
axes[0].set_xlabel('$x$')
axes[0].set_ylabel('$u(t,x)$')
axes[0].set_title(f'$t = {t_n:.2f}$')
axes[0].legend(loc='upper right')
axes[0].set_xlim([-1, 1])

axes[1].plot(x_raw, u_exact_t1, 'b-', linewidth=2, label='Exacto')
axes[1].plot(x_raw, u_pred_t1, 'r--', linewidth=2, label='Predicción PINN')
axes[1].set_xlabel('$x$')
axes[1].set_ylabel('$u(t,x)$')
axes[1].set_title(f'$t = {t_n1:.2f}$ (Error L2: {error_l2:.2e})')
axes[1].legend(loc='upper right')
axes[1].set_xlim([-1, 1])

plt.tight_layout()
plt.show()