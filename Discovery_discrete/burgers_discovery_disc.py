import torch
import torch.nn as nn
import numpy as np
import scipy.io
import matplotlib.pyplot as plt

# CRÍTICO: Doble precisión obligatoria para métodos IRK de alto orden
torch.set_default_dtype(torch.float64)

# ==========================================
# 0. CUADRO DE BUTCHER (Gauss-Legendre IRK)
# ==========================================
def get_implicit_rk_coeffs(q):
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
            torch.tensor(b, dtype=torch.float64))

# ==========================================
# 1. MODELO FÍSICO DISCRETO (Inverso)
# ==========================================
class PINNDiscreteDiscovery(nn.Module):
    def __init__(self, layers, q, dt, A, b):
        super(PINNDiscreteDiscovery, self).__init__()
        self.q = q
        self.dt = dt
        self.register_buffer('A', A)
        self.register_buffer('b', b)
        
        self.linears = nn.ModuleList([nn.Linear(layers[i], layers[i+1]) for i in range(len(layers)-1)])
        self.activation = nn.Tanh()

        # Parámetros a descubrir (inicializados arbitrariamente)
        self.lambda_1 = nn.Parameter(torch.tensor([0.0], dtype=torch.float64))
        self.lambda_2 = nn.Parameter(torch.tensor([-6.0], dtype=torch.float64)) # Se usará exp(lambda_2)

    def forward(self, x):
        # La red predice las q etapas intermedias: [u^{n+c_1}, ..., u^{n+c_q}]
        inputs = x
        for i in range(len(self.linears) - 1):
            inputs = self.activation(self.linears[i](inputs))
        U_stages = self.linears[-1](inputs)
        return U_stages

    def net_residuals(self, x):
        U_stages = self.forward(x) # (N, q)
        
        N_u = []
        l2 = torch.exp(self.lambda_2)
        
        # Evaluamos el operador no lineal para cada etapa (Ec. B.10)[cite: 25]
        for j in range(self.q):
            u_j = U_stages[:, j]
            u_j_x = torch.autograd.grad(u_j.sum(), x, retain_graph=True, create_graph=True)[0]
            u_j_xx = torch.autograd.grad(u_j_x.sum(), x, retain_graph=True, create_graph=True)[0]
            
            # N[u] = lambda_1 * u * u_x - lambda_2 * u_xx
            n_j = self.lambda_1 * u_j.unsqueeze(1) * u_j_x - l2 * u_j_xx
            N_u.append(n_j)
            
        N_u_tensor = torch.cat(N_u, dim=1) # (N, q)
        
        # Retroproyección a t^n y t^{n+1} (Ec. B.5)[cite: 23, 24]
        U_n = U_stages + self.dt * torch.matmul(N_u_tensor, self.A.T)
        A_minus_b = self.A - self.b.unsqueeze(0)
        U_n_plus_1 = U_stages + self.dt * torch.matmul(N_u_tensor, A_minus_b.T)
        
        return U_n, U_n_plus_1

def loss_function(model, x_n, u_n_train, x_n1, u_n1_train):
    U_n_pred, _ = model.net_residuals(x_n)
    _, U_n1_pred = model.net_residuals(x_n1)
    
    # SSE_n y SSE_{n+1} (Ec. B.9)[cite: 25]
    sse_n = torch.sum((U_n_pred - u_n_train) ** 2)
    sse_n1 = torch.sum((U_n1_pred - u_n1_train) ** 2)
    
    return sse_n + sse_n1

# ==========================================
# 2. PROCESAMIENTO DE DATOS (Dos Snapshots)
# ==========================================
data = scipy.io.loadmat(r'C:\Users\ematm\OneDrive\Desktop\TFG\Trabajo TFG propuesto\data\burgers_shock.mat') 

x_raw = data['x']  
t_raw = data['t']  
Exact = np.real(data['usol']) 

# Extraemos los dos snapshots: t^n = 0.1 y t^{n+1} = 0.9 (dt = 0.8)[cite: 25]
idx_t0 = np.argmin(np.abs(t_raw.flatten() - 0.1))
idx_t1 = np.argmin(np.abs(t_raw.flatten() - 0.9))
dt = t_raw[idx_t1, 0] - t_raw[idx_t0, 0]

t_n_val = t_raw[idx_t0, 0]
t_n1_val = t_raw[idx_t1, 0]

np.random.seed(1234)
N_n, N_n1 = 199, 201
idx_x_n = np.random.choice(x_raw.shape[0], N_n, replace=False)
idx_x_n1 = np.random.choice(x_raw.shape[0], N_n1, replace=False)

x_n = x_raw[idx_x_n, :]
u_n_clean = Exact[idx_x_n, idx_t0:idx_t0+1]

x_n1 = x_raw[idx_x_n1, :]
u_n1_clean = Exact[idx_x_n1, idx_t1:idx_t1+1]

noise_level = 0.01 # 1% ruido[cite: 25, 26]
np.random.seed(1234)
u_n = u_n_clean + noise_level * np.std(u_n_clean) * np.random.randn(*u_n_clean.shape)
u_n1 = u_n1_clean + noise_level * np.std(u_n1_clean) * np.random.randn(*u_n1_clean.shape)

x_n_tensor = torch.tensor(x_n, dtype=torch.float64, requires_grad=True)
u_n_tensor = torch.tensor(u_n, dtype=torch.float64)

x_n1_tensor = torch.tensor(x_n1, dtype=torch.float64, requires_grad=True)
u_n1_tensor = torch.tensor(u_n1, dtype=torch.float64)

# ==========================================
# 3. CONFIGURACIÓN DEL MODELO Y ENTRENAMIENTO
# ==========================================
q = 32 # Etapas Runge-Kutta[cite: 25]
layers = [1, 50, 50, 50, 50, q]
A, b = get_implicit_rk_coeffs(q)

model = PINNDiscreteDiscovery(layers, q, dt, A, b)

optimizer = torch.optim.LBFGS(
    model.parameters(), lr=1.0, max_iter=10000, max_eval=10000, history_size=50,
    tolerance_grad=1e-8, tolerance_change=1.0 * np.finfo(float).eps, line_search_fn="strong_wolfe"
)

iter_info = [0]

def closure():
    optimizer.zero_grad()
    loss = loss_function(model, x_n_tensor, u_n_tensor, x_n1_tensor, u_n1_tensor)
    loss.backward()
    
    iter_info[0] += 1
    if iter_info[0] % 100 == 0:
        l1_val = model.lambda_1.item()
        l2_val = torch.exp(model.lambda_2).item()
        print(f'L-BFGS Iter: {iter_info[0]} | SSE Loss: {loss.item():.4e} | l1: {l1_val:.4f} | l2: {l2_val:.6f}')
        
    return loss

print(f"Iniciando descubrimiento discreto (dt={dt:.1f}, Ruido={noise_level*100}%)...")
optimizer.step(closure)
print("Entrenamiento finalizado.")

# ==========================================
# 4. EVALUACIÓN Y RESULTADOS
# ==========================================
lambda_1_pred = model.lambda_1.item()
lambda_2_pred = torch.exp(model.lambda_2).item()
lambda_1_exact = 1.0
lambda_2_exact = 0.01 / np.pi

error_l1 = np.abs(lambda_1_pred - lambda_1_exact) / lambda_1_exact * 100
error_l2 = np.abs(lambda_2_pred - lambda_2_exact) / lambda_2_exact * 100

print(f"\n--- Resultados del Descubrimiento Discreto ---")
print(f"Lambda 1 exacto: {lambda_1_exact:.6f}  | Predicho: {lambda_1_pred:.6f}  | Error: {error_l1:.3f}%")
print(f"Lambda 2 exacto: {lambda_2_exact:.6f} | Predicho: {lambda_2_pred:.6f} | Error: {error_l2:.3f}%")

# ==========================================
# 5. VISUALIZACIÓN (Figura B.9 Estilo Paper)
# ==========================================
fig = plt.figure(figsize=(10, 7))

# 1. Mapa de calor espaciotemporal superior
ax1 = plt.subplot2grid((3, 2), (0, 0), colspan=2)
X_grid, T_grid = np.meshgrid(x_raw.flatten(), t_raw.flatten(), indexing='ij')
h = ax1.imshow(Exact, interpolation='nearest', cmap='rainbow', 
              extent=[t_raw.min(), t_raw.max(), x_raw.min(), x_raw.max()], 
              origin='lower', aspect='auto')
fig.colorbar(h, ax=ax1)
# Líneas blancas verticales marcando los snapshots t^n y t^{n+1}[cite: 26]
ax1.axvline(x=t_n_val, color='white', linewidth=2)
ax1.axvline(x=t_n1_val, color='white', linewidth=2)
ax1.set_xlabel('$t$')
ax1.set_ylabel('$x$')
ax1.set_title('$u(t, x)$', fontsize=12)

# 2. Snapshot inferior izquierdo (t = 0.10)[cite: 26]
ax2 = plt.subplot2grid((3, 2), (1, 0), rowspan=2)
ax2.plot(x_raw, Exact[:, idx_t0], 'b-', linewidth=2, label='Exact')
ax2.plot(x_n, u_n, 'rx', markersize=6, label=f'Data ({N_n} points)')
ax2.set_xlabel('$x$')
ax2.set_ylabel('$u(t,x)$')
ax2.set_title(f'$t = {t_n_val:.2f}$\n{N_n} training data')
ax2.legend(loc='upper right', frameon=True)
ax2.set_xlim([-1, 1])
ax2.set_ylim([-1.1, 1.1])

# 3. Snapshot inferior derecho (t = 0.90)[cite: 26]
ax3 = plt.subplot2grid((3, 2), (1, 1), rowspan=2)
ax3.plot(x_raw, Exact[:, idx_t1], 'b-', linewidth=2, label='Exact')
ax3.plot(x_n1, u_n1, 'rx', markersize=6, label=f'Data ({N_n1} points)')
ax3.set_xlabel('$x$')
ax3.set_ylabel('$u(t,x)$')
ax3.set_title(f'$t = {t_n1_val:.2f}$\n{N_n1} training data')
ax3.legend(loc='upper right', frameon=True)
ax3.set_xlim([-1, 1])
ax3.set_ylim([-0.8, 0.8])

plt.tight_layout()
plt.show()