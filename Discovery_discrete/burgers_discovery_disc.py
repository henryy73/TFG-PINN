import os
import random
import warnings
import numpy as np
import scipy.io
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning)

# ==========================================
# 0. CONFIGURACIÓN GLOBAL, CUDA Y SEMILLA (42)
# ==========================================
# Doble precisión obligatoria para esquemas IRK de Gauss-Legendre de alto orden
torch.set_default_dtype(torch.float64)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.cuda.init()
    _ = torch.matmul(torch.zeros((1, 1), dtype=torch.float64, device=DEVICE), 
                     torch.zeros((1, 1), dtype=torch.float64, device=DEVICE))

print(f"Ejecutando en dispositivo: {DEVICE}" + (f" ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else ""))

# ==========================================
# 1. CUADRO DE BUTCHER (Gauss-Legendre IRK)
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
    return (torch.tensor(A, dtype=torch.float64, device=DEVICE), 
            torch.tensor(b, dtype=torch.float64, device=DEVICE))

# ==========================================
# 2. MODELO FÍSICO DISCRETO (Apéndice B.2 y B.3)
# ==========================================
class PINNDiscreteDiscovery(nn.Module):
    def __init__(self, layers, q, dt, A, b, lb=-1.0, ub=1.0):
        super(PINNDiscreteDiscovery, self).__init__()
        self.q = q
        self.dt = dt
        self.lb = lb
        self.ub = ub
        self.register_buffer('A', A)
        self.register_buffer('b', b)
        
        self.linears = nn.ModuleList([nn.Linear(layers[i], layers[i+1]) for i in range(len(layers)-1)])
        self.activation = nn.Tanh()

        # Inicialización Truncated Normal de Xavier idéntica al código original de Raissi et al.
        for m in self.linears:
            in_dim, out_dim = m.weight.shape[1], m.weight.shape[0]
            std = np.sqrt(2.0 / (in_dim + out_dim))
            nn.init.trunc_normal_(m.weight, mean=0.0, std=std, a=-2.0*std, b=2.0*std)
            nn.init.zeros_(m.bias)

        self.lambda_1 = nn.Parameter(torch.tensor([0.0], dtype=torch.float64))
        self.lambda_2 = nn.Parameter(torch.tensor([-6.0], dtype=torch.float64))

    def forward(self, x):
        # Normalización de entradas al dominio [-1, 1]
        H = 2.0 * (x - self.lb) / (self.ub - self.lb) - 1.0
        for i in range(len(self.linears) - 1):
            H = self.activation(self.linears[i](H))
        return self.linears[-1](H)

    def net_residuals(self, x):
        # Ecuación (B.6): Red prior sobre las q etapas [u^{n+c_1}(x), ..., u^{n+c_q}(x)]
        U_stages = self.forward(x) # (N, q)
        N_u = []
        l2 = torch.exp(self.lambda_2)
        
        # Ecuación (B.10): Operador espacial no lineal N[u] = lambda_1 * u * u_x - lambda_2 * u_xx
        for j in range(self.q):
            u_j = U_stages[:, j:j+1]
            u_j_x = torch.autograd.grad(u_j.sum(), x, retain_graph=True, create_graph=True)[0]
            u_j_xx = torch.autograd.grad(u_j_x.sum(), x, retain_graph=True, create_graph=True)[0]
            n_j = self.lambda_1 * u_j * u_j_x - l2 * u_j_xx
            N_u.append(n_j)
            
        N_u_tensor = torch.cat(N_u, dim=1) # (N, q)
        
        # Ecuación (B.5): Proyección hacia t^n y t^{n+1}
        U_n = U_stages + self.dt * torch.matmul(N_u_tensor, self.A.T)
        A_minus_b = self.A - self.b.unsqueeze(0)
        U_n_plus_1 = U_stages + self.dt * torch.matmul(N_u_tensor, A_minus_b.T)
        return U_n, U_n_plus_1

def loss_function(model, x_n, u_n_train, x_n1, u_n1_train):
    # Ecuación (B.9) del artículo: Suma de Errores Cuadráticos (SSE = SSE_n + SSE_{n+1})
    U_n_pred, _ = model.net_residuals(x_n)
    _, U_n1_pred = model.net_residuals(x_n1)
    sse_n = torch.sum((U_n_pred - u_n_train) ** 2)
    sse_n1 = torch.sum((U_n1_pred - u_n1_train) ** 2)
    return sse_n + sse_n1

# ==========================================
# 3. CARGA DE DATOS Y BUCLE (0% y 1% RUIDO)
# ==========================================
MAT_PATH = r'C:\Users\ematm\OneDrive\Desktop\TFG\Trabajo TFG propuesto\data\burgers_shock.mat'
if not os.path.exists(MAT_PATH):
    MAT_PATH = 'burgers_shock.mat'

data = scipy.io.loadmat(MAT_PATH)
x_raw = data['x']  
t_raw = data['t']  
Exact = np.real(data['usol'])  

# Dos snapshots en t^n = 0.1 y t^{n+1} = 0.9 (dt = 0.8), Sección B.3
idx_t0 = np.argmin(np.abs(t_raw.flatten() - 0.1))
idx_t1 = np.argmin(np.abs(t_raw.flatten() - 0.9))
dt = float(t_raw[idx_t1, 0] - t_raw[idx_t0, 0])

t_n_val = t_raw[idx_t0, 0]
t_n1_val = t_raw[idx_t1, 0]

q = 32
layers = [1, 50, 50, 50, 50, q] # 4 capas ocultas con 50 neuronas (Sección B.3)
noise_levels = [0.0, 0.01]

lambda_1_exact = 1.0
lambda_2_exact = 0.01 / np.pi

for noise_level in noise_levels:
    print(f"\n==================================================")
    print(f" EJECUTANDO DESCUBRIMIENTO DISCRETO (Ruido: {noise_level*100}%)")
    print(f"==================================================")
    
    set_seed(42)
    
    N_n, N_n1 = 199, 201
    idx_x_n = np.random.choice(x_raw.shape[0], N_n, replace=False)
    idx_x_n1 = np.random.choice(x_raw.shape[0], N_n1, replace=False)

    x_n = x_raw[idx_x_n, :]
    u_n_clean = Exact[idx_x_n, idx_t0:idx_t0+1]

    x_n1 = x_raw[idx_x_n1, :]
    u_n1_clean = Exact[idx_x_n1, idx_t1:idx_t1+1]

    if noise_level > 0:
        u_n = u_n_clean + noise_level * np.std(u_n_clean) * np.random.randn(*u_n_clean.shape)
        u_n1 = u_n1_clean + noise_level * np.std(u_n1_clean) * np.random.randn(*u_n1_clean.shape)
    else:
        u_n = u_n_clean.copy()
        u_n1 = u_n1_clean.copy()

    x_n_tensor = torch.tensor(x_n, dtype=torch.float64, requires_grad=True, device=DEVICE)
    u_n_tensor = torch.tensor(u_n, dtype=torch.float64, device=DEVICE)
    x_n1_tensor = torch.tensor(x_n1, dtype=torch.float64, requires_grad=True, device=DEVICE)
    u_n1_tensor = torch.tensor(u_n1, dtype=torch.float64, device=DEVICE)

    A, b = get_implicit_rk_coeffs(q)
    model = PINNDiscreteDiscovery(layers, q, dt, A, b, lb=float(x_raw.min()), ub=float(x_raw.max())).to(DEVICE)

    # Entrenamiento conjunto mediante un único L-BFGS full-batch (fiel a Raissi et al.)
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
            print(f'  L-BFGS Iter: {iter_info[0]:5d} | SSE Loss: {loss.item():.4e} | l1: {l1_val:.5f} | l2: {l2_val:.6f}')
            
        return loss

    print(f"Iniciando descubrimiento discreto (dt={dt:.1f}, Ruido={noise_level*100}%)...")
    optimizer.step(closure)
    print(f"Entrenamiento finalizado en {iter_info[0]} iteraciones.")

    # ==========================================
    # 4. EVALUACIÓN Y RESULTADOS
    # ==========================================
    lambda_1_pred = model.lambda_1.item()
    lambda_2_pred = torch.exp(model.lambda_2).item()

    error_l1 = np.abs(lambda_1_pred - lambda_1_exact) / lambda_1_exact * 100
    error_l2 = np.abs(lambda_2_pred - lambda_2_exact) / lambda_2_exact * 100

    print(f"\n--- Resultados del Descubrimiento Discreto (Ruido: {noise_level*100}%) ---")
    print(f"Lambda 1 exacto: {lambda_1_exact:.6f}  | Predicho: {lambda_1_pred:.6f}  | Error: {error_l1:.3f}%")
    print(f"Lambda 2 exacto: {lambda_2_exact:.6f} | Predicho: {lambda_2_pred:.6f} | Error: {error_l2:.3f}%")

    # ==========================================
    # 5. VISUALIZACIÓN EN DOS VENTANAS SEPARADAS
    # ==========================================
    model.eval()
    x_eval_tensor = torch.tensor(x_raw, dtype=torch.float64, requires_grad=True, device=DEVICE)
    U_n_pred_full, U_n1_pred_full = model.net_residuals(x_eval_tensor)

    u_n_pred_profile = U_n_pred_full.detach().cpu().numpy().mean(axis=1)
    u_n1_pred_profile = U_n1_pred_full.detach().cpu().numpy().mean(axis=1)

    # VENTANA 1: Mapa de calor espaciotemporal superior (Fig. B.9 Top)
    plt.figure(figsize=(9, 4.5))
    h = plt.imshow(Exact, interpolation='nearest', cmap='rainbow', 
                   extent=[t_raw.min(), t_raw.max(), x_raw.min(), x_raw.max()], 
                   origin='lower', aspect='auto')
    plt.colorbar(h)
    plt.axvline(x=t_n_val, color='white', linewidth=2, linestyle='--', label=f'$t^n = {t_n_val:.2f}$')
    plt.axvline(x=t_n1_val, color='white', linewidth=2, linestyle='--', label=f'$t^{{n+1}} = {t_n1_val:.2f}$')
    plt.xlabel('$t$')
    plt.ylabel('$x$')
    plt.title(f'Mapa Espaciotemporal $u(t, x)$ y Snapshots (Ruido: {noise_level*100}%)', fontsize=12)
    plt.legend(loc='upper right', frameon=True)
    plt.tight_layout()
    plt.show()

    # VENTANA 2: Curvas de predicción en t = 0.10 y t = 0.90 (Fig. B.9 Middle)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].plot(x_raw, Exact[:, idx_t0], 'b-', linewidth=2, label='Exact')
    axes[0].plot(x_raw, u_n_pred_profile, 'r--', linewidth=2, label='Prediction')
    axes[0].plot(x_n, u_n, 'kx', markersize=4, label=f'Data ({N_n} points)')
    axes[0].set_xlabel('$x$')
    axes[0].set_ylabel('$u(t,x)$')
    axes[0].set_title(f'$t = {t_n_val:.2f}$ (Ruido: {noise_level*100}%)\n{N_n} training data')
    axes[0].legend(loc='upper right', frameon=True)
    axes[0].set_xlim([-1, 1])
    axes[0].set_ylim([-1.1, 1.1])

    axes[1].plot(x_raw, Exact[:, idx_t1], 'b-', linewidth=2, label='Exact')
    axes[1].plot(x_raw, u_n1_pred_profile, 'r--', linewidth=2, label='Prediction')
    axes[1].plot(x_n1, u_n1, 'kx', markersize=4, label=f'Data ({N_n1} points)')
    axes[1].set_xlabel('$x$')
    axes[1].set_ylabel('$u(t,x)$')
    axes[1].set_title(f'$t = {t_n1_val:.2f}$ (Ruido: {noise_level*100}%)\n{N_n1} training data')
    axes[1].legend(loc='upper right', frameon=True)
    axes[1].set_xlim([-1, 1])
    axes[1].set_ylim([-0.8, 0.8])

    plt.tight_layout()
    plt.show()