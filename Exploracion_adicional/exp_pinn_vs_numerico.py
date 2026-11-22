import os
import random
import warnings
import numpy as np
import scipy.io
from scipy.interpolate import RegularGridInterpolator
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import qmc

warnings.filterwarnings("ignore", category=UserWarning)

# ==========================================
# 0. CONFIGURACIÓN GLOBAL, CUDA Y SEMILLA (42)
# ==========================================
SEED = 42
torch.set_default_dtype(torch.float32)

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.cuda.init()
    _ = torch.matmul(torch.zeros((1, 1), dtype=torch.float32, device=DEVICE),
                     torch.zeros((1, 1), dtype=torch.float32, device=DEVICE))

print(f"Ejecutando comparativa PINN vs Numérico en: {DEVICE} (SEED = {SEED})")

CSV_PATH = 'burgers_tabla_PINN_vs_Numerico.csv'

# ==========================================
# 1. CARGA DE LA SOLUCIÓN EXACTA DE REFERENCIA
# ==========================================
MAT_PATH = r'C:\Users\ematm\OneDrive\Desktop\TFG\Trabajo TFG propuesto\data\burgers_shock.mat'
if not os.path.exists(MAT_PATH):
    MAT_PATH = 'burgers_shock.mat'
    if not os.path.exists(MAT_PATH):
        raise FileNotFoundError(f"No se encontró el archivo .mat en {MAT_PATH} ni en el directorio actual.")

data = scipy.io.loadmat(MAT_PATH)
x_ref = data['x'].flatten()       # 256 puntos en [-1, 1]
t_ref = data['t'].flatten()       # 100 puntos en [0, 0.99]
Exact = np.real(data['usol'])     # (256, 100)

idx_t025 = np.argmin(np.abs(t_ref - 0.25))
idx_t075 = np.argmin(np.abs(t_ref - 0.75))

# ==========================================
# 2. RESOLUTOR NUMÉRICO CLÁSICO (FD + RK4)
# ==========================================
def solve_burgers_fd_rk4(Nx, Nt):
    x_grid = np.linspace(-1.0, 1.0, Nx)
    t_grid = np.linspace(0.0, 1.0, Nt)
    dx = x_grid[1] - x_grid[0]
    dt = t_grid[1] - t_grid[0]
    nu = 0.01 / np.pi

    U = np.zeros((Nx, Nt))
    U[:, 0] = -np.sin(np.pi * x_grid)

    def rhs(u):
        dudt = np.zeros_like(u)
        # Diferencias centradas en el interior (Dirichlet u=0 en los bordes)
        u_x = (u[2:] - u[:-2]) / (2.0 * dx)
        u_xx = (u[2:] - 2.0 * u[1:-1] + u[:-2]) / (dx ** 2)
        dudt[1:-1] = -u[1:-1] * u_x + nu * u_xx
        return dudt

    with np.errstate(all='ignore'):
        for n in range(Nt - 1):
            u_curr = U[:, n]
            k1 = rhs(u_curr)
            k2 = rhs(u_curr + 0.5 * dt * k1)
            k3 = rhs(u_curr + 0.5 * dt * k2)
            k4 = rhs(u_curr + dt * k3)
            U[:, n+1] = u_curr + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    if not np.all(np.isfinite(U)):
        U = np.nan_to_num(U, nan=0.0, posinf=1.0, neginf=-1.0)

    interp = RegularGridInterpolator((x_grid, t_grid), U, method='linear', bounds_error=False, fill_value=None)
    X_r, T_r = np.meshgrid(x_ref, t_ref, indexing='ij')
    pts = np.array([X_r.flatten(), T_r.flatten()]).T
    U_interp = interp(pts).reshape(Exact.shape)

    err_025 = np.linalg.norm(Exact[:, idx_t025] - U_interp[:, idx_t025], 2) / np.linalg.norm(Exact[:, idx_t025], 2)
    err_075 = np.linalg.norm(Exact[:, idx_t075] - U_interp[:, idx_t075], 2) / np.linalg.norm(Exact[:, idx_t075], 2)
    err_global = np.linalg.norm(Exact - U_interp, 'fro') / np.linalg.norm(Exact, 'fro')

    return err_025, err_075, err_global

# ==========================================
# 3. EVALUACIÓN DE MALLAS NUMÉRICAS Y PINN
# ==========================================
completados = {}
if os.path.exists(CSV_PATH):
    try:
        df_prev = pd.read_csv(CSV_PATH)
        for _, row in df_prev.iterrows():
            if pd.notna(row['Error_Global']):
                completados[row['Metodo']] = row.to_dict()
        print(f"📂 Archivo '{CSV_PATH}' detectado: {len(completados)} métodos ya guardados.", flush=True)
    except Exception as e:
        print(f"Aviso al leer CSV previo: {e}. Se recalculará.", flush=True)

resultados = []

mallas = [
    ('Numérico (Malla gruesa)', 64, 500),
    ('Numérico (Malla media)', 128, 1000),
    ('Numérico (Malla fina)', 512, 4000),
]

print("Calculando resolutores numéricos clásicos (FD + RK4)...", flush=True)
for nombre, nx, nt in mallas:
    if nombre in completados:
        print(f" -> [SKIP] {nombre} ya guardado en CSV (Global = {completados[nombre]['Error_Global']}).", flush=True)
        resultados.append(completados[nombre])
        continue

    print(f" -> Ejecutando {nombre} (Nx={nx}, Nt={nt})...", flush=True)
    e25, e75, eglob = solve_burgers_fd_rk4(nx, nt)
    res_item = {
        'Metodo': nombre,
        'Malla_Puntos': f"$N_x={nx}, N_t={nt}$",
        'Error_t025': f"{e25:.2e}",
        'Error_t075': f"{e75:.2e}",
        'Error_Global': f"{eglob:.2e}"
    }
    resultados.append(res_item)
    pd.DataFrame(resultados).to_csv(CSV_PATH, index=False)

# Entrenamiento del modelo PINN Base para extraer sus métricas en t=0.25 y t=0.75
nombre_pinn = 'PINN (Tiempo Continuo)'
if nombre_pinn in completados:
    print(f" -> [SKIP] {nombre_pinn} ya guardado en CSV (Global = {completados[nombre_pinn]['Error_Global']}).", flush=True)
    resultados.append(completados[nombre_pinn])
else:
    X_r, T_r = np.meshgrid(x_ref, t_ref, indexing='ij')
    xx1, uu1 = np.hstack((X_r[:, 0:1], T_r[:, 0:1])), Exact[:, 0:1]
    xx2, uu2 = np.hstack((X_r[0:1, :].T, T_r[0:1, :].T)), Exact[0:1, :].T
    xx3, uu3 = np.hstack((X_r[-1:, :].T, T_r[-1:, :].T)), Exact[-1:, :].T
    X_u_all, u_all = np.vstack([xx1, xx2, xx3]), np.vstack([uu1, uu2, uu3])

    set_seed(SEED)
    idx = np.random.choice(X_u_all.shape[0], 100, replace=False)
    X_u_train, u_train = X_u_all[idx, :], u_all[idx, :]

    set_seed(SEED)
    sampler = qmc.LatinHypercube(d=2, seed=SEED)
    pts = sampler.random(n=10000)
    pts[:, 0] = -1.0 + 2.0 * pts[:, 0]
    pts = np.vstack((pts, X_u_train))

    x_u = torch.tensor(X_u_train[:, 0:1], dtype=torch.float32, device=DEVICE)
    t_u = torch.tensor(X_u_train[:, 1:2], dtype=torch.float32, device=DEVICE)
    u_target = torch.tensor(u_train, dtype=torch.float32, device=DEVICE)
    x_f = torch.tensor(pts[:, 0:1], dtype=torch.float32, requires_grad=True, device=DEVICE)
    t_f = torch.tensor(pts[:, 1:2], dtype=torch.float32, requires_grad=True, device=DEVICE)

    class PINNBase(nn.Module):
        def __init__(self, layers):
            super().__init__()
            self.linears = nn.ModuleList([nn.Linear(layers[i], layers[i+1]) for i in range(len(layers)-1)])
        def forward(self, x, t):
            z = torch.cat([x, t], dim=1)
            for i in range(len(self.linears) - 1):
                z = torch.tanh(self.linears[i](z))
            return self.linears[-1](z)
        def net_f(self, x, t):
            u = self.forward(x, t)
            u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
            u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
            u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), retain_graph=True, create_graph=True)[0]
            return u_t + u * u_x - (0.01 / np.pi) * u_xx

    set_seed(SEED)
    model = PINNBase([2, 20, 20, 20, 20, 20, 20, 20, 20, 1]).to(DEVICE)
    opt = torch.optim.LBFGS(
        model.parameters(), lr=1.0, max_iter=10000, max_eval=10000,
        history_size=50, tolerance_grad=1e-5,
        tolerance_change=1.0 * np.finfo(float).eps, line_search_fn="strong_wolfe"
    )

    iter_count = [0]
    def closure():
        opt.zero_grad()
        loss = torch.mean((model(x_u, t_u) - u_target)**2) + torch.mean(model.net_f(x_f, t_f)**2)
        loss.backward()
        
        iter_count[0] += 1
        if iter_count[0] % 500 == 0 or iter_count[0] == 1:
            print(f"   [PINN L-BFGS Iter {iter_count[0]}/10000] Loss: {loss.item():.5e}", flush=True)
        return loss

    print("\nEntrenando PINN (Tiempo Continuo) para comparar en t=0.25 y t=0.75...", flush=True)
    opt.step(closure)

    model.eval()
    with torch.no_grad():
        X_star = np.hstack((X_r.flatten()[:, None], T_r.flatten()[:, None]))
        u_pinn = model(
            torch.tensor(X_star[:, 0:1], dtype=torch.float32, device=DEVICE),
            torch.tensor(X_star[:, 1:2], dtype=torch.float32, device=DEVICE)
        ).detach().cpu().numpy().reshape(Exact.shape)

    e25_pinn = np.linalg.norm(Exact[:, idx_t025] - u_pinn[:, idx_t025], 2) / np.linalg.norm(Exact[:, idx_t025], 2)
    e75_pinn = np.linalg.norm(Exact[:, idx_t075] - u_pinn[:, idx_t075], 2) / np.linalg.norm(Exact[:, idx_t075], 2)
    eglob_pinn = np.linalg.norm(Exact - u_pinn, 'fro') / np.linalg.norm(Exact, 'fro')

    res_pinn = {
        'Metodo': nombre_pinn,
        'Malla_Puntos': '$N_u=100, N_f=10000$',
        'Error_t025': f"{e25_pinn:.2e}",
        'Error_t075': f"{e75_pinn:.2e}",
        'Error_Global': f"{eglob_pinn:.2e}"
    }
    resultados.append(res_pinn)
    pd.DataFrame(resultados).to_csv(CSV_PATH, index=False)

df_comp = pd.DataFrame(resultados)
print("\n" + "="*50)
print("=== RESULTADOS PINN vs NUMÉRICO CLÁSICO ===")
print("="*50)
print(df_comp.to_string(index=False))
df_comp.to_csv(CSV_PATH, index=False)
print(f"\nResultados guardados correctamente en '{CSV_PATH}'.")