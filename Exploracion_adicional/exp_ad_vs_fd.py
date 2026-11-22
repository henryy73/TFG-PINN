import os
import time
import random
import warnings
import torch
import torch.nn as nn
import numpy as np
import scipy.io
import pandas as pd
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

print(f"Ejecutando comparativa AD vs FD en: {DEVICE} (SEED = {SEED})")

CSV_PATH = 'burgers_tabla_AD_vs_FD.csv'

# ==========================================
# 1. CARGA DE DATOS Y MUESTREO BASE
# ==========================================
MAT_PATH = r'C:\Users\ematm\OneDrive\Desktop\TFG\Trabajo TFG propuesto\data\burgers_shock.mat'
if not os.path.exists(MAT_PATH):
    MAT_PATH = 'burgers_shock.mat'
    if not os.path.exists(MAT_PATH):
        raise FileNotFoundError(f"No se encontró el archivo .mat en {MAT_PATH} ni en el directorio actual.")

data = scipy.io.loadmat(MAT_PATH)
x = data['x']
t = data['t']
Exact = np.real(data['usol'])

X, T = np.meshgrid(x.flatten(), t.flatten(), indexing='ij')
X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
u_star = Exact.flatten()[:, None]

# Condición inicial (t = 0) y condiciones de contorno (x = -1, x = 1)
xx1 = np.hstack((X[:, 0:1], T[:, 0:1]))
uu1 = Exact[:, 0:1]
xx2 = np.hstack((X[0:1, :].T, T[0:1, :].T))
uu2 = Exact[0:1, :].T
xx3 = np.hstack((X[-1:, :].T, T[-1:, :].T))
uu3 = Exact[-1:, :].T

X_u_all = np.vstack([xx1, xx2, xx3])
u_all = np.vstack([uu1, uu2, uu3])

set_seed(SEED)
N_u = 100
N_f = 10000
idx = np.random.choice(X_u_all.shape[0], N_u, replace=False)
X_u_train = X_u_all[idx, :]
u_train = u_all[idx, :]

# Puntos de colocación por Hipercubo Latino (LHS) en [-1, 1] x [0, 1]
set_seed(SEED)
sampler = qmc.LatinHypercube(d=2, seed=SEED)
lhs_samples = sampler.random(n=N_f)
X_f_train = np.zeros_like(lhs_samples)
X_f_train[:, 0] = -1.0 + 2.0 * lhs_samples[:, 0]  # x en [-1, 1]
X_f_train[:, 1] = lhs_samples[:, 1]               # t en [0, 1]
X_f_train = np.vstack((X_f_train, X_u_train))

x_u = torch.tensor(X_u_train[:, 0:1], dtype=torch.float32, device=DEVICE)
t_u = torch.tensor(X_u_train[:, 1:2], dtype=torch.float32, device=DEVICE)
u_target = torch.tensor(u_train, dtype=torch.float32, device=DEVICE)

x_f = torch.tensor(X_f_train[:, 0:1], dtype=torch.float32, requires_grad=True, device=DEVICE)
t_f = torch.tensor(X_f_train[:, 1:2], dtype=torch.float32, requires_grad=True, device=DEVICE)
x_star_t = torch.tensor(X_star[:, 0:1], dtype=torch.float32, device=DEVICE)
t_star_t = torch.tensor(X_star[:, 1:2], dtype=torch.float32, device=DEVICE)

# ==========================================
# 2. DEFINICIÓN DEL MODELO CON AD O FD
# ==========================================
class PINN_Derivadas(nn.Module):
    def __init__(self, layers, mode='AD', h=1e-3):
        super().__init__()
        self.linears = nn.ModuleList([nn.Linear(layers[i], layers[i+1]) for i in range(len(layers)-1)])
        self.activation = nn.Tanh()
        self.mode = mode
        self.h = h
        self.nu = 0.01 / np.pi

    def forward(self, x, t):
        inputs = torch.cat([x, t], dim=1)
        for i in range(len(self.linears) - 1):
            inputs = self.activation(self.linears[i](inputs))
        return self.linears[-1](inputs)

    def net_f(self, x, t):
        u = self.forward(x, t)
        if self.mode == 'AD':
            u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
            u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
            u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), retain_graph=True, create_graph=True)[0]
        else:
            # Esquemas centrados de Diferencias Finitas de segundo orden
            h = self.h
            u_t = (self.forward(x, t + h) - self.forward(x, t - h)) / (2.0 * h)
            u_x_plus = self.forward(x + h, t)
            u_x_minus = self.forward(x - h, t)
            u_x = (u_x_plus - u_x_minus) / (2.0 * h)
            u_xx = (u_x_plus - 2.0 * u + u_x_minus) / (h ** 2)

        return u_t + u * u_x - self.nu * u_xx

# ==========================================
# 3. BUCLE EXPERIMENTAL (AD vs FD) CON CHECKPOINT
# ==========================================
layers = [2, 20, 20, 20, 20, 20, 20, 20, 20, 1]
configs = [
    ('Diferenciación Automática', 'AD', 0.0),
    ('Diferencias Finitas', 'FD', 1e-2),
    ('Diferencias Finitas', 'FD', 1e-3),
    ('Diferencias Finitas', 'FD', 1e-4),
    ('Diferencias Finitas', 'FD', 1e-5),
]

completados = {}
if os.path.exists(CSV_PATH):
    try:
        df_prev = pd.read_csv(CSV_PATH)
        for _, row in df_prev.iterrows():
            clave = (str(row['Metodo']), str(row['Paso_h']))
            if pd.notna(row['Error_L2']):
                completados[clave] = row.to_dict()
        print(f"📂 Archivo '{CSV_PATH}' detectado: {len(completados)}/{len(configs)} configuraciones ya guardadas.", flush=True)
    except Exception as e:
        print(f"Aviso al leer CSV previo: {e}. Se creará uno nuevo.", flush=True)

resultados = []
total_configs = len(configs)

for idx_cfg, (nombre, modo, h_val) in enumerate(configs, 1):
    paso_str = "Exacta" if modo == 'AD' else f"{h_val:.0e}"
    clave_actual = (nombre, paso_str)

    if clave_actual in completados:
        res_guardado = completados[clave_actual]
        print(f"[{idx_cfg}/{total_configs}] ⏭️ Saltando {nombre} (h={paso_str}) -> ya en CSV (L2 = {res_guardado['Error_L2']})", flush=True)
        resultados.append(res_guardado)
        continue

    set_seed(SEED)
    model = PINN_Derivadas(layers, mode=modo, h=h_val).to(DEVICE)
    
    optimizer = torch.optim.LBFGS(
        model.parameters(), lr=1.0, max_iter=10000, max_eval=10000, history_size=50,
        tolerance_grad=1e-5, tolerance_change=1.0 * np.finfo(float).eps, line_search_fn="strong_wolfe"
    )

    iter_count = [0]

    def closure():
        optimizer.zero_grad()
        u_pred = model(x_u, t_u)
        f_pred = model.net_f(x_f, t_f)
        loss = torch.mean((u_pred - u_target)**2) + torch.mean(f_pred**2)
        loss.backward()
        
        iter_count[0] += 1
        if iter_count[0] % 100 == 0 or iter_count[0] == 1:
            print(f"   [{idx_cfg}/{total_configs}] {nombre} (h={paso_str}): "
                  f"Iteración L-BFGS {iter_count[0]} | Loss actual: {loss.item():.5e}", flush=True)
        return loss

    print(f"\n[{idx_cfg}/{total_configs}] Iniciando entrenamiento: {nombre} (h = {paso_str})...", flush=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()
    optimizer.step(closure)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.time() - t0

    model.eval()
    with torch.no_grad():
        u_pred_star = model(x_star_t, t_star_t).detach().cpu().numpy()
        u_pred_train = model(x_u, t_u)
        
    f_final = model.net_f(x_f, t_f)
    loss_final = (torch.mean((u_pred_train - u_target)**2) + torch.mean(f_final**2)).item()
    error_l2 = np.linalg.norm(u_star - u_pred_star, 2) / np.linalg.norm(u_star, 2)

    print(f" ✔️ Finalizado {nombre} (h={paso_str}) en {elapsed/60:.2f} min -> Loss Final: {loss_final:.3e} | Error L2: {error_l2:.3e}\n", flush=True)
    
    res_item = {
        'Metodo': nombre,
        'Paso_h': paso_str,
        'Perdida_L': f"{loss_final:.2e}",
        'Error_L2': f"{error_l2:.2e}",
        'Tiempo_s': f"{elapsed:.1f}"
    }
    resultados.append(res_item)
    pd.DataFrame(resultados).to_csv(CSV_PATH, index=False)

df_res = pd.DataFrame(resultados)
print("\n" + "="*40)
print("=== RESULTADOS FINALES TABLA AD vs FD ===")
print("="*40)
print(df_res.to_string(index=False))
df_res.to_csv(CSV_PATH, index=False)
print(f"\nResultados guardados correctamente en '{CSV_PATH}'.")