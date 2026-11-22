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

print(f"Ejecutando exploración de hiperparámetros en: {DEVICE} (SEED = {SEED})")

CSV_PATH = 'burgers_tabla_hiperparametros.csv'

# ==========================================
# 1. CARGA DE DATOS
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

x_u = torch.tensor(X_u_train[:, 0:1], dtype=torch.float32, device=DEVICE)
t_u = torch.tensor(X_u_train[:, 1:2], dtype=torch.float32, device=DEVICE)
u_target = torch.tensor(u_train, dtype=torch.float32, device=DEVICE)

x_star_t = torch.tensor(X_star[:, 0:1], dtype=torch.float32, device=DEVICE)
t_star_t = torch.tensor(X_star[:, 1:2], dtype=torch.float32, device=DEVICE)

def get_collocation_points(strategy, n_f=10000):
    set_seed(SEED)
    if strategy == 'LHS':
        sampler = qmc.LatinHypercube(d=2, seed=SEED)
        pts = sampler.random(n=n_f)
        pts[:, 0] = -1.0 + 2.0 * pts[:, 0]
    elif strategy == 'Uniform':
        pts = np.random.rand(n_f, 2)
        pts[:, 0] = -1.0 + 2.0 * pts[:, 0]
    elif strategy == 'Grid':
        nx, nt = int(np.sqrt(n_f)), int(np.sqrt(n_f))
        xg, tg = np.meshgrid(np.linspace(-1.0, 1.0, nx), np.linspace(0.0, 1.0, nt))
        pts = np.hstack((xg.flatten()[:, None], tg.flatten()[:, None]))
    pts = np.vstack((pts, X_u_train))
    return (torch.tensor(pts[:, 0:1], dtype=torch.float32, requires_grad=True, device=DEVICE),
            torch.tensor(pts[:, 1:2], dtype=torch.float32, requires_grad=True, device=DEVICE))

# ==========================================
# 2. MODELO PARAMETRIZABLE
# ==========================================
class PINN_Ablacion(nn.Module):
    def __init__(self, layers, act_name='tanh'):
        super().__init__()
        self.linears = nn.ModuleList([nn.Linear(layers[i], layers[i+1]) for i in range(len(layers)-1)])
        self.act_name = act_name
        self.nu = 0.01 / np.pi

    def act(self, z):
        if self.act_name == 'tanh': return torch.tanh(z)
        if self.act_name == 'sin':  return torch.sin(z)
        if self.act_name == 'silu': return torch.nn.functional.silu(z)
        if self.act_name == 'relu': return torch.relu(z)

    def forward(self, x, t):
        inputs = torch.cat([x, t], dim=1)
        for i in range(len(self.linears) - 1):
            inputs = self.act(self.linears[i](inputs))
        return self.linears[-1](inputs)

    def net_f(self, x, t):
        u = self.forward(x, t)
        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), retain_graph=True, create_graph=True)[0]
        return u_t + u * u_x - self.nu * u_xx

class DivergenceError(Exception):
    pass

# ==========================================
# 3. EJECUCIÓN CON AUTO-RECUPERACIÓN Y CACHÉ
# ==========================================
layers = [2, 20, 20, 20, 20, 20, 20, 20, 20, 1]
cache_tecnica = {}  # Evita recalcular configuraciones matemáticamente idénticas

def run_experiment(idx_exp, total_exp, familia, nombre, opt_type='LBFGS', act_name='tanh', sampling='LHS', w_u=1.0, w_f=1.0):
    clave_tecnica = (opt_type, act_name, sampling, w_u, w_f)
    if clave_tecnica in cache_tecnica:
        res_prev = cache_tecnica[clave_tecnica]
        print(f"\n[{idx_exp}/{total_exp}] ⚡ [{familia} -> {nombre}] Configuración idéntica ya calculada en memoria. "
              f"Reutilizando -> MSE_u: {res_prev['MSE_u']} | MSE_f: {res_prev['MSE_f']} | L2: {res_prev['Error_L2']}", flush=True)
        return {
            'Variable': familia,
            'Configuracion': nombre,
            'MSE_u': res_prev['MSE_u'],
            'MSE_f': res_prev['MSE_f'],
            'Error_L2': res_prev['Error_L2']
        }

    print(f"\n[{idx_exp}/{total_exp}] Iniciando [{familia} -> {nombre}] (Opt: {opt_type}, Act: {act_name}, Muestreo: {sampling})...", flush=True)
    t_start = time.time()
    
    x_f, t_f = get_collocation_points(sampling, N_f)
    set_seed(SEED)
    model = PINN_Ablacion(layers, act_name=act_name).to(DEVICE)

    try:
        if opt_type in ['Adam', 'Hybrid']:
            n_adam = 10000 if opt_type == 'Adam' else 2000
            print(f"   -> Entrenando con Adam ({n_adam} épocas)...", flush=True)
            opt_adam = torch.optim.Adam(model.parameters(), lr=1e-3)
            for epoch in range(1, n_adam + 1):
                opt_adam.zero_grad()
                mse_u = torch.mean((model(x_u, t_u) - u_target)**2)
                mse_f = torch.mean(model.net_f(x_f, t_f)**2)
                loss = w_u * mse_u + w_f * mse_f
                
                if not torch.isfinite(loss):
                    raise DivergenceError(f"Loss NaN/Inf en época Adam {epoch}")
                
                loss.backward()
                opt_adam.step()
                
                if epoch % 500 == 0 or epoch == n_adam:
                    print(f"      [Adam Época {epoch}/{n_adam}] Loss: {loss.item():.5e}", flush=True)

        if opt_type in ['LBFGS', 'Hybrid']:
            max_lbfgs = 10000 if opt_type == 'LBFGS' else 8000
            print(f"   -> Entrenando con L-BFGS (máx. {max_lbfgs} iteraciones)...", flush=True)
            opt_lbfgs = torch.optim.LBFGS(
                model.parameters(), lr=1.0, max_iter=max_lbfgs, max_eval=max_lbfgs,
                history_size=50, tolerance_grad=1e-5,
                tolerance_change=1.0 * np.finfo(float).eps, line_search_fn="strong_wolfe"
            )
            
            iter_lbfgs = [0]
            def closure():
                opt_lbfgs.zero_grad()
                mse_u = torch.mean((model(x_u, t_u) - u_target)**2)
                mse_f = torch.mean(model.net_f(x_f, t_f)**2)
                loss = w_u * mse_u + w_f * mse_f
                
                iter_lbfgs[0] += 1
                if not torch.isfinite(loss):
                    raise DivergenceError(f"Loss NaN/Inf en iteración L-BFGS {iter_lbfgs[0]}")
                
                loss.backward()
                
                for p in model.parameters():
                    if p.grad is not None and not torch.all(torch.isfinite(p.grad)):
                        raise DivergenceError(f"Gradiente NaN/Inf en iteración L-BFGS {iter_lbfgs[0]}")
                
                if iter_lbfgs[0] % 500 == 0 or iter_lbfgs[0] == 1:
                    print(f"      [L-BFGS Iter {iter_lbfgs[0]}/{max_lbfgs}] Loss: {loss.item():.5e}", flush=True)
                return loss
                
            opt_lbfgs.step(closure)

        model.eval()
        with torch.no_grad():
            mse_u_val = torch.mean((model(x_u, t_u) - u_target)**2).item()
            u_pred = model(x_star_t, t_star_t).detach().cpu().numpy()
        mse_f_val = torch.mean(model.net_f(x_f, t_f)**2).item()
        err_l2 = np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)

        if not (np.isfinite(mse_u_val) and np.isfinite(mse_f_val) and np.isfinite(err_l2)):
            raise DivergenceError("Métricas finales contienen NaN/Inf.")

        elapsed_min = (time.time() - t_start) / 60.0
        print(f" ✔️ Completado [{familia} -> {nombre}] en {elapsed_min:.2f} min | MSE_u: {mse_u_val:.2e} | MSE_f: {mse_f_val:.2e} | L2: {err_l2:.2e}", flush=True)
        
        res_dict = {
            'Variable': familia,
            'Configuracion': nombre,
            'MSE_u': f"{mse_u_val:.2e}",
            'MSE_f': f"{mse_f_val:.2e}",
            'Error_L2': f"{err_l2:.2e}"
        }
        cache_tecnica[clave_tecnica] = res_dict
        return res_dict

    except (DivergenceError, RuntimeError, Exception) as e:
        elapsed_min = (time.time() - t_start) / 60.0
        print(f" ⚠️ [{familia} -> {nombre}] No convergió ({e}) tras {elapsed_min:.2f} min. Registrando divergencia y siguiendo...", flush=True)
        res_dict = {
            'Variable': familia,
            'Configuracion': nombre,
            'MSE_u': 'Diverge (NaN)',
            'MSE_f': 'Diverge (NaN)',
            'Error_L2': 'Diverge (NaN)'
        }
        cache_tecnica[clave_tecnica] = res_dict
        return res_dict

experimentos = [
    # 1. Optimizador
    ('Optimizador', 'Solo Adam', 'Adam', 'tanh', 'LHS', 1.0, 1.0),
    ('Optimizador', 'Solo L-BFGS', 'LBFGS', 'tanh', 'LHS', 1.0, 1.0),
    ('Optimizador', 'Híbrido (Adam + L-BFGS)', 'Hybrid', 'tanh', 'LHS', 1.0, 1.0),
    # 2. Función de Activación
    ('Función de Activación', 'tanh(x)', 'LBFGS', 'tanh', 'LHS', 1.0, 1.0),
    ('Función de Activación', 'sin(x)', 'LBFGS', 'sin', 'LHS', 1.0, 1.0),
    ('Función de Activación', 'SiLU(x)', 'LBFGS', 'silu', 'LHS', 1.0, 1.0),
    ('Función de Activación', 'ReLU(x)', 'LBFGS', 'relu', 'LHS', 1.0, 1.0),
    # 3. Muestreo de Colocación
    ('Muestreo de Colocación', 'Malla Equiespaciada', 'LBFGS', 'tanh', 'Grid', 1.0, 1.0),
    ('Muestreo de Colocación', 'Aleatorio Uniforme', 'LBFGS', 'tanh', 'Uniform', 1.0, 1.0),
    ('Muestreo de Colocación', 'Hipercubo Latino (LHS)', 'LBFGS', 'tanh', 'LHS', 1.0, 1.0),
    # 4. Pesos (w_u : w_f)
    ('Pesos (w_u : w_f)', '1 : 1 (Estándar)', 'LBFGS', 'tanh', 'LHS', 1.0, 1.0),
    ('Pesos (w_u : w_f)', '10 : 1', 'LBFGS', 'tanh', 'LHS', 10.0, 1.0),
    ('Pesos (w_u : w_f)', '1 : 10', 'LBFGS', 'tanh', 'LHS', 1.0, 10.0),
]

# ==========================================
# 4. GESTIÓN AUTOMÁTICA DE PROGRESO (CHECKPOINT)
# ==========================================
total_exp = len(experimentos)
completados = {}

if os.path.exists(CSV_PATH):
    try:
        df_prev = pd.read_csv(CSV_PATH)
        for _, row in df_prev.iterrows():
            par = (row['Variable'], row['Configuracion'])
            if pd.notna(row['Error_L2']):
                completados[par] = row.to_dict()
                for exp in experimentos:
                    if (exp[0], exp[1]) == par:
                        clave_tec = (exp[2], exp[3], exp[4], exp[5], exp[6])
                        cache_tecnica[clave_tec] = row.to_dict()
        print(f"📂 Archivo '{CSV_PATH}' detectado: {len(completados)}/{total_exp} experimentos ya guardados.", flush=True)
    except Exception as e:
        print(f"Aviso al leer CSV previo: {e}. Se creará uno nuevo.", flush=True)

lista_res = []
for i, exp in enumerate(experimentos, 1):
    par_actual = (exp[0], exp[1])
    if par_actual in completados:
        res = completados[par_actual]
        print(f"[{i}/{total_exp}] ⏭️ Saltando [{exp[0]} -> {exp[1]}] (ya guardado en CSV: L2 = {res['Error_L2']})", flush=True)
        lista_res.append(res)
    else:
        res = run_experiment(i, total_exp, *exp)
        lista_res.append(res)
        completados[par_actual] = res
        pd.DataFrame(lista_res).to_csv(CSV_PATH, index=False)

df_hip = pd.DataFrame(lista_res)
print("\n" + "="*55)
print("=== RESULTADOS FINALES EXPLORACIÓN DE HIPERPARÁMETROS ===")
print("="*55)
print(df_hip.to_string(index=False))
print(f"\nTodos los resultados están guardados de forma segura en '{CSV_PATH}'.")