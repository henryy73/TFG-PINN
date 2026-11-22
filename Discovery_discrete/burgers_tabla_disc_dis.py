import os
import time
import random
import warnings
import torch
import torch.nn as nn
import numpy as np
import scipy.io
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

# ==========================================
# 0. CONFIGURACIÓN GLOBAL, CUDA Y SEMILLA (42)
# ==========================================
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

print(f"Ejecutando sensibilidad discreta (Tablas B.8 y B.9) en: {DEVICE}")

# Valores exactos de referencia (Sección B.3)
LAMBDA_1_EXACT = 1.0
LAMBDA_2_EXACT = 0.01 / np.pi

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

        # Inicialización Truncated Normal de Xavier idéntica a Raissi et al.
        for m in self.linears:
            in_dim, out_dim = m.weight.shape[1], m.weight.shape[0]
            std = np.sqrt(2.0 / (in_dim + out_dim))
            nn.init.trunc_normal_(m.weight, mean=0.0, std=std, a=-2.0*std, b=2.0*std)
            nn.init.zeros_(m.bias)

        self.lambda_1 = nn.Parameter(torch.tensor([0.0], dtype=torch.float64))
        self.lambda_2 = nn.Parameter(torch.tensor([-6.0], dtype=torch.float64))

    def forward(self, x):
        H = 2.0 * (x - self.lb) / (self.ub - self.lb) - 1.0
        for i in range(len(self.linears) - 1):
            H = self.activation(self.linears[i](H))
        return self.linears[-1](H)

    def net_residuals(self, x):
        U_stages = self.forward(x)
        N_u = []
        l2 = torch.exp(self.lambda_2)
        
        for j in range(self.q):
            u_j = U_stages[:, j:j+1]
            u_j_x = torch.autograd.grad(u_j.sum(), x, retain_graph=True, create_graph=True)[0]
            u_j_xx = torch.autograd.grad(u_j_x.sum(), x, retain_graph=True, create_graph=True)[0]
            n_j = self.lambda_1 * u_j * u_j_x - l2 * u_j_xx
            N_u.append(n_j)
            
        N_u_tensor = torch.cat(N_u, dim=1)
        U_n = U_stages + self.dt * torch.matmul(N_u_tensor, self.A.T)
        A_minus_b = self.A - self.b.unsqueeze(0)
        U_n_plus_1 = U_stages + self.dt * torch.matmul(N_u_tensor, A_minus_b.T)
        return U_n, U_n_plus_1

def loss_function(model, x_n, u_n_train, x_n1, u_n1_train):
    # Ecuación (B.9) del artículo: Suma de Errores Cuadráticos (SSE)
    U_n_pred, _ = model.net_residuals(x_n)
    _, U_n1_pred = model.net_residuals(x_n1)
    sse_n = torch.sum((U_n_pred - u_n_train) ** 2)
    sse_n1 = torch.sum((U_n1_pred - u_n1_train) ** 2)
    return sse_n + sse_n1

# ==========================================
# 3. FUNCIÓN DE ENTRENAMIENTO (SOLO L-BFGS)
# ==========================================
def run_discrete_experiment(x_raw, t_raw, Exact, dt_target, noise_level, num_hidden_layers, num_neurons, q_fixed=32):
    set_seed(42)
    t_start = time.time()
    
    t0_target = 0.1
    t1_target = t0_target + dt_target
    idx_t0 = np.argmin(np.abs(t_raw.flatten() - t0_target))
    idx_t1 = np.argmin(np.abs(t_raw.flatten() - t1_target))
    dt = float(t_raw[idx_t1, 0] - t_raw[idx_t0, 0])
    
    # Ecuación (B.11) acotada a q_fixed=32 etapas
    eps = np.finfo(float).eps
    q_paper = int(np.ceil(0.5 * np.log(eps) / np.log(dt)))
    q = min(q_paper, q_fixed)
    
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
    
    layers = [1] + [num_neurons] * num_hidden_layers + [q]
    A, b = get_implicit_rk_coeffs(q)
    model = PINNDiscreteDiscovery(layers, q, dt, A, b, lb=float(x_raw.min()), ub=float(x_raw.max())).to(DEVICE)

    # Entrenamiento conjunto L-BFGS (10000 iteraciones máx.)
    optimizer = torch.optim.LBFGS(
        model.parameters(), lr=1.0, max_iter=10000, max_eval=10000, history_size=50,
        tolerance_grad=1e-8, tolerance_change=1.0 * np.finfo(float).eps, line_search_fn="strong_wolfe"
    )
    
    iter_count = [0]
    def closure():
        optimizer.zero_grad()
        loss = loss_function(model, x_n_tensor, u_n_tensor, x_n1_tensor, u_n1_tensor)
        loss.backward()
        iter_count[0] += 1
        if iter_count[0] % 500 == 0:
            l1_val = model.lambda_1.item()
            l2_val = torch.exp(model.lambda_2).item()
            print(f"      [Iter {iter_count[0]:5d}] SSE Loss: {loss.item():.4e} | l1: {l1_val:.5f} | l2: {l2_val:.6f}", flush=True)
        return loss
        
    optimizer.step(closure)
    elapsed = time.time() - t_start
    
    lambda_1_pred = model.lambda_1.item()
    lambda_2_pred = torch.exp(model.lambda_2).item()
    
    error_l1 = np.abs(lambda_1_pred - LAMBDA_1_EXACT) / LAMBDA_1_EXACT * 100.0
    error_l2 = np.abs(lambda_2_pred - LAMBDA_2_EXACT) / LAMBDA_2_EXACT * 100.0
    
    print(f"    -> Fin (q={q}, {iter_count[0]} iters en {elapsed:.1f}s) | l1: {lambda_1_pred:.5f} (Err: {error_l1:.3f}%) | l2: {lambda_2_pred:.6f} (Err: {error_l2:.3f}%)", flush=True)
    return error_l1, error_l2

def print_latex_rows(df, title):
    print(f"\n--- FILAS LATEX PARA {title} ---")
    for idx, row in df.iterrows():
        vals = " & ".join([f"\\texttt{{{v:.3f}}}" for v in row.values])
        print(f"\\textbf{{{idx}}} & {vals} \\\\")

# ==========================================
# 4. EJECUCIÓN DE TABLAS B.8 Y B.9
# ==========================================
if __name__ == "__main__":
    
    MAT_PATH = r'C:\Users\ematm\OneDrive\Desktop\TFG\Trabajo TFG propuesto\data\burgers_shock.mat'
    if not os.path.exists(MAT_PATH):
        MAT_PATH = 'burgers_shock.mat'
        if not os.path.exists(MAT_PATH):
            raise FileNotFoundError(f"No se encontró el archivo .mat en {MAT_PATH} ni en el directorio actual.")
            
    data = scipy.io.loadmat(MAT_PATH)
    x_raw = data['x']
    t_raw = data['t']
    Exact = np.real(data['usol'])

    # ---------------------------------------------------------
    # TABLA B.8: Paso de tiempo dt vs Nivel de Ruido
    # ---------------------------------------------------------
    dt_list = [0.2, 0.4, 0.6, 0.8]
    noise_list = [0.0, 0.01, 0.05, 0.10]
    cols_B8 = [f"l1_{int(n*100)}%" for n in noise_list] + [f"l2_{int(n*100)}%" for n in noise_list]
    file_B8 = "burgers_tabla_B8.csv"

    df_B8 = pd.DataFrame(index=dt_list, columns=cols_B8, dtype=float)
    df_B8.index.name = "dt"

    if os.path.exists(file_B8):
        try:
            df_existing_B8 = pd.read_csv(file_B8, index_col=0)
            df_existing_B8.index = df_existing_B8.index.astype(float)
            df_B8.update(df_existing_B8)
            print(f"\n[INFO] Se ha cargado el progreso existente de '{file_B8}'.")
        except Exception as e:
            print(f"\n[AVISO] No se pudo leer '{file_B8}' correctamente ({e}). Se iniciará desde cero.")

    if not df_B8.isna().any().any():
        print(f"[OK] '{file_B8}' ya está completamente calculada. Se omite su ejecución.")
    else:
        print("\n========================================================")
        print("Calculando Tabla B.8 (dt vs Ruido - 16 experimentos)...")
        print("========================================================")
        exp_idx = 0
        total_B8 = len(dt_list) * len(noise_list)
        for dt_val in dt_list:
            for noise in noise_list:
                exp_idx += 1
                col_l1 = f"l1_{int(noise*100)}%"
                col_l2 = f"l2_{int(noise*100)}%"

                if pd.notna(df_B8.loc[dt_val, col_l1]) and pd.notna(df_B8.loc[dt_val, col_l2]):
                    print(f"[{exp_idx}/{total_B8}] [SKIP] B.8 -> dt={dt_val}, Ruido={int(noise*100)}% ya completado "
                          f"(l1={df_B8.loc[dt_val, col_l1]:.3f}%, l2={df_B8.loc[dt_val, col_l2]:.3f}%).", flush=True)
                    continue

                print(f"\n[{exp_idx}/{total_B8}] Ejecutando B.8 -> dt={dt_val}, Ruido={int(noise*100)}%...", flush=True)
                e1, e2 = run_discrete_experiment(x_raw, t_raw, Exact, dt_target=dt_val, noise_level=noise, 
                                                 num_hidden_layers=4, num_neurons=50, q_fixed=32)
                df_B8.loc[dt_val, col_l1] = e1
                df_B8.loc[dt_val, col_l2] = e2
                df_B8.to_csv(file_B8)

    print("\n=== TABLA B.8 ===")
    print(df_B8.round(3))
    print_latex_rows(df_B8, "TABLA B.8")

    # ---------------------------------------------------------
    # TABLA B.9: Capas ocultas vs Neuronas (Fijo: dt=0.8, Ruido=0%)
    # ---------------------------------------------------------
    layers_B9 = [1, 2, 3, 4]
    neurons_B9 = [10, 25, 50]
    cols_B9 = [f"l1_{neu}n" for neu in neurons_B9] + [f"l2_{neu}n" for neu in neurons_B9]
    file_B9 = "burgers_tabla_B9.csv"

    df_B9 = pd.DataFrame(index=layers_B9, columns=cols_B9, dtype=float)
    df_B9.index.name = "Capas"

    if os.path.exists(file_B9):
        try:
            df_existing_B9 = pd.read_csv(file_B9, index_col=0)
            df_existing_B9.index = df_existing_B9.index.astype(int)
            df_B9.update(df_existing_B9)
            print(f"\n[INFO] Se ha cargado el progreso existente de '{file_B9}'.")
        except Exception as e:
            print(f"\n[AVISO] No se pudo leer '{file_B9}' correctamente ({e}). Se iniciará desde cero.")

    if not df_B9.isna().any().any():
        print(f"[OK] '{file_B9}' ya está completamente calculada. Se omite su ejecución.")
    else:
        print("\n========================================================")
        print("Calculando Tabla B.9 (Capas vs Neuronas - 12 experimentos)...")
        print("========================================================")
        exp_idx = 0
        total_B9 = len(layers_B9) * len(neurons_B9)
        for n_lay in layers_B9:
            for n_neu in neurons_B9:
                exp_idx += 1
                col_l1 = f"l1_{n_neu}n"
                col_l2 = f"l2_{n_neu}n"

                if pd.notna(df_B9.loc[n_lay, col_l1]) and pd.notna(df_B9.loc[n_lay, col_l2]):
                    print(f"[{exp_idx}/{total_B9}] [SKIP] B.9 -> Capas={n_lay}, Neuronas={n_neu} ya completado "
                          f"(l1={df_B9.loc[n_lay, col_l1]:.3f}%, l2={df_B9.loc[n_lay, col_l2]:.3f}%).", flush=True)
                    continue

                print(f"\n[{exp_idx}/{total_B9}] Ejecutando B.9 -> Capas={n_lay}, Neuronas={n_neu}...", flush=True)
                e1, e2 = run_discrete_experiment(x_raw, t_raw, Exact, dt_target=0.8, noise_level=0.0, 
                                                 num_hidden_layers=n_lay, num_neurons=n_neu, q_fixed=32)
                df_B9.loc[n_lay, col_l1] = e1
                df_B9.loc[n_lay, col_l2] = e2
                df_B9.to_csv(file_B9)

    print("\n=== TABLA B.9 ===")
    print(df_B9.round(3))
    print_latex_rows(df_B9, "TABLA B.9")