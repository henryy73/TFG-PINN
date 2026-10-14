import os
import torch
import torch.nn as nn
import numpy as np
import scipy.io
import pandas as pd
import time

# ==========================================
# 0. CONFIGURACIÓN GLOBAL (Fiel al paper)
# ==========================================
torch.set_default_dtype(torch.float64)

# Detección de hardware (Raissi et al. usaron NVIDIA Titan X)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Hardware detectado para ejecución: {device}")

DATA_PATH = r'C:\Users\ematm\OneDrive\Desktop\TFG\Trabajo TFG propuesto\data\burgers_shock.mat'
RESULTS_FILE = r'C:\Users\ematm\OneDrive\Desktop\TFG\Trabajo TFG propuesto\Burgers\Direct_discrete\burgers_tabla_A4.csv'

# ==========================================
# CUADRO DE BUTCHER (Gauss-Legendre IRK)
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

    return (torch.tensor(A, dtype=torch.float64).to(device), 
            torch.tensor(b, dtype=torch.float64).to(device), 
            torch.tensor(c, dtype=torch.float64).to(device))

# ==========================================
# 1. MODELO FÍSICO (Sección A.2 - Ec. A.5 a A.8)
# ==========================================
class DiscretePhysicsInformedNN(nn.Module):
    def __init__(self, layers, q, dt, A, b):
        super(DiscretePhysicsInformedNN, self).__init__()
        self.q = q
        self.dt = dt
        self.A = A
        self.b = b
        
        # Arquitectura estricta: Capas densas simples con Tanh
        self.linears = nn.ModuleList([nn.Linear(layers[i], layers[i+1]) for i in range(len(layers)-1)])
        self.activation = nn.Tanh()

    def forward(self, x):
        inputs = x
        for i in range(len(self.linears) - 1):
            inputs = self.activation(self.linears[i](inputs))
        outputs = self.linears[-1](inputs)
        return outputs  # Salida estricta: [u^{n+c_1}, ..., u^{n+c_q}, u^{n+1}]

    def net_u_n(self, x):
        U = self.forward(x)
        U_stages = U[:, :self.q]       
        u_next = U[:, self.q:self.q+1] 
        
        nu = 0.01 / np.pi
        N_u = []
        
        # Optimización extrema de VJP en PyTorch:
        # Se sustituye ones_like por sum() para evitar cuellos de botella de memoria en GPU
        for j in range(self.q):
            u_j = U_stages[:, j]
            
            # Primera derivada
            u_j_x = torch.autograd.grad(u_j.sum(), x, retain_graph=True, create_graph=True)[0]
            # Segunda derivada
            u_j_xx = torch.autograd.grad(u_j_x.sum(), x, retain_graph=True, create_graph=True)[0]
            
            # Operador no lineal Ec. (A.6)
            n_j = u_j.unsqueeze(1) * u_j_x - nu * u_j_xx
            N_u.append(n_j)
            
        N_u_tensor = torch.cat(N_u, dim=1)
        
        # Proyección hacia atrás (Ec. A.5 y A.6 reconstruidas)
        u_i_n = U_stages + self.dt * torch.matmul(N_u_tensor, self.A.T)
        u_q_plus_1_n = u_next + self.dt * torch.matmul(N_u_tensor, self.b.unsqueeze(1))
        
        U_n = torch.cat([u_i_n, u_q_plus_1_n], dim=1)
        return U_n

def loss_function(model, x_n, u_n_train, x_b_left, x_b_right):
    U_n_pred = model.net_u_n(x_n)
    sse_n = torch.sum((U_n_pred - u_n_train) ** 2)
    
    U_b_l = model(x_b_left)
    U_b_r = model(x_b_right)
    sse_b = torch.sum(U_b_l ** 2) + torch.sum(U_b_r ** 2)
    
    return sse_n + sse_b

# ==========================================
# 2. MOTOR DE ENTRENAMIENTO PARA CADA CELDA
# ==========================================
def run_discrete_experiment(q, dt):
    A, b, c = get_implicit_rk_coeffs(q)
    
    data = scipy.io.loadmat(DATA_PATH)
    x_raw = data['x']  
    t_raw = data['t']  
    Exact = np.real(data['usol']) 

    idx_t0 = np.argmin(np.abs(t_raw.flatten() - 0.1))
    idx_t1 = np.argmin(np.abs(t_raw.flatten() - (0.1 + dt)))

    np.random.seed(1234)
    N_n = 250 
    idx_x = np.random.choice(x_raw.shape[0], N_n, replace=False)
    x_n_train = x_raw[idx_x, :]
    u_n_train = Exact[idx_x, idx_t0:idx_t0+1]

    # Envío de tensores al hardware disponible
    x_n_tensor = torch.tensor(x_n_train, dtype=torch.float64, requires_grad=True).to(device)
    u_n_tensor = torch.tensor(u_n_train, dtype=torch.float64).to(device)
    x_b_left = torch.tensor([[-1.0]], dtype=torch.float64).to(device)
    x_b_right = torch.tensor([[1.0]], dtype=torch.float64).to(device)

    # 4 capas ocultas con 50 neuronas cada una
    layers = [1, 50, 50, 50, 50, q + 1]
    model = DiscretePhysicsInformedNN(layers, q, dt, A, b).to(device)
    
    # Optimizador L-BFGS de lote completo ajustado para no estancarse
    optimizer = torch.optim.LBFGS(
        model.parameters(), 
        lr=1.0, 
        max_iter=10000, 
        max_eval=10000, 
        history_size=50,
        tolerance_grad=1e-5, 
        tolerance_change=1.0 * np.finfo(float).eps, 
        line_search_fn="strong_wolfe"
    )
    
    eval_count = 0
    def closure():
        nonlocal eval_count
        optimizer.zero_grad()
        loss = loss_function(model, x_n_tensor, u_n_tensor, x_b_left, x_b_right)
        loss.backward()
        
        eval_count += 1
        if eval_count % 500 == 0:
            print(f"         [Evaluación L-BFGS {eval_count}] SSE Loss: {loss.item():.4e}")
        return loss

    print(f"      -> Entrenando con L-BFGS (Replicación estricta)...")
    optimizer.step(closure)
    
    # Evaluación
    model.eval()
    x_star_tensor = torch.tensor(x_raw, dtype=torch.float64).to(device)
    with torch.no_grad():
        U_pred_all = model(x_star_tensor).cpu().numpy()
        u_pred_t1 = U_pred_all[:, -1:]
        
    u_exact_t1 = Exact[:, idx_t1:idx_t1+1]
    error_l2 = np.linalg.norm(u_exact_t1 - u_pred_t1, 2) / np.linalg.norm(u_exact_t1, 2)
    
    return error_l2

# ==========================================
# 3. BUCLE PRINCIPAL DE AUTOMATIZACIÓN
# ==========================================
if __name__ == "__main__":
    q_values = [1, 2, 4, 8, 16, 32, 64, 100]
    dt_values = [0.2, 0.4, 0.6, 0.8]
    
    results_dir = os.path.dirname(RESULTS_FILE)
    if results_dir:
        os.makedirs(results_dir, exist_ok=True)
    
    if os.path.exists(RESULTS_FILE):
        results_table = pd.read_csv(RESULTS_FILE, index_col=0)
        results_table.columns = [float(c) for c in results_table.columns]
        results_table = results_table.astype(object)
    else:
        results_table = pd.DataFrame(index=q_values, columns=dt_values, dtype=object)
        results_table.index.name = 'q \ dt'
    
    print("\nIniciando generación de la Tabla A.4 (Fiel a Raissi et al., 2019)...")
    start_time_total = time.time()
    
    for dt in dt_values:
        for q in q_values:
            val = results_table.at[q, dt]
            try:
                if pd.notna(val) and str(val) != "ERROR" and float(val) < 0.1:
                    print(f"Saltando q={q}, dt={dt} (Calculado previamente: {val})")
                    continue
            except ValueError:
                pass
                
            print(f"\n======================================")
            print(f"Ejecutando q={q}, dt={dt}")
            print(f"======================================")
            
            start_time_exp = time.time()
            try:
                error = run_discrete_experiment(q, dt)
                formatted_error = f"{error:.1e}"
                results_table.at[q, dt] = formatted_error
                
                print(f"  -> FINALIZADO | Error L2: {formatted_error} | Tiempo: {(time.time() - start_time_exp)/60:.1f} min")
                results_table.to_csv(RESULTS_FILE)
                
            except Exception as e:
                print(f"  -> ERROR CRÍTICO en q={q}, dt={dt}:")
                import traceback
                traceback.print_exc()
                
                results_table.at[q, dt] = "ERROR"
                results_table.to_csv(RESULTS_FILE)

    print("\n======================================")
    print("        TABLA A.4 COMPLETADA          ")
    print("======================================")
    print(f"Tiempo total invertido: {(time.time() - start_time_total)/3600:.2f} horas")
    print(results_table)
    
    # Forzar notación científica estricta en la visualización
    def formato_cientifico(val):
        try:
            return f"{float(val):.1e}"
        except:
            return val
            
    # Usamos applymap (o map en versiones nuevas de pandas) para dar formato celda a celda
    if hasattr(results_table, 'map'):
        print(results_table.map(formato_cientifico))
    else:
        print(results_table.applymap(formato_cientifico))