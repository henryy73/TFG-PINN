# -*- coding: utf-8 -*-
"""
Generación de las Tablas A.1 y A.2 (Sección A.1 - Tiempo Continuo)
Basado en burgers_table_A1.py con semilla de reproducibilidad SEED=42
"""

import os
import time
import random
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import torch
import torch.nn as nn
import numpy as np
import scipy.io
from scipy.stats import qmc
import pandas as pd

# ==========================================
# 0. CONFIGURACIÓN DE SEMILLA Y HARDWARE
# ==========================================
SEED = 42

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(SEED)

# Configuración global en float32 para Tiempo Continuo
torch.set_default_dtype(torch.float32)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.init()
    _ = torch.matmul(torch.zeros((1, 1), dtype=torch.float32, device=device),
                     torch.zeros((1, 1), dtype=torch.float32, device=device))

print(f"Hardware detectado para ejecución (Sección A.1 - float32): {device}")

DATA_PATH = r'C:\Users\ematm\OneDrive\Desktop\TFG\Trabajo TFG propuesto\data\burgers_shock.mat'
RUTA_GUARDADO = r'C:\Users\ematm\OneDrive\Desktop\TFG\Trabajo TFG propuesto\Burgers\Direct_continuous'

def safe_to_csv(df, path):
    for _ in range(3):
        try:
            df.to_csv(path)
            return
        except PermissionError:
            print(f"  [AVISO] '{path}' abierto en Excel o bloqueado. Reintentando en 2s...")
            time.sleep(2)
    df.to_csv(path.replace(".csv", f"_backup_{int(time.time())}.csv"))

# ==========================================
# 1. DEFINICIÓN DEL MODELO FÍSICO
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

# ==========================================
# 2. FUNCIÓN PRINCIPAL DE EVALUACIÓN
# ==========================================
def evaluar_configuracion(N_u, N_f, num_layers=8, num_neurons=20, max_iter=10000): 
    # Aseguramos la misma semilla para cada evaluación
    set_seed(SEED)

    # Cargar y preparar datos base
    data = scipy.io.loadmat(DATA_PATH)  
    x = data['x']
    t = data['t']
    Exact = np.real(data['usol'])
    
    X, T = np.meshgrid(x.flatten(), t.flatten(), indexing='ij')
    X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
    
    xx1 = np.hstack((x, np.zeros_like(x)))
    uu1 = Exact[:, 0:1]
    xx2 = np.hstack((-1.0 * np.ones_like(t), t))
    uu2 = Exact[0:1, :].T
    xx3 = np.hstack((1.0 * np.ones_like(t), t))
    uu3 = Exact[-1:, :].T
    
    X_u_train = np.vstack([xx1, xx2, xx3])
    u_train = np.vstack([uu1, uu2, uu3])
    
    # Submuestreo de N_u
    idx = np.random.choice(X_u_train.shape[0], N_u, replace=False)
    X_u_train = X_u_train[idx, :]
    u_train = u_train[idx, :]
    
    # Generación de N_f mediante Hipercubo Latino con semilla
    lb = X_star.min(0)
    ub = X_star.max(0)
    sampler = qmc.LatinHypercube(d=2, seed=SEED)
    sample = sampler.random(n=N_f)
    X_f_train = lb + (ub - lb) * sample
    X_f_train = np.vstack((X_f_train, X_u_train))
    
    # Tensores enviados a device
    x_u = torch.tensor(X_u_train[:, 0:1], dtype=torch.float32, device=device)
    t_u = torch.tensor(X_u_train[:, 1:2], dtype=torch.float32, device=device)
    u_train_tensor = torch.tensor(u_train, dtype=torch.float32, device=device)
    x_f = torch.tensor(X_f_train[:, 0:1], dtype=torch.float32, requires_grad=True, device=device)
    t_f = torch.tensor(X_f_train[:, 1:2], dtype=torch.float32, requires_grad=True, device=device)
    
    # Inicialización del modelo con arquitectura parametrizada
    layers = [2] + [num_neurons] * num_layers + [1]
    model = PhysicsInformedNN(layers).to(device)
    optimizer = torch.optim.LBFGS(
        model.parameters(), lr=1.0, max_iter=max_iter, max_eval=max_iter, history_size=50,
        tolerance_grad=1e-5, tolerance_change=1.0 * np.finfo(float).eps, line_search_fn="strong_wolfe"
    )
    
    # Uso de propiedad interna de la función para evitar advertencias de ámbito (global/nonlocal)
    def closure():
        optimizer.zero_grad()
        loss = loss_function(model, x_u, t_u, u_train_tensor, x_f, t_f)
        loss.backward()
        
        closure.eval_count += 1
        if closure.eval_count % 500 == 0:
            print(f"         [Evaluación L-BFGS {closure.eval_count}] MSE Loss: {loss.item():.6e}", flush=True)
            
        return loss
    
    closure.eval_count = 0
    
    optimizer.step(closure)
    
    # Evaluación del error relativo L2
    model.eval()
    with torch.no_grad():
        x_star_tensor = torch.tensor(X_star[:, 0:1], dtype=torch.float32, device=device)
        t_star_tensor = torch.tensor(X_star[:, 1:2], dtype=torch.float32, device=device)
        u_pred = model(x_star_tensor, t_star_tensor).cpu().numpy().reshape(Exact.shape)
        
    error_l2 = np.linalg.norm(Exact - u_pred, 2) / np.linalg.norm(Exact, 2)
    return error_l2

def formato_cientifico(val):
    try:
        return f"{float(val):.2e}"
    except:
        return val

def print_latex_table(df, nombre):
    print(f"\n--- FILAS LATEX PARA {nombre} ---")
    for idx, row in df.iterrows():
        vals = " & ".join([f"\\texttt{{{float(v):.2e}}}" if pd.notna(v) and str(v) != "ERROR" else "--" for v in row.values])
        print(f"\\textbf{{{idx}}} & {vals} \\\\")

# ==========================================
# 3. BUCLES PARA GENERAR LAS TABLAS A.1 Y A.2
# ==========================================
if __name__ == "__main__":
    os.makedirs(RUTA_GUARDADO, exist_ok=True)
    t_global = time.time()

    # ---------------------------------------------------------
    # TABLA A.1: Variación de N_u y N_f (Red fija: 8 capas ocultas, 20 neuronas)
    # ---------------------------------------------------------
    Nu_list = [20, 40, 60, 80, 100, 200]
    Nf_list = [2000, 4000, 6000, 7000, 8000, 10000]
    archivo_A1 = os.path.join(RUTA_GUARDADO, 'burgers_tabla_A1.csv')
    
    if os.path.exists(archivo_A1):
        print("\nCargando Tabla A.1 existente...")
        resultados_A1 = pd.read_csv(archivo_A1, index_col=0)
        resultados_A1.columns = [int(float(c)) for c in resultados_A1.columns] 
        resultados_A1 = resultados_A1.astype(object)
    else:
        resultados_A1 = pd.DataFrame(index=Nu_list, columns=Nf_list, dtype=object)
        resultados_A1.index.name = "N_u / N_f"
    
    print("\nIniciando verificación/generación de la Tabla A.1...")
    for nu in Nu_list:
        for nf in Nf_list:
            val = resultados_A1.at[nu, nf]
            try:
                if pd.notna(val) and str(val) != "ERROR":
                    print(f"Saltando A.1 -> Nu={nu}, Nf={nf} (Ya calculado: {val})")
                    continue
            except ValueError:
                pass
            
            print(f"\n======================================")
            print(f"Ejecutando Tabla A.1 -> Nu={nu}, Nf={nf}")
            print(f"======================================")
            t0 = time.time()
            try:
                error = evaluar_configuracion(nu, nf, num_layers=8, num_neurons=20, max_iter=10000)
                formatted_error = f"{error:.2e}"
                resultados_A1.at[nu, nf] = formatted_error
                print(f"  -> FINALIZADO | Error L2: {formatted_error} | Tiempo: {time.time()-t0:.1f}s")
                safe_to_csv(resultados_A1, archivo_A1)
            except Exception:
                import traceback
                traceback.print_exc()
                resultados_A1.at[nu, nf] = "ERROR"
                safe_to_csv(resultados_A1, archivo_A1)

    print("\n======================================")
    print("         TABLA A.1 COMPLETADA          ")
    print("======================================")
    print(resultados_A1)
    print_latex_table(resultados_A1, "TABLA A.1")

    # ---------------------------------------------------------
    # TABLA A.2: Capas ocultas vs Neuronas (Fijo: N_u=100, N_f=10000)
    # ---------------------------------------------------------
    layers_list = [2, 4, 6, 8]
    neurons_list = [10, 20, 40]
    archivo_A2 = os.path.join(RUTA_GUARDADO, 'burgers_tabla_A2.csv')

    if os.path.exists(archivo_A2):
        print("\nCargando Tabla A.2 existente...")
        resultados_A2 = pd.read_csv(archivo_A2, index_col=0)
        resultados_A2.columns = [int(float(c)) for c in resultados_A2.columns]
        resultados_A2 = resultados_A2.astype(object)
    else:
        resultados_A2 = pd.DataFrame(index=layers_list, columns=neurons_list, dtype=object)
        resultados_A2.index.name = "Layers / Neurons"

    print("\nIniciando generación de la Tabla A.2...")
    for n_lay in layers_list:
        for n_neu in neurons_list:
            val = resultados_A2.at[n_lay, n_neu]
            try:
                if pd.notna(val) and str(val) != "ERROR":
                    print(f"Saltando A.2 -> Capas={n_lay}, Neuronas={n_neu} (Ya calculado: {val})")
                    continue
            except ValueError:
                pass

            print(f"\n======================================")
            print(f"Ejecutando Tabla A.2 -> Capas={n_lay}, Neuronas={n_neu} (Nu=100, Nf=10000)")
            print(f"======================================")
            t0 = time.time()
            try:
                error = evaluar_configuracion(100, 10000, num_layers=n_lay, num_neurons=n_neu, max_iter=10000)
                formatted_error = f"{error:.2e}"
                resultados_A2.at[n_lay, n_neu] = formatted_error
                print(f"  -> FINALIZADO | Error L2: {formatted_error} | Tiempo: {time.time()-t0:.1f}s")
                safe_to_csv(resultados_A2, archivo_A2)
            except Exception:
                import traceback
                traceback.print_exc()
                resultados_A2.at[n_lay, n_neu] = "ERROR"
                safe_to_csv(resultados_A2, archivo_A2)

    print("\n======================================")
    print("         TABLA A.2 COMPLETADA          ")
    print("======================================")
    print(f"Tiempo total invertido: {(time.time() - t_global)/3600:.2f} horas")
    print(resultados_A2)
    print_latex_table(resultados_A2, "TABLA A.2")