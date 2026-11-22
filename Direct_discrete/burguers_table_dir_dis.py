# -*- coding: utf-8 -*-
"""Generación de las Tablas A.3, A.4 y A.5 (Sección A.2 - Tiempo Discreto)

Basado exactamente en burguers_table_A4.py
"""

import os
import random
import time
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
import scipy.io
import torch
import torch.nn as nn

# ==========================================
# 0. CONFIGURACIÓN GLOBAL Y SEMILLA (SEED = 42)
# ==========================================
SEED = 42


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


set_seed(SEED)
torch.set_default_dtype(torch.float64)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.init()
    _ = torch.matmul(
        torch.zeros((1, 1), dtype=torch.float64, device=device),
        torch.zeros((1, 1), dtype=torch.float64, device=device),
    )

print(
    f"Hardware detectado para ejecución (Sección A.2 - float64): {device} |"
    f" SEED={SEED}"
)

DATA_PATH = r"C:\Users\ematm\OneDrive\Desktop\TFG\Trabajo TFG propuesto\data\burgers_shock.mat"
RESULTS_DIR = r"C:\Users\ematm\OneDrive\Desktop\TFG\Trabajo TFG propuesto\Burgers\Direct_discrete"

FILE_A3 = os.path.join(RESULTS_DIR, "burgers_tabla_A3.csv")
FILE_A4 = os.path.join(RESULTS_DIR, "burgers_tabla_A4.csv")
FILE_A5 = os.path.join(RESULTS_DIR, "burgers_tabla_A5.csv")


Q_FIJO_A3_A5 = 100


def safe_to_csv(df, path):
    for _ in range(3):
        try:
            df.to_csv(path)
            return
        except PermissionError:
            print(
                f"  [AVISO] '{path}' abierto en Excel o bloqueado. Reintentando en"
                " 2s..."
            )
            time.sleep(2)
    df.to_csv(path.replace(".csv", f"_backup_{int(time.time())}.csv"))


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

    return (
        torch.tensor(A, dtype=torch.float64).to(device),
        torch.tensor(b, dtype=torch.float64).to(device),
        torch.tensor(c, dtype=torch.float64).to(device),
    )


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
        self.linears = nn.ModuleList([
            nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)
        ])
        self.activation = nn.Tanh()

    def forward(self, x):
        inputs = x
        for i in range(len(self.linears) - 1):
            inputs = self.activation(self.linears[i](inputs))
        outputs = self.linears[-1](inputs)
        return outputs  # Salida estricta: [u^{n+c_1}, ..., u^{n+c_q}, u^{n+1}]

    def net_u_n(self, x):
        U = self.forward(x)
        U_stages = U[:, : self.q]
        u_next = U[:, self.q : self.q + 1]

        nu = 0.01 / np.pi
        N_u = []

        for j in range(self.q):
            u_j = U_stages[:, j]
            u_j_x = torch.autograd.grad(
                u_j.sum(), x, retain_graph=True, create_graph=True
            )[0]
            u_j_xx = torch.autograd.grad(
                u_j_x.sum(), x, retain_graph=True, create_graph=True
            )[0]
            n_j = u_j.unsqueeze(1) * u_j_x - nu * u_j_xx
            N_u.append(n_j)

        N_u_tensor = torch.cat(N_u, dim=1)

        # Proyección hacia atrás (Ec. A.5 y A.6 reconstruidas)
        u_i_n = U_stages + self.dt * torch.matmul(N_u_tensor, self.A.T)
        u_q_plus_1_n = u_next + self.dt * torch.matmul(
            N_u_tensor, self.b.unsqueeze(1)
        )

        U_n = torch.cat([u_i_n, u_q_plus_1_n], dim=1)
        return U_n


def loss_function(model, x_n, u_n_train, x_b_left, x_b_right):
    U_n_pred = model.net_u_n(x_n)
    sse_n = torch.sum((U_n_pred - u_n_train) ** 2)

    U_b_l = model(x_b_left)
    U_b_r = model(x_b_right)
    sse_b = torch.sum(U_b_l**2) + torch.sum(U_b_r**2)

    return sse_n + sse_b


# ==========================================
# 2. MOTOR DE ENTRENAMIENTO (Vale para A.3, A.4 y A.5)
# ==========================================
def run_discrete_experiment(q, dt, num_layers=4, num_neurons=50, N_n=250):
    # Reiniciamos la semilla a 42 antes de cada experimento para consistencia total
    set_seed(SEED)

    A, b, c = get_implicit_rk_coeffs(q)

    data = scipy.io.loadmat(DATA_PATH)
    x_raw = data["x"]
    t_raw = data["t"]
    Exact = np.real(data["usol"])

    idx_t0 = np.argmin(np.abs(t_raw.flatten() - 0.1))
    idx_t1 = np.argmin(np.abs(t_raw.flatten() - (0.1 + dt)))

    idx_x = np.random.choice(x_raw.shape[0], N_n, replace=False)
    x_n_train = x_raw[idx_x, :]
    u_n_train = Exact[idx_x, idx_t0 : idx_t0 + 1]

    # Envío de tensores al hardware disponible
    x_n_tensor = torch.tensor(
        x_n_train, dtype=torch.float64, requires_grad=True
    ).to(device)
    u_n_tensor = torch.tensor(u_n_train, dtype=torch.float64).to(device)
    x_b_left = torch.tensor([[-1.0]], dtype=torch.float64).to(device)
    x_b_right = torch.tensor([[1.0]], dtype=torch.float64).to(device)

    # Arquitectura parametrizada (por defecto 4 capas, 50 neuronas)
    layers = [1] + [num_neurons] * num_layers + [q + 1]
    model = DiscretePhysicsInformedNN(layers, q, dt, A, b).to(device)

    # Optimizador L-BFGS de lote completo con max_iter = 10000
    optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=1.0,
        max_iter=10000,
        max_eval=10000,
        history_size=50,
        tolerance_grad=1e-8,
        tolerance_change=1.0 * np.finfo(float).eps,
        line_search_fn="strong_wolfe",
    )

    eval_count = 0

    def closure():
        nonlocal eval_count
        optimizer.zero_grad()
        loss = loss_function(model, x_n_tensor, u_n_tensor, x_b_left, x_b_right)
        loss.backward()

        eval_count += 1
        if eval_count % 500 == 0:
            # Formato en notación científica e-02 con 2 decimales
            print(
                f"          [Evaluación L-BFGS {eval_count}] SSE Loss:"
                f" {loss.item():.2e}",
                flush=True,
            )
        return loss

    print(
        f"     -> Entrenando con L-BFGS (q={q}, dt={dt}, Capas={num_layers},"
        f" Neuronas={num_neurons}, N_n={N_n})..."
    )
    optimizer.step(closure)

    # Evaluación
    model.eval()
    x_star_tensor = torch.tensor(x_raw, dtype=torch.float64).to(device)
    with torch.no_grad():
        U_pred_all = model(x_star_tensor).cpu().numpy()
        u_pred_t1 = U_pred_all[:, -1:]

    u_exact_t1 = Exact[:, idx_t1 : idx_t1 + 1]
    error_l2 = np.linalg.norm(u_exact_t1 - u_pred_t1, 2) / np.linalg.norm(
        u_exact_t1, 2
    )

    return error_l2


def print_latex_table(df, nombre):
    print(f"\n--- FILAS LATEX PARA {nombre} ---")
    for idx, row in df.iterrows():
        vals = " & ".join([
            f"\\texttt{{{float(v):.2e}}}"
            if pd.notna(v) and str(v) != "ERROR"
            else "--"
            for v in row.values
        ])
        print(f"\\textbf{{{idx}}} & {vals} \\\\")


# ==========================================
# 3. BUCLE PRINCIPAL DE AUTOMATIZACIÓN (A.3, A.4 y A.5)
# ==========================================
if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    start_time_total = time.time()

    # ---------------------------------------------------------
    # 1) TABLA A.4: q vs dt (4 capas, 50 neuronas, N_n=250)
    # ---------------------------------------------------------
    q_values = [1, 2, 4, 8, 16, 32, 64, 100]
    dt_values = [0.2, 0.4, 0.6, 0.8]

    if os.path.exists(FILE_A4):
        table_A4 = pd.read_csv(FILE_A4, index_col=0)
        table_A4.columns = [float(c) for c in table_A4.columns]
        table_A4 = table_A4.reindex(index=q_values, columns=dt_values).astype(
            object
        )
    else:
        table_A4 = pd.DataFrame(index=q_values, columns=dt_values, dtype=object)
        table_A4.index.name = "q \\ dt"

    print("\nIniciando verificación/generación de la Tabla A.4...")
    for dt in dt_values:
        for q in q_values:
            val = table_A4.at[q, dt]
            try:
                if pd.notna(val) and str(val) != "ERROR":
                    print(
                        f"Saltando A.4 -> q={q}, dt={dt} (Calculado"
                        f" previamente: {val})"
                    )
                    continue
            except ValueError:
                pass

            print(f"\n======================================")
            print(f"Ejecutando Tabla A.4 -> q={q}, dt={dt}")
            print(f"======================================")
            t0 = time.time()
            try:
                error = run_discrete_experiment(
                    q, dt, num_layers=4, num_neurons=50, N_n=250
                )
                formatted_error = f"{error:.2e}"
                table_A4.at[q, dt] = formatted_error
                print(
                    f"  -> FINALIZADO | Error L2: {formatted_error} | Tiempo:"
                    f" {(time.time() - t0)/60:.1f} min"
                )
                safe_to_csv(table_A4, FILE_A4)
            except Exception:
                import traceback

                traceback.print_exc()
                table_A4.at[q, dt] = "ERROR"
                safe_to_csv(table_A4, FILE_A4)

    print("\n======================================")
    print("         TABLA A.4 COMPLETADA          ")
    print("======================================")
    print(table_A4)
    print_latex_table(table_A4, "TABLA A.4")

    # ---------------------------------------------------------
    # 2) TABLA A.3: Capas vs Neuronas (q=Q_FIJO_A3_A5, dt=0.8, N_n=250)
    # ---------------------------------------------------------
    layers_A3 = [1, 2, 3]
    neurons_A3 = [10, 25, 50]

    if os.path.exists(FILE_A3):
        table_A3 = pd.read_csv(FILE_A3, index_col=0)
        table_A3.columns = [int(float(c)) for c in table_A3.columns]
        table_A3 = table_A3.astype(object)
    else:
        table_A3 = pd.DataFrame(
            index=layers_A3, columns=neurons_A3, dtype=object
        )
        table_A3.index.name = "Layers / Neurons"

    print("\nIniciando generación de la Tabla A.3...")
    for n_lay in layers_A3:
        for n_neu in neurons_A3:
            val = table_A3.at[n_lay, n_neu]
            try:
                if pd.notna(val) and str(val) != "ERROR":
                    print(
                        f"Saltando A.3 -> Capas={n_lay}, Neuronas={n_neu}"
                        f" (Calculado previamente: {val})"
                    )
                    continue
            except ValueError:
                pass

            print(f"\n======================================")
            print(
                f"Ejecutando Tabla A.3 -> Capas={n_lay}, Neuronas={n_neu}"
                f" (q={Q_FIJO_A3_A5}, dt=0.8)"
            )
            print(f"======================================")
            t0 = time.time()
            try:
                error = run_discrete_experiment(
                    Q_FIJO_A3_A5, 0.8, num_layers=n_lay, num_neurons=n_neu, N_n=250
                )
                formatted_error = f"{error:.2e}"
                table_A3.at[n_lay, n_neu] = formatted_error
                print(
                    f"  -> FINALIZADO | Error L2: {formatted_error} | Tiempo:"
                    f" {(time.time() - t0)/60:.1f} min"
                )
                safe_to_csv(table_A3, FILE_A3)
            except Exception:
                import traceback

                traceback.print_exc()
                table_A3.at[n_lay, n_neu] = "ERROR"
                safe_to_csv(table_A3, FILE_A3)

    print("\n======================================")
    print("         TABLA A.3 COMPLETADA          ")
    print("======================================")
    print(table_A3)
    print_latex_table(table_A3, "TABLA A.3")

    # ---------------------------------------------------------
    # 3) TABLA A.5: Variación de N_n (q=Q_FIJO_A3_A5, dt=0.8, 3 capas, 50 neuronas)
    # ---------------------------------------------------------
    Nn_values = [250, 200, 150, 100, 50, 10]

    if os.path.exists(FILE_A5):
        table_A5 = pd.read_csv(FILE_A5, index_col=0)
        table_A5.columns = [int(float(c)) for c in table_A5.columns]
        table_A5 = table_A5.astype(object)
    else:
        table_A5 = pd.DataFrame(
            index=["Error"], columns=Nn_values, dtype=object
        )
        table_A5.index.name = "N_n"

    print("\nIniciando generación de la Tabla A.5...")
    for n_val in Nn_values:
        val = table_A5.at["Error", n_val]
        try:
            if pd.notna(val) and str(val) != "ERROR":
                print(
                    f"Saltando A.5 -> N_n={n_val} (Calculado previamente: {val})"
                )
                continue
        except ValueError:
            pass

        print(f"\n======================================")
        print(
            f"Ejecutando Tabla A.5 -> N_n={n_val} (q={Q_FIJO_A3_A5}, dt=0.8, 3"
            " capas, 50 neuronas)"
        )
        print(f"======================================")
        t0 = time.time()
        try:
            error = run_discrete_experiment(
                Q_FIJO_A3_A5, 0.8, num_layers=3, num_neurons=50, N_n=n_val
            )
            formatted_error = f"{error:.2e}"
            table_A5.at["Error", n_val] = formatted_error
            print(
                f"  -> FINALIZADO | Error L2: {formatted_error} | Tiempo:"
                f" {(time.time() - t0)/60:.1f} min"
            )
            safe_to_csv(table_A5, FILE_A5)
        except Exception:
            import traceback

            traceback.print_exc()
            table_A5.at["Error", n_val] = "ERROR"
            safe_to_csv(table_A5, FILE_A5)

    print("\n======================================")
    print("         TABLA A.5 COMPLETADA          ")
    print("======================================")
    print(
        f"Tiempo total invertido: {(time.time() - start_time_total)/3600:.2f}"
        " horas"
    )
    print(table_A5)
    print_latex_table(table_A5, "TABLA A.5")