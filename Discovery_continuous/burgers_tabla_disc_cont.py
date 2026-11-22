import warnings

warnings.filterwarnings('ignore', category=UserWarning)

import os
import random
import time
import numpy as np
import pandas as pd
import scipy.io
import torch
import torch.nn as nn

torch.set_default_dtype(torch.float32)

MAT_PATH = r'C:\Users\ematm\OneDrive\Desktop\TFG\Trabajo TFG propuesto\data\burgers_shock.mat'

# Forzamos el uso de dispositivo
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(
    f'Ejecutando sensibilidad continua (100% fiel a Raissi et al., solo L-BFGS)'
    f' en: {DEVICE}'
)

# Valores exactos de referencia (Sección B.1)
LAMBDA_1_EXACT = 1.0
LAMBDA_2_EXACT = 0.01 / np.pi


# ==========================================
# FUNCIÓN DE SEMILLA ESTANDARIZADA
# ==========================================
def set_seed(seed=42):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False


# ==========================================
# 1. DEFINICIÓN DEL MODELO FÍSICO (Continuo)
# ==========================================
class PINNDiscovery(nn.Module):

  def __init__(self, layers, lb, ub):
    super(PINNDiscovery, self).__init__()
    self.register_buffer('lb', torch.tensor(lb, dtype=torch.float32))
    self.register_buffer('ub', torch.tensor(ub, dtype=torch.float32))

    self.linears = nn.ModuleList([
        nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)
    ])
    self.activation = nn.Tanh()

    # Inicialización Xavier (Glorot) y sesgos a 0 como en el código original de Raissi
    for m in self.linears:
      nn.init.xavier_normal_(m.weight)
      nn.init.zeros_(m.bias)

    self.lambda_1 = nn.Parameter(torch.tensor([0.0], dtype=torch.float32))
    self.lambda_2 = nn.Parameter(torch.tensor([-6.0], dtype=torch.float32))

  def forward(self, x, t):
    X = torch.cat([x, t], dim=1)
    # Normalización de entradas a [-1, 1] exacta del código original de Raissi
    H = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
    for i in range(len(self.linears) - 1):
      H = self.activation(self.linears[i](H))
    u = self.linears[-1](H)
    return u

  def net_f(self, x, t):
    u = self.forward(x, t)
    u_t = torch.autograd.grad(
        u,
        t,
        grad_outputs=torch.ones_like(u),
        retain_graph=True,
        create_graph=True,
    )[0]
    u_x = torch.autograd.grad(
        u,
        x,
        grad_outputs=torch.ones_like(u),
        retain_graph=True,
        create_graph=True,
    )[0]
    u_xx = torch.autograd.grad(
        u_x,
        x,
        grad_outputs=torch.ones_like(u_x),
        retain_graph=True,
        create_graph=True,
    )[0]

    l2 = torch.exp(self.lambda_2)
    f = u_t + self.lambda_1 * u * u_x - l2 * u_xx
    return f


def loss_function(model, x, t, u_train):
  u_pred = model(x, t)
  mse_u = torch.mean((u_pred - u_train) ** 2)
  f_pred = model.net_f(x, t)
  mse_f = torch.mean(f_pred**2)
  return mse_u + mse_f


# ==========================================
# 2. FUNCIÓN DE ENTRENAMIENTO (SOLO L-BFGS)
# ==========================================
def run_continuous_experiment(
    X_star,
    u_star,
    lb,
    ub,
    N_train,
    noise_level,
    num_hidden_layers,
    num_neurons,
):
  # Aplicamos la semilla estandarizada al inicio de cada iteración
  set_seed(42)
  t_start = time.time()

  layers = [2] + [num_neurons] * num_hidden_layers + [1]
  model = PINNDiscovery(layers, lb, ub).to(DEVICE)

  idx = np.random.choice(X_star.shape[0], N_train, replace=False)
  x_train_np = X_star[idx, 0:1]
  t_train_np = X_star[idx, 1:2]
  u_train_np = u_star[idx, :].copy()

  if noise_level > 0:
    noise = (
        noise_level
        * np.std(u_train_np)
        * np.random.randn(*u_train_np.shape)
    )
    u_train_np = u_train_np + noise

  x_train = torch.tensor(
      x_train_np, dtype=torch.float32, requires_grad=True, device=DEVICE
  )
  t_train = torch.tensor(
      t_train_np, dtype=torch.float32, requires_grad=True, device=DEVICE
  )
  u_train_tensor = torch.tensor(u_train_np, dtype=torch.float32, device=DEVICE)

  optimizer = torch.optim.LBFGS(
      model.parameters(),
      lr=1.0,
      max_iter=10000,
      max_eval=10000,
      history_size=50,
      tolerance_grad=1e-5,
      tolerance_change=1.0 * np.finfo(float).eps,
      line_search_fn='strong_wolfe',
  )

  iter_count = [0]

  def closure():
    optimizer.zero_grad()
    loss = loss_function(model, x_train, t_train, u_train_tensor)
    loss.backward()

    iter_count[0] += 1
    if iter_count[0] % 500 == 0:
      l1_val = model.lambda_1.item()
      l2_val = torch.exp(model.lambda_2).item()
      print(
          f'     [Iter {iter_count[0]:5d}] Loss: {loss.item():.4e} | l1:'
          f' {l1_val:.5f} | l2: {l2_val:.6f}',
          flush=True,
      )
    return loss

  optimizer.step(closure)
  elapsed = time.time() - t_start

  lambda_1_pred = model.lambda_1.item()
  lambda_2_pred = torch.exp(model.lambda_2).item()

  error_lambda_1 = (
      np.abs(lambda_1_pred - LAMBDA_1_EXACT) / LAMBDA_1_EXACT * 100.0
  )
  error_lambda_2 = (
      np.abs(lambda_2_pred - LAMBDA_2_EXACT) / LAMBDA_2_EXACT * 100.0
  )

  print(
      f'    -> Fin ({iter_count[0]} iters en {elapsed:.1f}s) | l1:'
      f' {lambda_1_pred:.5f} (Err: {error_lambda_1:.3f}%) | l2:'
      f' {lambda_2_pred:.6f} (Err: {error_lambda_2:.3f}%)',
      flush=True,
  )
  return error_lambda_1, error_lambda_2


def print_latex_rows(df, title):
  print(f'\n--- FILAS LATEX PARA {title} ---')
  for idx, row in df.iterrows():
    vals = ' & '.join([f'\\texttt{{{v:.3f}}}' for v in row.values])
    print(f'\\textbf{{{idx}}} & {vals} \\\\')


# ==========================================
# 3. EJECUCIÓN DE TABLAS B.6 Y B.7 (con reanudación)
# ==========================================
if __name__ == '__main__':
  data = scipy.io.loadmat(MAT_PATH)
  x = data['x']
  t = data['t']
  Exact = np.real(data['usol'])

  X, T = np.meshgrid(x.flatten(), t.flatten(), indexing='ij')
  X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
  u_star = Exact.flatten()[:, None]

  lb = X_star.min(0)
  ub = X_star.max(0)

  # ---------------------------------------------------------
  # TABLA B.6: N vs Nivel de Ruido
  # ---------------------------------------------------------
  print('\n========================================================')
  print('Calculando Tabla B.6 (N vs Ruido - 16 experimentos)...')
  print('========================================================')
  N_list = [500, 1000, 1500, 2000]
  noise_list = [0.0, 0.01, 0.05, 0.10]
  cols_B6 = [f'l1_{int(n*100)}%' for n in noise_list] + [
      f'l2_{int(n*100)}%' for n in noise_list
  ]

  csv_b6 = 'burgers_tabla_B6.csv'
  if os.path.exists(csv_b6):
    df_B6 = pd.read_csv(csv_b6, index_col=0)
    print(f"Cargado archivo existente '{csv_b6}' para reanudar.")
  else:
    df_B6 = pd.DataFrame(index=N_list, columns=cols_B6, dtype=float)
    df_B6.index.name = 'N'

  exp_idx = 0
  total_B6 = len(N_list) * len(noise_list)
  for N_val in N_list:
    for noise in noise_list:
      exp_idx += 1
      col_l1 = f'l1_{int(noise*100)}%'
      col_l2 = f'l2_{int(noise*100)}%'

      # Omitir si la celda ya tiene un valor numérico registrado en el CSV
      if (
          N_val in df_B6.index
          and col_l1 in df_B6.columns
          and col_l2 in df_B6.columns
          and pd.notna(df_B6.loc[N_val, col_l1])
          and pd.notna(df_B6.loc[N_val, col_l2])
      ):
        print(
            f'\n[{exp_idx}/{total_B6}] Omitiendo B.6 -> N={N_val},'
            f' Ruido={int(noise*100)}% (ya completado en CSV).',
            flush=True,
        )
        continue

      print(
          f'\n[{exp_idx}/{total_B6}] Ejecutando B.6 -> N={N_val},'
          f' Ruido={int(noise*100)}%...',
          flush=True,
      )
      e1, e2 = run_continuous_experiment(
          X_star,
          u_star,
          lb,
          ub,
          N_val,
          noise,
          num_hidden_layers=8,
          num_neurons=20,
      )

      if N_val not in df_B6.index:
        df_B6.loc[N_val] = np.nan
      df_B6.loc[N_val, col_l1] = e1
      df_B6.loc[N_val, col_l2] = e2
      df_B6.to_csv(csv_b6)

  print('\n=== TABLA B.6 COMPLETADA ===')
  print(df_B6.round(3))
  print_latex_rows(df_B6, 'TABLA B.6')

  # ---------------------------------------------------------
  # TABLA B.7: Capas ocultas vs Neuronas
  # ---------------------------------------------------------
  print('\n========================================================')
  print('Calculando Tabla B.7 (Capas vs Neuronas - 12 experimentos)...')
  print('========================================================')
  layers_B7 = [2, 4, 6, 8]
  neurons_B7 = [10, 20, 40]
  cols_B7 = [f'l1_{neu}n' for neu in neurons_B7] + [
      f'l2_{neu}n' for neu in neurons_B7
  ]

  csv_b7 = 'burgers_tabla_B7.csv'
  if os.path.exists(csv_b7):
    df_B7 = pd.read_csv(csv_b7, index_col=0)
    print(f"Cargado archivo existente '{csv_b7}' para reanudar.")
  else:
    df_B7 = pd.DataFrame(index=layers_B7, columns=cols_B7, dtype=float)
    df_B7.index.name = 'Capas'

  exp_idx = 0
  total_B7 = len(layers_B7) * len(neurons_B7)
  for n_lay in layers_B7:
    for n_neu in neurons_B7:
      exp_idx += 1
      col_l1 = f'l1_{n_neu}n'
      col_l2 = f'l2_{n_neu}n'

      # Omitir si la celda ya tiene un valor numérico registrado en el CSV
      if (
          n_lay in df_B7.index
          and col_l1 in df_B7.columns
          and col_l2 in df_B7.columns
          and pd.notna(df_B7.loc[n_lay, col_l1])
          and pd.notna(df_B7.loc[n_lay, col_l2])
      ):
        print(
            f'\n[{exp_idx}/{total_B7}] Omitiendo B.7 -> Capas={n_lay},'
            f' Neuronas={n_neu} (ya completado en CSV).',
            flush=True,
        )
        continue

      print(
          f'\n[{exp_idx}/{total_B7}] Ejecutando B.7 -> Capas={n_lay},'
          f' Neuronas={n_neu}...',
          flush=True,
      )
      e1, e2 = run_continuous_experiment(
          X_star,
          u_star,
          lb,
          ub,
          N_train=2000,
          noise_level=0.0,
          num_hidden_layers=n_lay,
          num_neurons=n_neu,
      )

      if n_lay not in df_B7.index:
        df_B7.loc[n_lay] = np.nan
      df_B7.loc[n_lay, col_l1] = e1
      df_B7.loc[n_lay, col_l2] = e2
      df_B7.to_csv(csv_b7)

  print('\n=== TABLA B.7 COMPLETADA ===')
  print(df_B7.round(3))
  print_latex_rows(df_B7, 'TABLA B.7')
  print(
      "\n¡Archivos guardados correctamente como 'burgers_tabla_B6.csv' y"
      " 'burgers_tabla_B7.csv'!"
  )