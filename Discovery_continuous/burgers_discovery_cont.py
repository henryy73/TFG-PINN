import os
import random
import numpy as np
import scipy.io
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ==========================================
# 0. CONFIGURACIÓN DE HARDWARE Y SEMILLA
# ==========================================
torch.set_default_dtype(torch.float32)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Ejecutando en dispositivo: {device}" + (f" ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else ""))

# ==========================================
# 1. DEFINICIÓN DEL MODELO FÍSICO (Inverso)
# ==========================================
class PINNDiscovery(nn.Module):
    def __init__(self, layers, lb, ub):
        super(PINNDiscovery, self).__init__()
        self.register_buffer('lb', torch.tensor(lb, dtype=torch.float32))
        self.register_buffer('ub', torch.tensor(ub, dtype=torch.float32))

        self.linears = nn.ModuleList([nn.Linear(layers[i], layers[i+1]) for i in range(len(layers)-1)])
        self.activation = nn.Tanh()

        for m in self.linears:
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)

        self.lambda_1 = nn.Parameter(torch.tensor([0.0], dtype=torch.float32))
        self.lambda_2 = nn.Parameter(torch.tensor([-6.0], dtype=torch.float32))

    def forward(self, x, t):
        X = torch.cat([x, t], dim=1)
        H = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
        for i in range(len(self.linears) - 1):
            H = self.activation(self.linears[i](H))
        u = self.linears[-1](H)
        return u

    def net_f(self, x, t):
        u = self.forward(x, t)
        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), retain_graph=True, create_graph=True)[0]
        
        l2 = torch.exp(self.lambda_2)
        f = u_t + self.lambda_1 * u * u_x - l2 * u_xx
        return f

def loss_function(model, x, t, u_train):
    u_pred = model(x, t)
    mse_u = torch.mean((u_pred - u_train) ** 2)
    f_pred = model.net_f(x, t)
    mse_f = torch.mean(f_pred ** 2)
    return mse_u + mse_f

# ==========================================
# 2. CARGA DE DATOS Y CONFIGURACIÓN GLOBAL
# ==========================================
data = scipy.io.loadmat(r'C:\Users\ematm\OneDrive\Desktop\TFG\Trabajo TFG propuesto\data\burgers_shock.mat') 

x = data['x']
t = data['t']
Exact = np.real(data['usol'])

X, T = np.meshgrid(x.flatten(), t.flatten(), indexing='ij')
X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
u_star = Exact.flatten()[:,None]

lb = X_star.min(0)
ub = X_star.max(0)

layers = [2, 20, 20, 20, 20, 20, 20, 20, 20, 1]
noise_levels = [0.0, 0.01]  # Bucle para 0% y 1% de ruido

lambda_1_exact = 1.0
lambda_2_exact = 0.01 / np.pi

# ==========================================
# 3. BUCLE DE ENTRENAMIENTO Y EVALUACIÓN
# ==========================================
for noise_level in noise_levels:
    print(f"\n==================================================")
    print(f" EJECUTANDO EXPERIMENTO CON RUIDO: {noise_level*100}%")
    print(f"==================================================")
    
    set_seed(42)
    
    model = PINNDiscovery(layers, lb, ub).to(device)
    
    N_train = 2000
    idx = np.random.choice(X_star.shape[0], N_train, replace=False)

    x_train_np = X_star[idx, 0:1]
    t_train_np = X_star[idx, 1:2]
    u_train_np = u_star[idx, :].copy()

    if noise_level > 0:
        noise = noise_level * np.std(u_train_np) * np.random.randn(*u_train_np.shape)
        u_train_np = u_train_np + noise

    x_train = torch.tensor(x_train_np, dtype=torch.float32, device=device, requires_grad=True)
    t_train = torch.tensor(t_train_np, dtype=torch.float32, device=device, requires_grad=True)
    u_train_tensor = torch.tensor(u_train_np, dtype=torch.float32, device=device)

    optimizer = torch.optim.LBFGS(
        model.parameters(), lr=1.0, max_iter=10000, max_eval=10000, history_size=50,
        tolerance_grad=1e-5, tolerance_change=1.0 * np.finfo(float).eps, line_search_fn="strong_wolfe"
    )

    # Uso de lista mutable [0] para evitar conflictos de ámbito
    iter_count = [0]
    def closure():
        optimizer.zero_grad()
        loss = loss_function(model, x_train, t_train, u_train_tensor)
        loss.backward()
        
        iter_count[0] += 1
        if iter_count[0] % 100 == 0:
            l1_val = model.lambda_1.item()
            l2_val = torch.exp(model.lambda_2).item()
            print(f'  Iter: {iter_count[0]}, Loss: {loss.item():.5e}, l1: {l1_val:.5f}, l2: {l2_val:.6f}')
            
        return loss

    print(f"Iniciando descubrimiento...")
    optimizer.step(closure)
    print(f"Entrenamiento finalizado en {iter_count[0]} iteraciones.")

    # ==========================================
    # 4. EVALUACIÓN DE PARÁMETROS DESCUBIERTOS
    # ==========================================
    lambda_1_pred = model.lambda_1.item()
    lambda_2_pred = torch.exp(model.lambda_2).item()

    error_lambda_1 = np.abs(lambda_1_pred - lambda_1_exact) / lambda_1_exact * 100
    error_lambda_2 = np.abs(lambda_2_pred - lambda_2_exact) / lambda_2_exact * 100

    print(f"\n--- Resultados del Descubrimiento (Ruido: {noise_level*100}%) ---")
    print(f"Lambda 1 exacto: {lambda_1_exact:.6f}  | Predicho: {lambda_1_pred:.6f}  | Error: {error_lambda_1:.3f}%")
    print(f"Lambda 2 exacto: {lambda_2_exact:.6f} | Predicho: {lambda_2_pred:.6f} | Error: {error_lambda_2:.3f}%")

    # ==========================================
    # 5. VISUALIZACIÓN EN DOS VENTANAS
    # ==========================================
    model.eval()
    with torch.no_grad():
        x_star_tensor = torch.tensor(X_star[:, 0:1], dtype=torch.float32, device=device)
        t_star_tensor = torch.tensor(X_star[:, 1:2], dtype=torch.float32, device=device)
        u_pred = model(x_star_tensor, t_star_tensor).cpu().detach().numpy()

    U_pred = u_pred.reshape(Exact.shape)

    # ----------------------------------------------------
    # VENTANA 1: Mapa espaciotemporal con puntos de datos
    # ----------------------------------------------------
    plt.figure(figsize=(9, 4.5))
    h = plt.imshow(U_pred, interpolation='nearest', cmap='rainbow', 
                   extent=[t.min(), t.max(), x.min(), x.max()], 
                   origin='lower', aspect='auto')
    plt.colorbar(h)
    plt.scatter(t_train_np, x_train_np, color='black', marker='x', s=10, label=f'Data ({N_train} points)')

    for t_val in [0.25, 0.50, 0.75]:
        plt.axvline(x=t_val, color='w', linestyle='--', linewidth=1.5)

    plt.xlabel('$t$')
    plt.ylabel('$x$')
    plt.title(f'$u(t,x)$ - PINN Inversa (Ruido: {noise_level*100}%)', fontsize=12)
    plt.legend(loc='upper right', frameon=True)
    plt.tight_layout()
    plt.show()

    # ----------------------------------------------------
    # VENTANA 2: Cortes transversales en t = 0.25, 0.50, 0.75
    # ----------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    times = [0.25, 0.50, 0.75]

    for i, t_val in enumerate(times):
        idx_t = np.argmin(np.abs(t.flatten() - t_val))
        
        axes[i].plot(x.flatten(), Exact[:, idx_t], 'b-', linewidth=2, label='Exact')
        axes[i].plot(x.flatten(), U_pred[:, idx_t], 'r--', linewidth=2, label='Prediction')
        axes[i].set_xlabel('$x$')
        axes[i].set_ylabel('$u(t,x)$')
        axes[i].set_title(f'$t = {t_val:.2f}$ (Ruido: {noise_level*100}%)')
        axes[i].set_xlim([-1, 1])
        if i == 0:
            axes[i].legend(loc='lower left', frameon=True)

    plt.tight_layout()
    plt.show()