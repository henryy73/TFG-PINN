# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 12:27:26 2026

@author: ematm
"""

import os
import torch
import torch.nn as nn
import numpy as np
import scipy.io
from scipy.stats import qmc
import pandas as pd
import time

# 1. Definición del Modelo Físico (Idéntico a tu script)
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

# 2. Función principal de evaluación
def evaluar_configuracion(N_u, N_f, max_iter=2000): 
    # Cargar y preparar datos base
    data = scipy.io.loadmat(r'C:\Users\ematm\OneDrive\Desktop\TFG\Trabajo TFG propuesto\data\burgers_shock.mat')  
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
    
    # Generación de N_f
    lb = X_star.min(0)
    ub = X_star.max(0)
    sampler = qmc.LatinHypercube(d=2)
    sample = sampler.random(n=N_f)
    X_f_train = lb + (ub - lb) * sample
    X_f_train = np.vstack((X_f_train, X_u_train))
    
    # Tensores
    x_u = torch.tensor(X_u_train[:, 0:1], dtype=torch.float32)
    t_u = torch.tensor(X_u_train[:, 1:2], dtype=torch.float32)
    u_train_tensor = torch.tensor(u_train, dtype=torch.float32)
    x_f = torch.tensor(X_f_train[:, 0:1], dtype=torch.float32, requires_grad=True)
    t_f = torch.tensor(X_f_train[:, 1:2], dtype=torch.float32, requires_grad=True)
    
    # Inicialización limpia del modelo
    layers = [2, 20, 20, 20, 20, 20, 20, 20, 20, 1]
    model = PhysicsInformedNN(layers)
    optimizer = torch.optim.LBFGS(
        model.parameters(), lr=1.0, max_iter=max_iter, max_eval=max_iter, 
        tolerance_grad=1e-5, tolerance_change=1.0 * np.finfo(float).eps, line_search_fn="strong_wolfe"
    )
    
    def closure():
        optimizer.zero_grad()
        loss = loss_function(model, x_u, t_u, u_train_tensor, x_f, t_f)
        loss.backward()
        return loss
    
    optimizer.step(closure)
    
    # Evaluación del L2
    model.eval()
    with torch.no_grad():
        x_star_tensor = torch.tensor(X_star[:, 0:1], dtype=torch.float32)
        t_star_tensor = torch.tensor(X_star[:, 1:2], dtype=torch.float32)
        u_pred = model(x_star_tensor, t_star_tensor).detach().numpy().reshape(Exact.shape)
        
    error_l2 = np.linalg.norm(Exact - u_pred, 2) / np.linalg.norm(Exact, 2)
    return error_l2

# 3. Bucle para generar la Tabla A.1 (Con Guardado Continuo)
if __name__ == "__main__":
    Nu_list = [20, 40, 60, 80, 100, 200]
    Nf_list = [2000, 4000, 6000, 7000, 8000, 10000]
    
    # Asegurar la creación de la carpeta
    ruta_guardado = r'C:\Users\ematm\OneDrive\Desktop\TFG\Trabajo TFG propuesto\Burgers\Direct_continuous'
    os.makedirs(ruta_guardado, exist_ok=True)  
    archivo_salida = os.path.join(ruta_guardado, 'burgers_tabla_A1.csv')
    
    # Cargar tabla si ya existe (para retomar progreso)
    if os.path.exists(archivo_salida):
        print("Cargando tabla de resultados parcial existente...")
        resultados = pd.read_csv(archivo_salida, index_col=0)
        # Convertir nombres de columnas a int para que coincidan con Nf_list
        resultados.columns = [int(float(c)) for c in resultados.columns] 
        resultados = resultados.astype(object)
    else:
        # Crear tabla nueva vacía
        resultados = pd.DataFrame(index=Nu_list, columns=Nf_list, dtype=object)
        resultados.index.name = "N_u / N_f"
    
    print("\nIniciando generación de la Tabla A.1...")
    t_global = time.time()
    
    for nu in Nu_list:
        for nf in Nf_list:
            val = resultados.at[nu, nf]
            
            # Si ya hay un valor numérico guardado (y no es un error), saltamos esta celda
            try:
                if pd.notna(val) and str(val) != "ERROR":
                    print(f"Saltando Nu={nu}, Nf={nf} (Ya calculado: {val})")
                    continue
            except ValueError:
                pass
            
            print(f"\n======================================")
            print(f"Ejecutando Nu={nu}, Nf={nf}")
            print(f"======================================")
            
            t0 = time.time()
            try:
                # OJO: Cambia el max_iter a 50000 cuando lances la prueba definitiva
                error = evaluar_configuracion(nu, nf, max_iter=1500) 
                
                # Guardar el error en notación científica
                formatted_error = f"{error:.1e}"
                resultados.at[nu, nf] = formatted_error
                
                print(f"  -> FINALIZADO | Error L2: {formatted_error} | Tiempo: {time.time()-t0:.1f}s")
                
                # Guardado Inmediato
                resultados.to_csv(archivo_salida)
                
            except Exception as e:
                print(f"  -> ERROR CRÍTICO en Nu={nu}, Nf={nf}:")
                import traceback
                traceback.print_exc()
                
                resultados.at[nu, nf] = "ERROR"
                resultados.to_csv(archivo_salida)

    print("\n======================================")
    print("        TABLA A.1 COMPLETADA          ")
    print("======================================")
    print(f"Tiempo total invertido: {(time.time() - t_global)/3600:.2f} horas")
    
    # Imprimir tabla final formateada en consola
    def formato_cientifico(val):
        try:
            return f"{float(val):.1e}"
        except:
            return val

    tabla_imprimir = resultados.map(formato_cientifico) if hasattr(resultados, 'map') else resultados.applymap(formato_cientifico)
    print(tabla_imprimir)