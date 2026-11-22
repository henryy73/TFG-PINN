# TFG-PINN: Physics-Informed Neural Networks for Burgers' Equation

Repositorio oficial para el Trabajo de Fin de Grado (TFG) centrado en la resolución e identificación de la ecuación de Burgers mediante métodos analíticos, numéricos y Redes Neuronales Informadas por la Física (**PINNs**).

---

## 📁 Estructura del Repositorio

El proyecto se organiza en seis directorios principales que abarcan los datos de referencia, las formulaciones continuas y discretas (problemas directo e inverso) y estudios comparativos adicionales:

- **`data/`**: Datos de entrada y pesos necesarios para los experimentos.
  - `burgers_shock.mat`: Solución de referencia de la ecuación de Burgers viscosa.
  - `IRK_weights/`: Pesos de los esquemas Implicit Runge-Kutta (IRK) utilizados en las formulaciones discretas.

- **`Direct_continuous/`**: Resolución del **problema directo** bajo una formulación **continua** en espacio y tiempo.
  - Scripts: `burgers_pinn_cont.py`, `burgers_table_dir_cont.py`.
  - Resultados: Tablas de métricas (`burgers_tabla_A1.csv`, `burgers_tabla_A2.csv`) y visualizaciones (`Foto_color_1.png`, `Foto_gráficos_1.png`).

- **`Direct_discrete/`**: Resolución del **problema directo** utilizando una formulación **discreta en el tiempo** (esquemas IRK).
  - Scripts: `burguers_pinn_dis.py`, `burguers_table_dir_dis.py`.
  - Resultados: Tablas de métricas (`burgers_tabla_A3.csv`, `burgers_tabla_A4.csv`, `burgers_tabla_A5.csv`) y visualizaciones (`Foto_color_2.png`, `Foto_graficos_2.png`).

- **`Discovery_continuous/`**: Resolución del **problema inverso** (descubrimiento de parámetros de la PDE) bajo un enfoque **continuo**.
  - Scripts: `burgers_discovery_cont.py`, `burgers_tabla_disc_cont.py`.
  - Resultados: Tablas de error (`burgers_tabla_B6.csv`, `burgers_tabla_B7.csv`) y visualizaciones (`Foto_color_3.png`, `Foto_grafico_3.png`).

- **`Discovery_discrete/`**: **Descubrimiento de parámetros** utilizando esquemas **discretos** (IRK).
  - Scripts: `burgers_discovery_disc.py`, `burgers_tabla_disc_dis.py`.
  - Resultados: Tablas de error (`burgers_tabla_B8.csv`, `burgers_tabla_B9.csv`) y visualizaciones (`Foto_color_4.png`, `Foto_gráfico 4.png`).

- **`Exploracion_adicional/`**: Experimentos complementarios, comparativas metodológicas y figuras esquemáticas.
  - **Diferenciación Automática vs. Diferencias Finitas**: `exp_ad_vs_fd.py` y `burgers_tabla_AD_vs_FD.csv`.
  - **Estudio de hiperparámetros**: `exp_hiperparametros.py` y `burgers_tabla_hiperparametros.csv`.
  - **PINN vs. Métodos Numéricos**: `exp_pinn_vs_numerico.py` y `burgers_tabla_PINN_vs_Numerico.csv`.
  - **Figuras**: `Esquema_PINN.png`, `Solucion_analitica_Burgers.png`, `Diferencias_finitas_Burgers.png`.

---

## 📊 Contenido y Resultados

- **Scripts de entrenamiento y evaluación**: Implementaciones en Python (PyTorch) con optimizadores basados en gradiente (Adam y L-BFGS) para reproducir tanto las soluciones como las tablas comparativas.
- **Tablas de resultados (`.csv`)**: Métricas de error relativo en norma $L_2$, análisis de sensibilidad a hiperparámetros y comparaciones entre diferenciación automática, diferencias finitas y PINNs.
- **Gráficas (`.png`)**: Mapas de calor espacio-temporales, cortes temporales comparativos y esquemas de arquitectura.

---

## 🚀 Requisitos y Ejecución

Asegúrate de tener instaladas las librerías necesarias (PyTorch, NumPy, Matplotlib, SciPy, Pandas):

```bash
pip install torch numpy matplotlib scipy pandaslotlib scipy
