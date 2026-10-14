# TFG-PINN: Physics-Informed Neural Networks for Burgers' Equation

Repositorio oficial para el Trabajo de Fin de Grado (TFG) centrado en la resolución e identificación de la ecuación de Burgers utilizando Redes Neuronales Informadas por la Física (**PINNs**).

---

## 📁 Estructura del Repositorio

El proyecto se divide en las siguientes carpetas principales, explorando formulaciones continuas y discretas tanto para problemas directos como inversos:

* **`direct cont/`**: Códigos y scripts para la resolución del **problema directo** bajo una formulación **continua** (evaluación de la PDE en puntos aleatorios del dominio espacio-temporal).
* **`direct discrete/`**: Implementaciones para el **problema directo** utilizando una formulación **discreta en el tiempo** (esquemas temporales acoplados en la red).
* **`discovery cont/`**: Experimentos orientados al **problema inverso** (descubrimiento de parámetros ocultos o coeficientes de la ecuación) bajo un enfoque **continuo**.
* **`discovery discrete/`**: Enfoques de **descubrimiento de parámetros** utilizando esquemas **discretos**.

---

## 📊 Contenido y Resultados

* **Scripts de entrenamiento**: Modelos implementados en Python (PyTorch) para el entrenamiento de las redes con optimizadores basados en gradiente (Adam y L-BFGS).
* **Tablas de resultados**: Métricas de error (como el error relativo en norma $L_2$) y comparaciones de rendimiento entre las distintas arquitecturas y enfoques metodológicos.

---

## 🚀 Requisitos y Ejecución

Asegúrate de tener instaladas las librerías necesarias (PyTorch, NumPy, Matplotlib, etc.):

```bash
pip install torch numpy matplotlib scipy
