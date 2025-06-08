import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Parámetros del problema
k = 0.000095
N_max = 5000
N0 = 100
h = 1
t_values = np.arange(0, 20 + h, h)  # de 0 a 20 años inclusive

# Definición de la función f(N)
def f(N):
    return k * N * (N_max - N)

# Método de Euler
def euler(N0):
    N = N0
    tabla = []
    for n, t in enumerate(t_values[:-1]):
        val_f = f(N)
        N_next = N + h * val_f
        tabla.append({
            'n': n,
            't_i': int(t),
            'N_i': round(N),
            'f(N)': round(val_f),
            'N_{i+1}': round(N_next)
        })
        N = N_next  # No redondear aquí
    return tabla

# Método de Heun
def heun(N0):
    N = N0
    tabla = []
    for n, t in enumerate(t_values[:-1]):
        predictor = N + h * f(N)
        corrector = N + h * 0.5 * (f(N) + f(predictor))
        tabla.append({
            'n': n,
            't_i': int(t),
            'N_i': round(N),
            'pred': round(predictor),
            'corr': round(corrector)
        })
        N = corrector  # No redondear aquí
    return tabla

# Método de Runge-Kutta 4
def rk4(N0):
    N = N0
    tabla = []
    for n, t in enumerate(t_values[:-1]):
        k1 = f(N)
        k2 = f(N + 0.5 * h * k1)
        k3 = f(N + 0.5 * h * k2)
        k4 = f(N + h * k3)
        N_next = N + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        tabla.append({
            'n': n,
            't_i': int(t),
            'N_i': round(N),
            'k1': round(k1),
            'k2': round(k2),
            'k3': round(k3),
            'k4': round(k4),
            'N_{i+1}': round(N_next)
        })
        N = N_next  # No redondear aquí
    return tabla

# Ejecutar los métodos
tabla_euler = euler(N0)
tabla_heun = heun(N0)
tabla_rk4 = rk4(N0)

# Mostrar tablas
print("\nTabla: Método de Euler")
df_euler = pd.DataFrame(tabla_euler)
print(df_euler.to_string(index=False))

print("\nTabla: Método de Heun")
df_heun = pd.DataFrame(tabla_heun)
print(df_heun.to_string(index=False))

print("\nTabla: Método de Runge-Kutta 4")
df_rk4 = pd.DataFrame(tabla_rk4)
print(df_rk4.to_string(index=False))

# Obtener N(t) finales para graficar
N_euler = [row['N_i'] for row in tabla_euler] + [tabla_euler[-1]['N_{i+1}']]
N_heun = [row['N_i'] for row in tabla_heun] + [tabla_heun[-1]['corr']]
N_rk4 = [row['N_i'] for row in tabla_rk4] + [tabla_rk4[-1]['N_{i+1}']]

# Tabla comparativa final
df_final = pd.DataFrame({
    "t (años)": t_values.astype(int),
    "Euler": N_euler,
    "Heun": N_heun,
    "RK4": N_rk4
})

print("\nTabla comparativa final:")
print(df_final.to_string(index=False))

# Valores finales
print(f"\nValor final de Euler en t = {t_values[-1]} es: {N_euler[-1]}")
print(f"Valor final de Heun en t = {t_values[-1]} es: {N_heun[-1]}")
print(f"Valor final de RK4 en t = {t_values[-1]} es: {N_rk4[-1]}")

# Graficar resultados
plt.figure(figsize=(10, 6))
plt.plot(t_values, N_euler, label="Euler", marker='o')
plt.plot(t_values, N_heun, label="Heun", marker='s')
plt.plot(t_values, N_rk4, label="Runge-Kutta 4", marker='^')
plt.axhline(y=N_max, color='r', linestyle='--', label='Capacidad máxima (N_M)')
plt.title("Crecimiento Poblacional con Capacidad Limitada")
plt.xlabel("Tiempo (años)")
plt.ylabel("Población N(t)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
