import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Parámetros del problema
alpha = 0.8
k = 60
nu = 0.25
A0 = 1
h = 1
t_values = np.arange(0, 31, h)  # de 0 a 30 días inclusive

# Definición de la función f(A)
def f(A):
    return alpha * A * (1 - (A / k) ** nu)

# Método de Euler
def euler(A0):
    A = A0
    tabla = []
    for t in t_values[:-1]:
        val_f = f(A)
        A_next = A + h * val_f
        tabla.append({
            't_i': int(t),
            'A_i': round(A, 2),
            'f(A)': round(val_f, 2),
            'A_{i+1}': round(A_next, 2)
        })
        A = A_next
    return tabla

# Método de Heun
def heun(A0):
    A = A0
    tabla = []
    for t in t_values[:-1]:
        predictor = A + h * f(A)
        corrector = A + h * 0.5 * (f(A) + f(predictor))
        tabla.append({
            't_i': int(t),
            'A_i': round(A, 2),
            'pred': round(predictor, 2),
            'corr': round(corrector, 2)
        })
        A = corrector
    return tabla

# Método de Runge-Kutta 4
def rk4(A0):
    A = A0
    tabla = []
    for t in t_values[:-1]:
        k1 = f(A)
        k2 = f(A + 0.5 * h * k1)
        k3 = f(A + 0.5 * h * k2)
        k4 = f(A + h * k3)
        A_next = A + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        tabla.append({
            't_i': int(t),
            'A_i': round(A, 2),
            'k1': round(k1, 2),
            'k2': round(k2, 2),
            'k3': round(k3, 2),
            'k4': round(k4, 2),
            'A_{i+1}': round(A_next, 2)
        })
        A = A_next
    return tabla

# Ejecutar los métodos
tabla_euler = euler(A0)
tabla_heun = heun(A0)
tabla_rk4 = rk4(A0)

# Mostrar tablas
print("\nTabla: Método de Euler")
print(pd.DataFrame(tabla_euler).to_string(index=False))

print("\nTabla: Método de Heun")
print(pd.DataFrame(tabla_heun).to_string(index=False))

print("\nTabla: Método de Runge-Kutta 4")
print(pd.DataFrame(tabla_rk4).to_string(index=False))

# Obtener A(t) finales para graficar
A_euler = [row['A_i'] for row in tabla_euler] + [tabla_euler[-1]['A_{i+1}']]
A_heun = [row['A_i'] for row in tabla_heun] + [tabla_heun[-1]['corr']]
A_rk4 = [row['A_i'] for row in tabla_rk4] + [tabla_rk4[-1]['A_{i+1}']]

# Tabla comparativa final
df_final = pd.DataFrame({
    "t (días)": t_values,
    "Euler": A_euler,
    "Heun": A_heun,
    "RK4": A_rk4
})
print("\nTabla comparativa final:")
print(df_final.to_string(index=False))

# Valores finales
print(f"\nValor final de Euler en t = {t_values[-1]} es: {A_euler[-1]:.2f} mm²")
print(f"Valor final de Heun en t = {t_values[-1]} es: {A_heun[-1]:.2f} mm²")
print(f"Valor final de RK4 en t = {t_values[-1]} es: {A_rk4[-1]:.2f} mm²")

# Graficar resultados
plt.figure(figsize=(10, 6))
plt.plot(t_values, A_euler, label="Euler", marker='o')
plt.plot(t_values, A_heun, label="Heun", marker='s')
plt.plot(t_values, A_rk4, label="Runge-Kutta 4", marker='^')
plt.axhline(y=k, color='r', linestyle='--', label='Límite del tumor (k = 60)')
plt.title("Crecimiento del Tumor (Área A vs. Tiempo)")
plt.xlabel("Tiempo (días)")
plt.ylabel("Área del tumor A(t) [mm²]")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
