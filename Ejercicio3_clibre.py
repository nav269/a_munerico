import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Parámetros del problema de caída libre
m = 5  # kg
g = 9.81  # m/s^2
k_drag = 0.05  # kg/m
v0 = 0  # m/s (condición inicial de velocidad)
h = 0.5  # Tamaño de paso
t_values = np.arange(0, 15.1, h)  # De 0 a 15 segundos inclusive

# Función dv/dt
def f_velocity(v):
    return -g + (k_drag / m) * (v**2)  # Corregido: gravedad negativa, resistencia positiva

# Método de Euler
def euler(v0):
    v = v0
    velocities = [round(v0, 2)]
    table_data = []
    for t_i in t_values[:-1]:
        dv_dt = f_velocity(v)
        v_next = v + h * dv_dt
        table_data.append({
            't_i': round(t_i, 2),
            'v_i': round(v, 2),
            'f(v)': round(dv_dt, 2),
            'v_{i+1}': round(v_next, 2)
        })
        v = v_next
        velocities.append(round(v, 2))
    return table_data, velocities

# Método de Heun
def heun(v0):
    v = v0
    velocities = [round(v0, 2)]
    table_data = []
    for t_i in t_values[:-1]:
        predictor = v + h * f_velocity(v)
        corrector = v + h * 0.5 * (f_velocity(v) + f_velocity(predictor))
        table_data.append({
            't_i': round(t_i, 2),
            'v_i': round(v, 2),
            'pred': round(predictor, 2),
            'corr': round(corrector, 2)
        })
        v = corrector
        velocities.append(round(v, 2))
    return table_data, velocities

# Método de Runge-Kutta 4
def rk4(v0):
    v = v0
    velocities = [round(v0, 2)]
    table_data = []
    for t_i in t_values[:-1]:
        k1 = f_velocity(v)
        k2 = f_velocity(v + 0.5 * h * k1)
        k3 = f_velocity(v + 0.5 * h * k2)
        k4 = f_velocity(v + h * k3)
        v_next = v + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        table_data.append({
            't_i': round(t_i, 2),
            'v_i': round(v, 2),
            'k1': round(k1, 2),
            'k2': round(k2, 2),
            'k3': round(k3, 2),
            'k4': round(k4, 2),
            'v_{i+1}': round(v_next, 2)
        })
        v = v_next
        velocities.append(round(v, 2))
    return table_data, velocities

# Ejecutar métodos
table_euler, v_euler_plot = euler(v0)
table_heun, v_heun_plot = heun(v0)
table_rk4, v_rk4_plot = rk4(v0)

# Mostrar resultados en tablas
print("---")
print("\nTabla de Resultados: Método de Euler")
print(pd.DataFrame(table_euler).to_string(index=False))

print("\nTabla de Resultados: Método de Heun")
print(pd.DataFrame(table_heun).to_string(index=False))

print("\nTabla de Resultados: Método de Runge-Kutta 4")
print(pd.DataFrame(table_rk4).to_string(index=False))

# Tabla comparativa
df_final = pd.DataFrame({
    "Tiempo (s)": t_values,
    "Euler": v_euler_plot,
    "Heun": v_heun_plot,
    "RK4": v_rk4_plot
})
print("---")
print("\nTabla Comparativa Final de Velocidades:")
print(df_final.to_string(index=False))

# Velocidad terminal teórica
vt_magnitude = np.sqrt((m * g) / k_drag)
print("---")
print(f"\nVelocidad terminal teórica: {vt_magnitude:.2f} m/s")
print(f"Velocidad final de Euler en t = {t_values[-1]} s: {v_euler_plot[-1]:.2f} m/s")
print(f"Velocidad final de Heun en t = {t_values[-1]} s: {v_heun_plot[-1]:.2f} m/s")
print(f"Velocidad final de RK4 en t = {t_values[-1]} s: {v_rk4_plot[-1]:.2f} m/s")

# Graficar resultados
plt.figure(figsize=(12, 7))
plt.plot(t_values, v_euler_plot, label="Euler", marker='o', markersize=6, linestyle='-')
plt.plot(t_values, v_heun_plot, label="Heun", marker='s', markersize=6, linestyle='--')
plt.plot(t_values, v_rk4_plot, label="Runge-Kutta 4", marker='^', markersize=6, linestyle='-.')
plt.axhline(y=vt_magnitude, color='r', linestyle=':', label='Velocidad Terminal Teórica')
plt.title('Velocidad de un Objeto en Caída Libre vs. Tiempo (h = 0.5 s)')
plt.xlabel('Tiempo (s)')
plt.ylabel('Velocidad (m/s)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
