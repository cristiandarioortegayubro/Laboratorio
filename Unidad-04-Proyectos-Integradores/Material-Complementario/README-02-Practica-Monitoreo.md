# 💻 Módulo 02 - Práctica: Detección de Drift

## Ejercicios de Data Drift y Concept Drift

```python
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt

# Generar datos de entrenamiento
np.random.seed(42)
n_train = 1000
X_train = np.random.normal(loc=50, scale=10, size=n_train)
y_train = (X_train + np.random.randn(n_train) * 5 > 55).astype(int)

print("✅ Datos de entrenamiento generados")
print(f"  Media de X: {X_train.mean():.2f}")
print(f"  Tasa positiva: {y_train.mean():.2%}")

# Ejercicio 1: Simulación de Data Drift
print("\n🔍 Ejercicio 1: Detectar Data Drift")

# Simular datos de producción con drift
n_prod = 500
X_prod_no_drift = np.random.normal(loc=50, scale=10, size=n_prod)
X_prod_with_drift = np.random.normal(loc=60, scale=12, size=n_prod)  # ⚠️ Drift

# KS Test para detectar drift
ks_stat_no_drift, p_value_no_drift = ks_2samp(X_train, X_prod_no_drift)
ks_stat_with_drift, p_value_with_drift = ks_2samp(X_train, X_prod_with_drift)

print("\nSin Drift:")
print(f"  KS Statistic: {ks_stat_no_drift:.4f}")
print(f"  P-value: {p_value_no_drift:.4f}")
print(f"  Resultado: {'Sin drift' if p_value_no_drift > 0.05 else 'Drift detectado'} ✅")

print("\nCon Drift:")
print(f"  KS Statistic: {ks_stat_with_drift:.4f}")
print(f"  P-value: {p_value_with_drift:.4f}")
print(f"  Resultado: {'Sin drift' if p_value_with_drift > 0.05 else 'Drift detectado'} ⚠️")

# Ejercicio 2: Population Stability Index (PSI)
print("\n📊 Ejercicio 2: Calcular PSI")

def calculate_psi(expected, actual, bins=10):
    expected_counts, bin_edges = np.histogram(expected, bins=bins)
    actual_counts, _ = np.histogram(actual, bins=bin_edges)
    
    # Evitar división por cero
    expected_pct = (expected_counts + 1) / (len(expected) + bins)
    actual_pct = (actual_counts + 1) / (len(actual) + bins)
    
    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return psi

psi_no_drift = calculate_psi(X_train, X_prod_no_drift)
psi_with_drift = calculate_psi(X_train, X_prod_with_drift)

print(f"\nPSI sin drift: {psi_no_drift:.4f}")
if psi_no_drift < 0.1:
    print("  Interpretación: Sin drift ✅")
elif psi_no_drift < 0.2:
    print("  Interpretación: Drift moderado ⚠️")
else:
    print("  Interpretación: Drift significativo ❌")

print(f"\nPSI con drift: {psi_with_drift:.4f}")
if psi_with_drift < 0.1:
    print("  Interpretación: Sin drift ✅")
elif psi_with_drift < 0.2:
    print("  Interpretación: Drift moderado ⚠️")
else:
    print("  Interpretación: Drift significativo ❌")

# Ejercicio 3: Visualización de Drift
print("\n📊 Ejercicio 3: Visualizar distribuciones")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Sin drift
axes[0].hist(X_train, bins=30, alpha=0.5, label='Train', density=True)
axes[0].hist(X_prod_no_drift, bins=30, alpha=0.5, label='Prod (sin drift)', density=True)
axes[0].set_title('Sin Data Drift')
axes[0].set_xlabel('Valor de Feature')
axes[0].set_ylabel('Densidad')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Con drift
axes[1].hist(X_train, bins=30, alpha=0.5, label='Train', density=True)
axes[1].hist(X_prod_with_drift, bins=30, alpha=0.5, label='Prod (con drift)', density=True)
axes[1].set_title('Con Data Drift')
axes[1].set_xlabel('Valor de Feature')
axes[1].set_ylabel('Densidad')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
display(plt.gcf())
plt.close()

# Ejercicio 4: Simulación de Concept Drift
print("\n🔄 Ejercicio 4: Detectar Concept Drift")

# Entrenar modelo simple
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

model = LogisticRegression()
model.fit(X_train.reshape(-1, 1), y_train)

print("✅ Modelo entrenado")

# Producción sin concept drift
y_prod_no_concept_drift = (X_prod_no_drift + np.random.randn(n_prod) * 5 > 55).astype(int)
pred_no_concept_drift = model.predict(X_prod_no_drift.reshape(-1, 1))
acc_no_concept_drift = accuracy_score(y_prod_no_concept_drift, pred_no_concept_drift)

print(f"\nSin Concept Drift:")
print(f"  Accuracy: {acc_no_concept_drift:.4f} ✅")

# Producción con concept drift (relación X->Y cambió)
y_prod_with_concept_drift = (X_prod_no_drift + np.random.randn(n_prod) * 5 > 45).astype(int)  # Threshold cambió
pred_with_concept_drift = model.predict(X_prod_no_drift.reshape(-1, 1))
acc_with_concept_drift = accuracy_score(y_prod_with_concept_drift, pred_with_concept_drift)

print(f"\nCon Concept Drift:")
print(f"  Accuracy: {acc_with_concept_drift:.4f} ❌")
print(f"  Degradación: {(acc_no_concept_drift - acc_with_concept_drift)*100:.1f}%")

if acc_with_concept_drift < acc_no_concept_drift - 0.05:
    print("\n⚠️ Concept drift detectado - Reentrenar modelo recomendado")

# Ejercicio 5: Dashboard de Monitoreo
print("\n📊 Ejercicio 5: Resumen de Monitoreo")

monitoring_summary = pd.DataFrame({
    'Métrica': [
        'KS Statistic (Data Drift)',
        'PSI (Data Drift)',
        'Accuracy (Concept Drift)'
    ],
    'Sin Drift': [
        f"{ks_stat_no_drift:.4f}",
        f"{psi_no_drift:.4f}",
        f"{acc_no_concept_drift:.4f}"
    ],
    'Con Drift': [
        f"{ks_stat_with_drift:.4f}",
        f"{psi_with_drift:.4f}",
        f"{acc_with_concept_drift:.4f}"
    ],
    'Estado': [
        '⚠️ Drift' if p_value_with_drift < 0.05 else '✅ OK',
        '❌ Drift Significativo' if psi_with_drift > 0.2 else '✅ OK',
        '⚠️ Degradación' if acc_with_concept_drift < 0.75 else '✅ OK'
    ]
})

print("\nDashboard de Monitoreo:")
print(monitoring_summary.to_string(index=False))

print("\n✅ Ejercicios de monitoreo completados")
print("💡 En producción, estos análisis correrían automáticamente")
print("💡 Alertas se enviarían cuando se detecte drift")
```

---

**Universidad del Aconcagua 🇦🇷**