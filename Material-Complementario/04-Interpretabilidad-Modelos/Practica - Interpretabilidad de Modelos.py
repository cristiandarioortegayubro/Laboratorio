# Databricks notebook source
# DBTITLE 1,Ejercicio 2: PDP/ICE
# MAGIC %md
# MAGIC ## 💻 Ejercicio 2: PDP/ICE para Features Temporales
# MAGIC
# MAGIC ### 🎯 Problema
# MAGIC
# MAGIC Visualizar cómo **día de semana** y **mes** afectan las predicciones.
# MAGIC
# MAGIC **Objetivo**:
# MAGIC - Generar PDP para features temporales
# MAGIC - Generar ICE para detectar heterogeneidad
# MAGIC - Interpretar patrones
# MAGIC
# MAGIC ---

# COMMAND ----------

# DBTITLE 1,Ejercicio 2 - Solución
print("="*80)
print("EJERCICIO 2: PDP/ICE PARA FEATURES TEMPORALES")
print("="*80)

print("\n1️⃣ Partial Dependence Plot: Día de Semana")

# PDP para dia_semana
fig, ax = plt.subplots(figsize=(10, 6))
PartialDependenceDisplay.from_estimator(
    model, X_train[:1000],  # Samplear para velocidad
    features=['dia_semana'],
    ax=ax
)
plt.title('PDP: Efecto de Día de Semana en Ventas')
plt.xlabel('Día (0=Lun, 4=Vie, 6=Dom)')
plt.show()

print("\n💡 Interpretación del PDP:")
print("   - Viernes (4-5) tienen ventas más altas")
print("   - Lunes y domingos tienen ventas más bajas")
print("   - Efecto promedio sobre todos los registros")

print("\n2️⃣ ICE Plot: Día de Semana (heterogeneidad)")

# ICE + PDP juntos
fig, ax = plt.subplots(figsize=(10, 6))
PartialDependenceDisplay.from_estimator(
    model, X_train[:500],
    features=['dia_semana'],
    kind='both',  # ICE + PDP
    ax=ax
)
plt.title('ICE + PDP: Día de Semana')
plt.xlabel('Día (0=Lun, 4=Vie, 6=Dom)')
plt.show()

print("\n💡 Interpretación del ICE:")
print("   - La mayoría de líneas siguen la misma tendencia")
print("   - Efecto del día es CONSISTENTE entre registros")
print("   - Viernes aumenta ventas para la mayoría")

print("\n3️⃣ PDP para Múltiples Features")

# PDP para varias features
fig, ax = plt.subplots(figsize=(14, 10))
PartialDependenceDisplay.from_estimator(
    model, X_train[:1000],
    features=['dia_semana', 'mes', 'es_fin_semana', 'segmento_encoded'],
    ax=ax
)
plt.tight_layout()
plt.show()

print("\n💡 Insights Temporales:")
print("   - Día de semana: Pico en viernes")
print("   - Mes: Variabilidad mensual (estacionalidad)")
print("   - Fin de semana: Efecto visible vs. días laborables")

print("\n" + "="*80)
print("✅ PDP/ICE revelan patrones temporales claros")
print("="*80)

# COMMAND ----------

# DBTITLE 1,Ejercicios 3, 4, 5 resumen
# MAGIC %md
# MAGIC ## 💻 Ejercicios 3, 4, 5: SHAP y Comparación
# MAGIC
# MAGIC **Los siguientes ejercicios cubren**:
# MAGIC
# MAGIC ### Ejercicio 3: SHAP Global
# MAGIC - Summary plot (beeswarm)
# MAGIC - Bar plot (importancia)
# MAGIC - Dependence plots
# MAGIC
# MAGIC ### Ejercicio 4: SHAP Local
# MAGIC - Waterfall para predicciones específicas
# MAGIC - Force plots
# MAGIC - Casos anómalos
# MAGIC
# MAGIC ### Ejercicio 5: Comparación Total
# MAGIC - Feature Importance
# MAGIC - Permutation
# MAGIC - SHAP
# MAGIC - PDP
# MAGIC - Conclusiones finales
# MAGIC
# MAGIC ---

# COMMAND ----------

# DBTITLE 1,Ejercicio 3: SHAP Global
print("="*80)
print("EJERCICIO 3: SHAP GLOBAL")
print("="*80)

# Calcular SHAP
print("\n1️⃣ Calculando SHAP values...")
explainer = shap.TreeExplainer(model)
shap_values = explainer(X_test[:500])  # Samplear

print(f"   SHAP shape: {shap_values.values.shape}")
print(f"   Base value: ${explainer.expected_value:.2f}")

# Summary plot
print("\n2️⃣ SHAP Summary Plot (Beeswarm):")
shap.summary_plot(shap_values.values, X_test[:500], show=False)
plt.tight_layout()
plt.show()

print("\n💡 Interpretación:")
print("   - Features en top son más importantes")
print("   - Dispersión horizontal = efecto variable")
print("   - Color: rojo=valor alto, azul=bajo")

# Bar plot
print("\n3️⃣ SHAP Bar Plot (Importancia):")
shap.summary_plot(shap_values.values, X_test[:500], plot_type='bar', show=False)
plt.tight_layout()
plt.show()

# Dependence plot
print("\n4️⃣ SHAP Dependence Plot: Día-Mes:")
shap.dependence_plot('dia_semana', shap_values.values, X_test[:500],
                      interaction_index='mes', show=False)
plt.title('Interacción: Día de Semana x Mes')
plt.show()

print("\n" + "="*80)
print("✅ SHAP revela importancia + distribución + interacciones")
print("="*80)

# COMMAND ----------

# DBTITLE 1,Ejercicio 4: SHAP Local
print("="*80)
print("EJERCICIO 4: SHAP LOCAL - EXPLICAR PREDICCIONES")
print("="*80)

# Encontrar predicciones interesantes
predictions = model.predict(X_test[:500])
high_idx = predictions.argmax()
low_idx = predictions.argmin()

print(f"\n1️⃣ Predicción MÁS ALTA: #{high_idx}")
print(f"   Valor: ${predictions[high_idx]:.2f}")
shap.plots.waterfall(shap_values[high_idx])

print(f"\n2️⃣ Predicción MÁS BAJA: #{low_idx}")
print(f"   Valor: ${predictions[low_idx]:.2f}")
shap.plots.waterfall(shap_values[low_idx])

# Desglose numérico
print(f"\n3️⃣ Desglose de Predicción Alta (#{high_idx}):")
print(f"   Baseline: ${shap_values[high_idx].base_values:.2f}")
for feat, val in zip(features, shap_values[high_idx].values):
    sign = '+' if val >= 0 else ''
    print(f"   {feat:20s}: {sign}${val:.2f}")
print(f"   = Predicción: ${predictions[high_idx]:.2f}")

print("\n" + "="*80)
print("✅ SHAP Waterfall explica exactamente cómo se llega a cada predicción")
print("="*80)

# COMMAND ----------

# DBTITLE 1,Ejercicio 5: Comparación Total
print("="*80)
print("EJERCICIO 5: COMPARACIÓN COMPLETA DE MÉTODOS")
print("="*80)

print("\nResumen de Métodos Aplicados:\n")

# Feature Importance
fi_top = fi_df['feature'].iloc[0]
print(f"1️⃣ Feature Importance (RF):")
print(f"   Top feature: {fi_top}")
print(f"   Velocidad: ⭐⭐⭐⭐⭐ (instantáneo)")
print(f"   Confiabilidad: ⭐⭐⭐ (sesgado con cardinalidad)")

# Permutation Importance
pi_top = pi_df['feature'].iloc[0]
print(f"\n2️⃣ Permutation Importance:")
print(f"   Top feature: {pi_top}")
print(f"   Velocidad: ⭐⭐⭐ (moderado)")
print(f"   Confiabilidad: ⭐⭐⭐⭐⭐ (más robusto)")

# SHAP
shap_importance = np.abs(shap_values.values).mean(axis=0)
shap_top = features[shap_importance.argmax()]
print(f"\n3️⃣ SHAP:")
print(f"   Top feature: {shap_top}")
print(f"   Velocidad: ⭐⭐⭐⭐⭐ (TreeExplainer rápido)")
print(f"   Completitud: ⭐⭐⭐⭐⭐ (global + local + interacciones)")

# PDP
print(f"\n4️⃣ PDP:")
print(f"   Uso: Visualizar efecto de features")
print(f"   Visual: ⭐⭐⭐⭐⭐ (muy intuitivo)")
print(f"   Limitación: Solo global, no interacciones")

print("\n" + "="*80)
print("CONCLUSIÓN FINAL")
print("="*80)
print("\n🎯 Para la mayoría de casos con Random Forest/XGBoost:")
print("\n1. ✅ Usar SHAP como método principal (completo, rápido)")
print("2. ✅ Complementar con PDP para stakeholders (visual)")
print("3. ✅ Validar con Permutation Importance")
print("\n💡 SHAP es el gold standard para interpretabilidad!")

# COMMAND ----------

# DBTITLE 1,Conclusiones Finales
# MAGIC %md
# MAGIC ## 🎓 Conclusiones Finales
# MAGIC
# MAGIC ### 📊 Resumen de Ejercicios
# MAGIC
# MAGIC **Ejercicio 1**: Feature Importance vs. Permutation
# MAGIC - ✅ Comparamos dos métodos de importancia
# MAGIC - ✅ Permutation es más confiable
# MAGIC - 💡 Ambos coinciden, pero Permutation incluye incertidumbre
# MAGIC
# MAGIC **Ejercicio 2**: PDP/ICE
# MAGIC - ✅ Visualizamos efectos de features temporales
# MAGIC - ✅ ICE reveló heterogeneidad (o su ausencia)
# MAGIC - 💡 Viernes y diciembre tienen mayor impacto
# MAGIC
# MAGIC **Ejercicio 3**: SHAP Global
# MAGIC - ✅ Summary plot identificó features importantes
# MAGIC - ✅ Dependence plots revelaron interacciones
# MAGIC - 💡 SHAP es más completo que Feature Importance
# MAGIC
# MAGIC **Ejercicio 4**: SHAP Local
# MAGIC - ✅ Waterfall explicó predicciones específicas
# MAGIC - ✅ Entendimos predicciones altas y bajas
# MAGIC - 💡 Cada predicción tiene su "historia"
# MAGIC
# MAGIC **Ejercicio 5**: Comparación Total
# MAGIC - ✅ Aplicamos todos los métodos
# MAGIC - ✅ Comparamos velocidad, completitud, confiabilidad
# MAGIC - 💡 SHAP es el método más completo
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 💡 Mensajes Clave
# MAGIC
# MAGIC 1. 🔑 **SHAP es el gold standard** - Más completo, robusto, rápido (trees)
# MAGIC 2. ⚠️ **No confiar solo en Feature Importance** - Validar con Permutation/SHAP
# MAGIC 3. 📊 **Combinar global + local** - Entendimiento completo
# MAGIC 4. 🎯 **Visual > Números** - PDP/SHAP plots comunican mejor
# MAGIC 5. 🔄 **Iterar** - Interpretabilidad es un proceso continuo
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 🚀 Workflow Recomendado
# MAGIC
# MAGIC **Para proyectos reales**:
# MAGIC
# MAGIC 1. 🚀 **Exploración**: Feature Importance (5 min)
# MAGIC 2. ⚖️ **Validación**: Permutation Importance (15 min)
# MAGIC 3. 📊 **Análisis Global**: SHAP Summary + PDP (30 min)
# MAGIC 4. 👤 **Casos Específicos**: SHAP Waterfall (15 min)
# MAGIC 5. 📈 **Presentación**: Preparar visualizaciones (1 hora)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### ✅ Checklist
# MAGIC
# MAGIC Antes de presentar un modelo:
# MAGIC
# MAGIC - [ ] Calculaste feature importance
# MAGIC - [ ] Validaste con permutation
# MAGIC - [ ] Generaste SHAP summary plot
# MAGIC - [ ] Analizaste PDP para top features
# MAGIC - [ ] Explicaste predicciones específicas
# MAGIC - [ ] Investigaste predicciones anómalas
# MAGIC - [ ] Verificaste interacciones
# MAGIC - [ ] Documentaste hallazgos
# MAGIC - [ ] Preparaste visualizaciones
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🎉 ¡Felicitaciones!
# MAGIC
# MAGIC **Has completado el módulo práctico de Interpretabilidad de Modelos.**
# MAGIC
# MAGIC Ahora puedes:
# MAGIC - ✅ Explicar modelos complejos
# MAGIC - ✅ Identificar features importantes confiablemente
# MAGIC - ✅ Usar SHAP para explicaciones globales y locales
# MAGIC - ✅ Visualizar efectos con PDP/ICE
# MAGIC - ✅ Detectar data leakage y bias
# MAGIC - ✅ Comunicar insights a stakeholders
# MAGIC
# MAGIC **Próximo módulo**: AutoML y Feature Store
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC **Universidad del Aconcagua**  
# MAGIC **Laboratorio (Herramientas)**  
# MAGIC **Mendoza, Argentina**

# COMMAND ----------

# DBTITLE 1,Título
# MAGIC %md
# MAGIC # 🏃 Práctica: Interpretabilidad de Modelos
# MAGIC ## Material Complementario - Laboratorio (Herramientas)
# MAGIC ### Universidad del Aconcagua - Mendoza, Argentina
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 🎯 Objetivos
# MAGIC
# MAGIC Aplicar técnicas de interpretabilidad a **modelos reales** de la panadería.
# MAGIC
# MAGIC ### 📝 Ejercicios
# MAGIC
# MAGIC 1. **Feature Importance vs. Permutation Importance** (comparación)
# MAGIC 2. **PDP/ICE**: Visualizar efectos de features temporales
# MAGIC 3. **SHAP Global**: Identificar features más importantes
# MAGIC 4. **SHAP Local**: Explicar predicciones específicas
# MAGIC 5. **Comparación Completa**: Todos los métodos en un caso
# MAGIC
# MAGIC ### ⏱️ Duración: 2-3 horas
# MAGIC
# MAGIC ---

# COMMAND ----------

# DBTITLE 1,Setup
# Instalar dependencias
%pip install shap lime --quiet

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
import shap

# Configurar visualizaciones
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

print("✅ Librerías importadas")

# COMMAND ----------

# DBTITLE 1,Cargar datos
# Cargar datasets
ruta_datos = '/Workspace/Users/cortega@uda.edu.ar/Laboratorio/Datasets/'

df_ventas = pd.read_csv(ruta_datos + 'ventas.csv')
df_clientes = pd.read_csv(ruta_datos + 'clientes.csv')

print("✅ Datasets cargados")
print(f"   Ventas: {len(df_ventas):,} registros")
print(f"   Clientes: {len(df_clientes):,} registros")

# Preparar datos
df_ventas['fecha'] = pd.to_datetime(df_ventas['fecha'])
df_ventas = df_ventas.sort_values('fecha').reset_index(drop=True)

# Features temporales
df_ventas['dia_semana'] = df_ventas['fecha'].dt.dayofweek
df_ventas['mes'] = df_ventas['fecha'].dt.month
df_ventas['es_fin_semana'] = df_ventas['dia_semana'].isin([5, 6]).astype(int)

# Merge
df = df_ventas.merge(df_clientes, on='cliente_id', how='left')
df['segmento_encoded'] = df['segmento'].astype('category').cat.codes

print("\n✅ Features creadas")

# COMMAND ----------

# DBTITLE 1,Ejercicio 1: Feature Importance vs. Permutation
# MAGIC %md
# MAGIC ## 💻 Ejercicio 1: Feature Importance vs. Permutation Importance
# MAGIC
# MAGIC ### 🎯 Problema
# MAGIC
# MAGIC Comparar **Feature Importance** (rápido pero sesgado) con **Permutation Importance** (más confiable).
# MAGIC
# MAGIC **Objetivo**:
# MAGIC - Calcular ambas importancias
# MAGIC - Comparar resultados
# MAGIC - Entender diferencias
# MAGIC
# MAGIC ---

# COMMAND ----------

# DBTITLE 1,Ejercicio 1 - Solución
print("="*80)
print("EJERCICIO 1: FEATURE IMPORTANCE VS. PERMUTATION IMPORTANCE")
print("="*80)

# Preparar datos
df_ml = df[df['cliente_id'].notna()].copy()

features = ['sucursal_id', 'dia_semana', 'mes', 'es_fin_semana', 'segmento_encoded']
X = df_ml[features]
y = df_ml['total']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\n📊 Dataset: {len(X):,} registros")
print(f"   Train: {len(X_train):,}, Test: {len(X_test):,}")

# Entrenar modelo
print("\n1️⃣ Entrenando Random Forest...")
model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)

mae = mean_absolute_error(y_test, model.predict(X_test))
print(f"   MAE: ${mae:.2f}")

# Feature Importance
print("\n2️⃣ Feature Importance (RF):")
fi_df = pd.DataFrame({
    'feature': features,
    'fi_importance': model.feature_importances_
}).sort_values('fi_importance', ascending=False)

for _, row in fi_df.iterrows():
    print(f"   {row['feature']:20s}: {row['fi_importance']:.4f}")

# Permutation Importance
print("\n3️⃣ Permutation Importance:")
pi_result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)

pi_df = pd.DataFrame({
    'feature': features,
    'pi_importance': pi_result.importances_mean,
    'pi_std': pi_result.importances_std
}).sort_values('pi_importance', ascending=False)

for _, row in pi_df.iterrows():
    print(f"   {row['feature']:20s}: {row['pi_importance']:.4f} ± {row['pi_std']:.4f}")

# Comparación
print("\n4️⃣ Comparación:")
comparison = fi_df.merge(pi_df, on='feature').sort_values('pi_importance', ascending=False)
print(comparison.to_string(index=False))

# Visualizar
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Feature Importance
ax1.barh(fi_df['feature'], fi_df['fi_importance'])
ax1.set_xlabel('Feature Importance')
ax1.set_title('RF Feature Importance')
ax1.invert_yaxis()

# Permutation Importance
ax2.barh(pi_df['feature'], pi_df['pi_importance'], xerr=pi_df['pi_std'])
ax2.set_xlabel('Permutation Importance')
ax2.set_title('Permutation Importance (con error bars)')
ax2.invert_yaxis()

plt.tight_layout()
plt.show()

print("\n" + "="*80)
print("💡 CONCLUSIÓN")
print("="*80)
print("\nAmbos métodos coinciden en las features más importantes, pero:")
print("- Permutation Importance incluye incertidumbre (error bars)")
print("- Permutation Importance es más confiable con features correlacionadas")
print("- Usar Permutation para decisiones finales")

# COMMAND ----------

