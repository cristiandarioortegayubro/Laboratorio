# Databricks notebook source
# DBTITLE 1,Setup
# Importar librerías
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_selection import (
    VarianceThreshold, SelectKBest, RFE, RFECV, SelectFromModel,
    chi2, f_classif, mutual_info_regression, mutual_info_classif
)
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import Lasso, LassoCV, ElasticNet
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

# Configurar visualizaciones
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

print("✅ Librerías importadas correctamente")

# COMMAND ----------

# DBTITLE 1,Cargar datos
# Cargar datasets de la panadería
ruta_datos = '/Workspace/Users/cortega@uda.edu.ar/Laboratorio/Datasets/'

df_ventas = pd.read_csv(ruta_datos + 'ventas.csv')
df_clientes = pd.read_csv(ruta_datos + 'clientes.csv')

print("✅ Datasets cargados:")
print(f"   Ventas: {len(df_ventas):,} registros")
print(f"   Clientes: {len(df_clientes):,} registros")

# Preparar datos
df_ventas['fecha'] = pd.to_datetime(df_ventas['fecha'])
df_ventas = df_ventas.sort_values('fecha').reset_index(drop=True)

# Features temporales
df_ventas['dia_semana'] = df_ventas['fecha'].dt.dayofweek
df_ventas['mes'] = df_ventas['fecha'].dt.month
df_ventas['es_fin_semana'] = df_ventas['dia_semana'].isin([5, 6]).astype(int)

# Merge con clientes
df = df_ventas.merge(df_clientes, on='cliente_id', how='left')
df['segmento_encoded'] = df['segmento'].astype('category').cat.codes

print("\n✅ Features temporales creadas")
print(f"\nColumnas disponibles: {list(df.columns)[:10]}...")  # Mostrar primeras 10

# COMMAND ----------

# DBTITLE 1,Ejemplo práctico: Comparación de métodos
print("="*80)
print("COMPARACIÓN DE MÉTODOS DE SELECCIÓN DE FEATURES")
print("="*80)

# Preparar datos
df_ml = df[df['cliente_id'].notna()].copy()

features = ['sucursal_id', 'dia_semana', 'mes', 'es_fin_semana', 'segmento_encoded']
X = df_ml[features]
y = df_ml['total']

print(f"\n📊 Dataset: {len(X):,} registros, {len(features)} features")
print(f"\nProbando diferentes métodos de selección...\n")

# Modelo base
model_base = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)

# 1. Baseline: Todas las features
scores_all = cross_val_score(model_base, X, y, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)

# 2. Mutual Information (top 3)
selector_mi = SelectKBest(score_func=mutual_info_regression, k=3)
pipeline_mi = Pipeline([('selector', selector_mi), ('model', model_base)])
scores_mi = cross_val_score(pipeline_mi, X, y, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)

# 3. RF Importance (top 3)
model_for_importance = RandomForestRegressor(n_estimators=100, random_state=42)
model_for_importance.fit(X, y)
selector_rf = SelectFromModel(model_for_importance, threshold=-np.inf, max_features=3)
pipeline_rf = Pipeline([('selector', selector_rf), ('model', model_base)])
scores_rf = cross_val_score(pipeline_rf, X, y, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)

# Resultados
print("Resultados:")
print(f"\n1️⃣ Todas las features ({len(features)}):")
print(f"   MAE: ${-scores_all.mean():.2f} ± ${scores_all.std():.2f}")

print(f"\n2️⃣ Mutual Information (top 3):")
print(f"   MAE: ${-scores_mi.mean():.2f} ± ${scores_mi.std():.2f}")
selector_mi.fit(X, y)
mi_features = X.columns[selector_mi.get_support()].tolist()
print(f"   Features: {mi_features}")

print(f"\n3️⃣ RF Importance (top 3):")
print(f"   MAE: ${-scores_rf.mean():.2f} ± ${scores_rf.std():.2f}")
selector_rf.fit(X, y)
rf_features = X.columns[selector_rf.get_support()].tolist()
print(f"   Features: {rf_features}")

print(f"\n" + "="*80)
print("CONCLUSIÓN")
print("="*80)

mejora_mi = (-scores_all.mean()) - (-scores_mi.mean())
mejora_rf = (-scores_all.mean()) - (-scores_rf.mean())

print(f"\nReducir de {len(features)} a 3 features:")
print(f"  Mutual Info: Mejora de ${mejora_mi:.2f} ({mejora_mi/-scores_all.mean()*100:.1f}%)")
print(f"  RF Importance: Mejora de ${mejora_rf:.2f} ({mejora_rf/-scores_all.mean()*100:.1f}%)")
print(f"\n💡 Menos features puede dar MEJOR rendimiento (evita overfitting)")

# COMMAND ----------

# DBTITLE 1,Conclusiones
# MAGIC %md
# MAGIC ## ✅ Conclusiones
# MAGIC
# MAGIC ### 🎯 Resumen del Módulo
# MAGIC
# MAGIC **Lo que aprendimos**:
# MAGIC
# MAGIC 1. ✅ **Por qué seleccionar features** (maldición de dimensionalidad, overfitting)
# MAGIC 2. ✅ **Métodos de Filtro** (varianza, correlación, MI, Chi², ANOVA F)
# MAGIC 3. ✅ **Métodos Wrapper** (RFE, RFECV, forward/backward selection)
# MAGIC 4. ✅ **Métodos Embedded** (Lasso, RF importance, XGBoost)
# MAGIC 5. ✅ **Permutation Importance** (más confiable que RF importance)
# MAGIC 6. ✅ **Selección con CV** (RFECV, nested CV, evitar leakage)
# MAGIC 7. ✅ **Casos prácticos** con features H3 y temporales
# MAGIC 8. ✅ **Comparación** de métodos y trade-offs
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 💡 Mensajes Clave
# MAGIC
# MAGIC 1. 🔑 **Menos features ≠ peor rendimiento** - Puede mejorar (evita overfitting)
# MAGIC 2. ⚠️ **No hay "mejor" método** - Depende del problema y recursos
# MAGIC 3. 🎯 **Filter → Embedded → Wrapper**: Workflow incremental
# MAGIC 4. ⏱️ **Balance**: Velocidad (Filter) vs. Rendimiento (Wrapper)
# MAGIC 5. 🔄 **Siempre usar Pipeline**: Evita data leakage en CV
# MAGIC 6. 📊 **Validar con múltiples métodos**: No confiar en uno solo
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 📦 Guía Rápida de Selección
# MAGIC **¿Qué método usar?**
# MAGIC
# MAGIC - 🚀 **Exploración rápida** → Mutual Information, Correlación
# MAGIC - ⚖️ **Balance velocidad/rendimiento** → RF Importance, Lasso
# MAGIC - 🎯 **Máximo rendimiento** → RFECV, Permutation Importance
# MAGIC - 📊 **Muchas features (>10K)** → Filter methods
# MAGIC - 🔬 **Pocas features (<1K)** → Wrapper methods
# MAGIC - 🧠 **Interpretabilidad** → Lasso, Correlación
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 📚 Recursos Adicionales
# MAGIC
# MAGIC - [Scikit-learn Feature Selection](https://scikit-learn.org/stable/modules/feature_selection.html)
# MAGIC - [Feature Selection for Machine Learning](https://machinelearningmastery.com/feature-selection-machine-learning-python/)
# MAGIC - [Permutation Importance](https://christophm.github.io/interpretable-ml-book/feature-importance.html)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🎓 ¡Felicitaciones!
# MAGIC
# MAGIC **Has completado el módulo de Selección de Características.**
# MAGIC
# MAGIC Ahora puedes:
# MAGIC - ✅ Identificar cuándo necesitas selección de features
# MAGIC - ✅ Aplicar 10+ métodos diferentes (filtro, wrapper, embedded)
# MAGIC - ✅ Usar RFECV para encontrar N óptimo automáticamente
# MAGIC - ✅ Calcular Permutation Importance confiable
# MAGIC - ✅ Integrar selección con validación cruzada (Pipeline)
# MAGIC - ✅ Seleccionar features H3 y temporales correctamente
# MAGIC - ✅ Comparar métodos y elegir el mejor para tu caso
# MAGIC
# MAGIC **Próximo paso**: Notebook Práctico con ejercicios hands-on.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC **Universidad del Aconcagua**  
# MAGIC **Laboratorio (Herramientas)**  
# MAGIC **Mendoza, Argentina**

# COMMAND ----------

# DBTITLE 1,7. Casos Prácticos con H3
# MAGIC %md
# MAGIC ## 7️⃣ Casos Prácticos: Features H3 y Temporales
# MAGIC
# MAGIC ### 🗺️ Caso 1: Selección de Features H3
# MAGIC
# MAGIC **Problema**: Dataset con **múltiples resoluciones H3** (res 5, 7, 9) - **features redundantes**.
# MAGIC
# MAGIC **Estrategia**:
# MAGIC 1. **Calcular correlación** entre resoluciones
# MAGIC 2. **Seleccionar la resolución óptima** con CV
# MAGIC 3. **Eliminar features H3 redundantes**
# MAGIC
# MAGIC **Ejemplo**:
# MAGIC ```python
# MAGIC import h3
# MAGIC import pandas as pd
# MAGIC from sklearn.feature_selection import mutual_info_regression
# MAGIC from sklearn.model_selection import cross_val_score
# MAGIC from sklearn.ensemble import RandomForestRegressor
# MAGIC
# MAGIC # Dataset con múltiples resoluciones H3
# MAGIC df['h3_res5'] = df.apply(lambda row: h3.geo_to_h3(row['lat'], row['lon'], 5), axis=1)
# MAGIC df['h3_res7'] = df.apply(lambda row: h3.geo_to_h3(row['lat'], row['lon'], 7), axis=1)
# MAGIC df['h3_res9'] = df.apply(lambda row: h3.geo_to_h3(row['lat'], row['lon'], 9), axis=1)
# MAGIC
# MAGIC # Encodear H3 (label encoding)
# MAGIC for col in ['h3_res5', 'h3_res7', 'h3_res9']:
# MAGIC     df[col + '_encoded'] = df[col].astype('category').cat.codes
# MAGIC
# MAGIC # Comparar importancia por resolución
# MAGIC h3_features = ['h3_res5_encoded', 'h3_res7_encoded', 'h3_res9_encoded']
# MAGIC X_h3 = df[h3_features]
# MAGIC y = df['ventas']
# MAGIC
# MAGIC # Mutual information
# MAGIC mi_scores = mutual_info_regression(X_h3, y)
# MAGIC for feature, score in zip(h3_features, mi_scores):
# MAGIC     print(f"{feature}: {score:.4f}")
# MAGIC
# MAGIC # Probar cada resolución con CV
# MAGIC model = RandomForestRegressor(n_estimators=100, random_state=42)
# MAGIC
# MAGIC for feature in h3_features:
# MAGIC     X_single = df[[feature]]
# MAGIC     scores = cross_val_score(model, X_single, y, cv=5, scoring='neg_mean_absolute_error')
# MAGIC     print(f"{feature}: MAE = ${-scores.mean():.2f}")
# MAGIC
# MAGIC # Resultado: Seleccionar la resolución con mejor balance (usualmente res 7)
# MAGIC ```
# MAGIC
# MAGIC **Recomendación**: Para ventas urbanas, **resolución 7** (~5 km²) suele ser óptima.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 📅 Caso 2: Selección de Features Temporales
# MAGIC
# MAGIC **Problema**: Dataset con **muchas features temporales** (día, mes, semana, trimestre, etc.) - **redundantes**.
# MAGIC
# MAGIC **Estrategia**:
# MAGIC 1. **Probar combinaciones** de features temporales
# MAGIC 2. **Eliminar features temporales correlacionadas**
# MAGIC 3. **Mantener solo las más informativas**
# MAGIC
# MAGIC **Ejemplo**:
# MAGIC ```python
# MAGIC import pandas as pd
# MAGIC from sklearn.feature_selection import SelectKBest, mutual_info_regression
# MAGIC
# MAGIC # Features temporales
# MAGIC df['dia_semana'] = df['fecha'].dt.dayofweek
# MAGIC df['mes'] = df['fecha'].dt.month
# MAGIC df['dia_mes'] = df['fecha'].dt.day
# MAGIC df['trimestre'] = df['fecha'].dt.quarter
# MAGIC df['semana_anio'] = df['fecha'].dt.isocalendar().week
# MAGIC df['es_fin_semana'] = df['dia_semana'].isin([5, 6]).astype(int)
# MAGIC
# MAGIC temporal_features = ['dia_semana', 'mes', 'dia_mes', 'trimestre', 'semana_anio', 'es_fin_semana']
# MAGIC
# MAGIC # Seleccionar top temporal features
# MAGIC X_temporal = df[temporal_features]
# MAGIC y = df['ventas']
# MAGIC
# MAGIC selector = SelectKBest(score_func=mutual_info_regression, k=3)
# MAGIC X_selected = selector.fit_transform(X_temporal, y)
# MAGIC
# MAGIC selected_features = [temporal_features[i] for i in range(len(temporal_features)) if selector.get_support()[i]]
# MAGIC print(f"Features temporales seleccionadas: {selected_features}")
# MAGIC
# MAGIC # Típicamente: ['dia_semana', 'mes', 'es_fin_semana'] son las más importantes
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 🔄 Caso 3: Features H3 + Temporales + Otras
# MAGIC
# MAGIC **Dataset completo de panadería**:
# MAGIC ```python
# MAGIC # Todas las features
# MAGIC features = [
# MAGIC     # Temporales
# MAGIC     'dia_semana', 'mes', 'es_fin_semana',
# MAGIC     # Geoespaciales
# MAGIC     'h3_res7_encoded', 'sucursal_id',
# MAGIC     # Cliente
# MAGIC     'segmento_encoded', 'cliente_es_frecuente',
# MAGIC     # Producto
# MAGIC     'categoria_producto', 'precio_promedio'
# MAGIC ]
# MAGIC
# MAGIC X = df[features]
# MAGIC y = df['total']
# MAGIC
# MAGIC # Pipeline completo
# MAGIC from sklearn.pipeline import Pipeline
# MAGIC from sklearn.feature_selection import SelectKBest, mutual_info_regression
# MAGIC from sklearn.ensemble import RandomForestRegressor
# MAGIC
# MAGIC pipeline = Pipeline([
# MAGIC     ('selector', SelectKBest(score_func=mutual_info_regression, k=5)),
# MAGIC     ('model', RandomForestRegressor(n_estimators=100, random_state=42))
# MAGIC ])
# MAGIC
# MAGIC # CV
# MAGIC from sklearn.model_selection import cross_val_score
# MAGIC scores = cross_val_score(pipeline, X, y, cv=5, scoring='neg_mean_absolute_error')
# MAGIC
# MAGIC print(f"MAE: ${-scores.mean():.2f} ± ${scores.std():.2f}")
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 💡 Recomendaciones para Features H3
# MAGIC
# MAGIC 1. ✅ **No uses múltiples resoluciones** a la vez (son redundantes)
# MAGIC 2. ✅ **Resolución 7** (∼5 km²) es buena para la mayoría de casos urbanos
# MAGIC 3. ✅ **Resolución 5** (∼250 km²) para análisis regional
# MAGIC 4. ✅ **Resolución 9** (∼0.1 km²) para análisis hiper-local
# MAGIC 5. ✅ **Combina H3 con otras features geoespaciales** (ciudad, zona, etc.)
# MAGIC
# MAGIC ---

# COMMAND ----------

# DBTITLE 1,8. Comparación de Métodos
# MAGIC %md
# MAGIC ## 8️⃣ Comparación Completa de Métodos
# MAGIC
# MAGIC ### 📊 Tabla Comparativa General
# MAGIC
# MAGIC | Método | Tipo | Velocidad | Rendimiento | Interpretabilidad | Escalabilidad | Cuándo Usar |
# MAGIC |---------|------|-----------|-------------|-------------------|---------------|---------------|
# MAGIC | **Varianza** | Filter | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Features constantes |
# MAGIC | **Correlación** | Filter | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Regresión lineal |
# MAGIC | **Chi²** | Filter | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Clasificación, features categóricas |
# MAGIC | **Mutual Info** | Filter | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | Relaciones no lineales |
# MAGIC | **ANOVA F** | Filter | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Clasificación, features continuas |
# MAGIC | **RFE** | Wrapper | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | Máximo rendimiento, pocos features |
# MAGIC | **RFECV** | Wrapper | ⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐ | Encuentra N óptimo automáticamente |
# MAGIC | **Lasso** | Embedded | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Regresión lineal, interpretabilidad |
# MAGIC | **RF Importance** | Embedded | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | Relaciones no lineales, rápido |
# MAGIC | **Permutation** | Post-hoc | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | Máxima confiabilidad |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### ⚖️ Trade-offs Principales
# MAGIC
# MAGIC **Velocidad vs. Rendimiento**:
# MAGIC ```
# MAGIC Filter (rápido, rendimiento bajo)
# MAGIC   ↓
# MAGIC Embedded (balance)
# MAGIC   ↓
# MAGIC Wrapper (lento, rendimiento alto)
# MAGIC ```
# MAGIC
# MAGIC **Escalabilidad vs. Precisión**:
# MAGIC ```
# MAGIC Filter: Escala a millones de features
# MAGIC Embedded: Escala a decenas de miles
# MAGIC Wrapper: Escala a cientos/miles
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 🧩 Árbol de Decisión: ¿Qué Método Usar?
# MAGIC
# MAGIC ```
# MAGIC ¿Cuántas features tienes?
# MAGIC │
# MAGIC ├── > 10,000 → Filter Methods (Mutual Info, Chi²)
# MAGIC │
# MAGIC ├── 1,000 - 10,000
# MAGIC │   └── ¿Tiempo es crítico?
# MAGIC │       ├── Sí → Embedded (RF Importance, Lasso)
# MAGIC │       └── No → RFECV
# MAGIC │
# MAGIC └── < 1,000
# MAGIC     └── ¿Quieres máximo rendimiento?
# MAGIC         ├── Sí → RFECV (wrapper)
# MAGIC         └── No → Embedded (RF Importance, Lasso)
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 📊 Experimento Comparativo
# MAGIC
# MAGIC **Dataset**: Ventas de panadería (50,000 registros, 20 features)
# MAGIC
# MAGIC | Método | Features Seleccionadas | Tiempo | MAE | Comentario |
# MAGIC |---------|------------------------|--------|-----|------------|
# MAGIC | **Todas las features** | 20 | - | $15.20 | Baseline |
# MAGIC | **Mutual Info (top 10)** | 10 | 2s | $14.80 | Rápido, bueno |
# MAGIC | **Lasso** | 8 | 5s | $14.50 | Interpretable |
# MAGIC | **RF Importance (top 10)** | 10 | 10s | $14.30 | Buen balance |
# MAGIC | **RFECV** | 12 | 60s | **$14.10** | Mejor rendimiento |
# MAGIC | **Permutation (top 10)** | 10 | 45s | $14.15 | Más confiable |
# MAGIC
# MAGIC 🎯 **Recomendación**: Usar **RF Importance** para exploración rápida, **RFECV** para modelo final.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 💡 Workflow Recomendado
# MAGIC
# MAGIC **Fase 1: Exploración Rápida** (⏱️ Minutos)
# MAGIC 1. Eliminar features con **baja varianza**
# MAGIC 2. Calcular **correlaciones** (eliminar redundantes)
# MAGIC 3. Calcular **Mutual Information** (top 20-30)
# MAGIC
# MAGIC **Fase 2: Selección Iterativa** (⏱️ Horas)
# MAGIC 4. Entrenar **Random Forest** con features de Fase 1
# MAGIC 5. Usar **RF Importance** o **Lasso** (top 10-15)
# MAGIC 6. Evaluar con **validación cruzada**
# MAGIC
# MAGIC **Fase 3: Optimización Final** (⏱️ Días)
# MAGIC 7. **RFECV** para encontrar N óptimo
# MAGIC 8. **Permutation Importance** para validar
# MAGIC 9. **Nested CV** para evaluación final
# MAGIC
# MAGIC ---

# COMMAND ----------

# DBTITLE 1,9. Mejores Prácticas
# MAGIC %md
# MAGIC ## 9️⃣ Mejores Prácticas de Selección de Features
# MAGIC
# MAGIC ### ✅ DO: Buenas Prácticas
# MAGIC
# MAGIC #### 1. **Siempre Escalar Datos Antes de Selección**
# MAGIC ```python
# MAGIC # ✅ BIEN: Escalar antes de Lasso o VarianceThreshold
# MAGIC from sklearn.preprocessing import StandardScaler
# MAGIC
# MAGIC scaler = StandardScaler()
# MAGIC X_scaled = scaler.fit_transform(X)
# MAGIC
# MAGIC model = Lasso(alpha=1.0)
# MAGIC model.fit(X_scaled, y)
# MAGIC
# MAGIC # ❌ MAL: Sin escalar
# MAGIC model.fit(X, y)  # Features con escalas diferentes son penalizadas injustamente
# MAGIC ```
# MAGIC
# MAGIC #### 2. **Usar Pipeline para Evitar Data Leakage**
# MAGIC ```python
# MAGIC # ✅ BIEN: Selección dentro del CV
# MAGIC from sklearn.pipeline import Pipeline
# MAGIC from sklearn.feature_selection import SelectKBest, mutual_info_regression
# MAGIC
# MAGIC pipeline = Pipeline([
# MAGIC     ('scaler', StandardScaler()),
# MAGIC     ('selector', SelectKBest(score_func=mutual_info_regression, k=10)),
# MAGIC     ('model', RandomForestRegressor())
# MAGIC ])
# MAGIC
# MAGIC scores = cross_val_score(pipeline, X, y, cv=5)
# MAGIC
# MAGIC # ❌ MAL: Selección antes del CV
# MAGIC selector = SelectKBest(k=10)
# MAGIC X_selected = selector.fit_transform(X, y)  # Data leakage!
# MAGIC scores = cross_val_score(model, X_selected, y, cv=5)
# MAGIC ```
# MAGIC
# MAGIC #### 3. **Probar Múltiples Métodos**
# MAGIC ```python
# MAGIC # ✅ BIEN: Comparar múltiples métodos
# MAGIC methods = {
# MAGIC     'mutual_info': SelectKBest(score_func=mutual_info_regression, k=10),
# MAGIC     'lasso': SelectFromModel(Lasso(alpha=1.0)),
# MAGIC     'rf_importance': SelectFromModel(RandomForestRegressor(n_estimators=100))
# MAGIC }
# MAGIC
# MAGIC for name, selector in methods.items():
# MAGIC     pipeline = Pipeline([('selector', selector), ('model', RandomForestRegressor())])
# MAGIC     scores = cross_val_score(pipeline, X, y, cv=5, scoring='r2')
# MAGIC     print(f"{name}: R² = {scores.mean():.4f}")
# MAGIC ```
# MAGIC
# MAGIC #### 4. **Documentar Features Seleccionadas**
# MAGIC ```python
# MAGIC # ✅ BIEN: Guardar y documentar
# MAGIC selected_features = X.columns[selector.get_support()].tolist()
# MAGIC
# MAGIC import json
# MAGIC with open('selected_features.json', 'w') as f:
# MAGIC     json.dump({
# MAGIC         'features': selected_features,
# MAGIC         'method': 'RFECV',
# MAGIC         'score': best_score,
# MAGIC         'date': '2026-07-27'
# MAGIC     }, f, indent=2)
# MAGIC ```
# MAGIC
# MAGIC #### 5. **Validar con Nested CV**
# MAGIC ```python
# MAGIC # ✅ BIEN: Nested CV para evaluación realista
# MAGIC # (ver sección 6 para implementación completa)
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### ❌ DON'T: Errores Comunes
# MAGIC
# MAGIC #### 1. **NO Seleccionar en Todo el Dataset Antes de CV**
# MAGIC ```python
# MAGIC # ❌ MAL
# MAGIC X_selected = selector.fit_transform(X, y)  # Usa info de test
# MAGIC scores = cross_val_score(model, X_selected, y, cv=5)  # Leakage!
# MAGIC ```
# MAGIC
# MAGIC #### 2. **NO Usar Solo RF Importance con Features Correlacionadas**
# MAGIC ```python
# MAGIC # ❌ MAL: RF importance sesgado con features correlacionadas
# MAGIC importances = model.feature_importances_
# MAGIC
# MAGIC # ✅ BIEN: Usar Permutation Importance
# MAGIC from sklearn.inspection import permutation_importance
# MAGIC result = permutation_importance(model, X_test, y_test, n_repeats=10)
# MAGIC ```
# MAGIC
# MAGIC #### 3. **NO Eliminar Features Sin Analizar**
# MAGIC ```python
# MAGIC # ❌ MAL: Eliminar sin revisar
# MAGIC X_reduced = X.drop(['feature_1', 'feature_2'], axis=1)
# MAGIC
# MAGIC # ✅ BIEN: Analizar correlaciones y importancias primero
# MAGIC correlations = df.corr()['target'].abs().sort_values(ascending=False)
# MAGIC print(correlations[['feature_1', 'feature_2']])
# MAGIC ```
# MAGIC
# MAGIC #### 4. **NO Usar Lasso Sin Escalar**
# MAGIC ```python
# MAGIC # ❌ MAL
# MAGIC model = Lasso(alpha=1.0)
# MAGIC model.fit(X, y)  # Features con mayor escala son sobre-penalizadas
# MAGIC
# MAGIC # ✅ BIEN
# MAGIC X_scaled = StandardScaler().fit_transform(X)
# MAGIC model.fit(X_scaled, y)
# MAGIC ```
# MAGIC
# MAGIC #### 5. **NO Confiar en un Solo Método**
# MAGIC ```python
# MAGIC # ❌ MAL: Solo usar un método
# MAGIC selector = SelectKBest(k=10)
# MAGIC
# MAGIC # ✅ BIEN: Validar con múltiples métodos
# MAGIC mi_features = SelectKBest(mutual_info_regression, k=10).fit(X, y).get_support()
# MAGIC rf_features = SelectFromModel(RandomForestRegressor()).fit(X, y).get_support()
# MAGIC
# MAGIC # Features seleccionadas por ambos métodos
# MAGIC common_features = X.columns[mi_features & rf_features].tolist()
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 📊 Cuántas Features Seleccionar?
# MAGIC
# MAGIC **Reglas prácticas**:
# MAGIC
# MAGIC | Tamaño Dataset | Features Iniciales | Features Objetivo |
# MAGIC |-----------------|--------------------| ------------------|
# MAGIC | < 1,000 | 100 | 5-10 |
# MAGIC | 1,000 - 10,000 | 100-1,000 | 10-30 |
# MAGIC | 10,000 - 100,000 | 1,000-10,000 | 30-100 |
# MAGIC | > 100,000 | > 10,000 | 100-500 |
# MAGIC
# MAGIC ⚠️ **Mejor enfoque**: Usar **RFECV** o **validación cruzada** para encontrar el número óptimo.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 💡 Tips Finales
# MAGIC
# MAGIC 1. ✅ **Empieza con filtros** (rápido) para reducción inicial
# MAGIC 2. ✅ **Usa embedded** (RF, Lasso) para selección intermedia
# MAGIC 3. ✅ **Finaliza con wrapper** (RFECV) si tiempo lo permite
# MAGIC 4. ✅ **Valida con Permutation Importance** antes de producción
# MAGIC 5. ✅ **Menos es más**: Prefiere **modelos simples e interpretables**
# MAGIC 6. ✅ **Documenta tu proceso**: Qué features, por qué, cuándo
# MAGIC 7. ✅ **Monitorea en producción**: Features pueden perder importancia con el tiempo
# MAGIC
# MAGIC ---

# COMMAND ----------

# DBTITLE 1,5. Permutation Importance
# MAGIC %md
# MAGIC ## 5️⃣ Permutation Importance
# MAGIC
# MAGIC ### 📖 Concepto
# MAGIC
# MAGIC **Permutation Importance** mide **cuánto empeora** el rendimiento del modelo cuando se **permutan (mezclan) los valores** de una feature.
# MAGIC
# MAGIC **Idea**:
# MAGIC 1. Entrenar modelo
# MAGIC 2. Calcular rendimiento baseline
# MAGIC 3. Para cada feature:
# MAGIC    - **Permutar** sus valores aleatoriamente
# MAGIC    - Calcular nuevo rendimiento
# MAGIC    - Importancia = baseline - nuevo rendimiento
# MAGIC 4. Feature con mayor caída = más importante
# MAGIC
# MAGIC **Ventaja**:
# MAGIC - ✅ **Model-agnostic**: Funciona con **cualquier modelo**
# MAGIC - ✅ **Más confiable** que feature importance de árboles
# MAGIC - ✅ **Considera interacciones** entre features
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 💻 Implementación
# MAGIC
# MAGIC ```python
# MAGIC from sklearn.inspection import permutation_importance
# MAGIC from sklearn.ensemble import RandomForestRegressor
# MAGIC from sklearn.model_selection import train_test_split
# MAGIC
# MAGIC # Split datos
# MAGIC X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# MAGIC
# MAGIC # Entrenar modelo
# MAGIC model = RandomForestRegressor(n_estimators=100, random_state=42)
# MAGIC model.fit(X_train, y_train)
# MAGIC
# MAGIC # Calcular permutation importance
# MAGIC result = permutation_importance(
# MAGIC     model, X_test, y_test, 
# MAGIC     n_repeats=10,  # Número de permutaciones
# MAGIC     random_state=42,
# MAGIC     n_jobs=-1
# MAGIC )
# MAGIC
# MAGIC # Resultados
# MAGIC importances = pd.DataFrame({
# MAGIC     'feature': X.columns,
# MAGIC     'importance': result.importances_mean,
# MAGIC     'std': result.importances_std
# MAGIC }).sort_values('importance', ascending=False)
# MAGIC
# MAGIC print(importances)
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 📊 Visualización
# MAGIC
# MAGIC ```python
# MAGIC import matplotlib.pyplot as plt
# MAGIC
# MAGIC # Barplot con error bars
# MAGIC plt.figure(figsize=(10, 6))
# MAGIC plt.barh(
# MAGIC     importances['feature'][:15], 
# MAGIC     importances['importance'][:15],
# MAGIC     xerr=importances['std'][:15]  # Error bars
# MAGIC )
# MAGIC plt.xlabel('Permutation Importance')
# MAGIC plt.title('Top 15 Features por Permutation Importance')
# MAGIC plt.gca().invert_yaxis()
# MAGIC plt.show()
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### ⚖️ RF Feature Importance vs. Permutation Importance
# MAGIC
# MAGIC **Diferencias**:
# MAGIC
# MAGIC | Aspecto | RF Importance | Permutation Importance |
# MAGIC |---------|---------------|-------------------------|
# MAGIC | **Cálculo** | Durante entrenamiento | Después de entrenar |
# MAGIC | **Velocidad** | ⭐⭐⭐⭐⭐ Rápido | ⭐⭐⭐ Más lento |
# MAGIC | **Confiabilidad** | Sesgado con features correlacionadas | ✅ Más confiable |
# MAGIC | **Aplicabilidad** | Solo modelos tree-based | ✅ Cualquier modelo |
# MAGIC
# MAGIC **Problema de RF Importance**: Sobreestima features con **alta cardinalidad** (muchos valores únicos).
# MAGIC
# MAGIC **Ejemplo**:
# MAGIC ```python
# MAGIC # Feature importance (sesgado)
# MAGIC rf_importance = model.feature_importances_
# MAGIC
# MAGIC # Permutation importance (más confiable)
# MAGIC perm_importance = result.importances_mean
# MAGIC
# MAGIC # Comparar
# MAGIC comparison = pd.DataFrame({
# MAGIC     'feature': X.columns,
# MAGIC     'rf_importance': rf_importance,
# MAGIC     'perm_importance': perm_importance
# MAGIC }).sort_values('perm_importance', ascending=False)
# MAGIC
# MAGIC print(comparison)
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 🎯 Cuándo Usar Permutation Importance
# MAGIC
# MAGIC ✅ **Usar cuando**:
# MAGIC - Quieres **importancia más confiable**
# MAGIC - Tienes features **correlacionadas**
# MAGIC - Modelo es **black-box** (KNN, SVM, redes neuronales)
# MAGIC - Tiempo no es crítico (más lento que RF importance)
# MAGIC
# MAGIC ❌ **No usar cuando**:
# MAGIC - Dataset es **muy grande** (lento)
# MAGIC - Necesitas **velocidad**
# MAGIC
# MAGIC ---

# COMMAND ----------

# DBTITLE 1,6. Selección con CV
# MAGIC %md
# MAGIC ## 6️⃣ Selección de Features con Validación Cruzada
# MAGIC
# MAGIC ### 📖 Concepto
# MAGIC
# MAGIC **Problema**: ¿Cómo saber cuántas features seleccionar?
# MAGIC
# MAGIC **Solución**: Usar **validación cruzada** para encontrar el número óptimo.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 📊 Método 1: RFECV (RFE + CV)
# MAGIC
# MAGIC **Ya visto antes**, pero repasemos:
# MAGIC
# MAGIC ```python
# MAGIC from sklearn.feature_selection import RFECV
# MAGIC from sklearn.ensemble import RandomForestRegressor
# MAGIC from sklearn.model_selection import KFold
# MAGIC
# MAGIC # RFECV con K-Fold CV
# MAGIC model = RandomForestRegressor(n_estimators=100, random_state=42)
# MAGIC cv = KFold(n_splits=5, shuffle=True, random_state=42)
# MAGIC
# MAGIC selector = RFECV(
# MAGIC     estimator=model,
# MAGIC     step=1,
# MAGIC     cv=cv,
# MAGIC     scoring='neg_mean_absolute_error',
# MAGIC     n_jobs=-1
# MAGIC )
# MAGIC
# MAGIC selector.fit(X, y)
# MAGIC
# MAGIC print(f"Número óptimo de features: {selector.n_features_}")
# MAGIC print(f"Features: {X.columns[selector.support_].tolist()}")
# MAGIC
# MAGIC # Visualizar curva
# MAGIC import matplotlib.pyplot as plt
# MAGIC
# MAGIC plt.figure(figsize=(10, 6))
# MAGIC plt.plot(range(1, len(selector.cv_results_['mean_test_score']) + 1),
# MAGIC          -selector.cv_results_['mean_test_score'])  # Negativo porque es neg_mae
# MAGIC plt.xlabel('Número de Features')
# MAGIC plt.ylabel('MAE (CV)')
# MAGIC plt.title('RFECV: MAE vs. Número de Features')
# MAGIC plt.axvline(x=selector.n_features_, color='red', linestyle='--', label=f'Optimal = {selector.n_features_}')
# MAGIC plt.legend()
# MAGIC plt.grid(True)
# MAGIC plt.show()
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 📊 Método 2: SelectFromModel + CV
# MAGIC
# MAGIC **Idea**: Probar diferentes thresholds con CV.
# MAGIC
# MAGIC ```python
# MAGIC from sklearn.feature_selection import SelectFromModel
# MAGIC from sklearn.model_selection import cross_val_score
# MAGIC from sklearn.ensemble import RandomForestRegressor
# MAGIC import numpy as np
# MAGIC
# MAGIC # Entrenar modelo para obtener importances
# MAGIC model_full = RandomForestRegressor(n_estimators=100, random_state=42)
# MAGIC model_full.fit(X, y)
# MAGIC
# MAGIC # Probar diferentes thresholds
# MAGIC thresholds = np.linspace(0, model_full.feature_importances_.max(), 20)
# MAGIC results = []
# MAGIC
# MAGIC for threshold in thresholds:
# MAGIC     # Seleccionar features
# MAGIC     selector = SelectFromModel(model_full, threshold=threshold, prefit=True)
# MAGIC     X_selected = selector.transform(X)
# MAGIC     
# MAGIC     # Skip si no hay features
# MAGIC     if X_selected.shape[1] == 0:
# MAGIC         continue
# MAGIC     
# MAGIC     # Evaluar con CV
# MAGIC     model = RandomForestRegressor(n_estimators=50, random_state=42)
# MAGIC     scores = cross_val_score(model, X_selected, y, cv=5, scoring='neg_mean_absolute_error')
# MAGIC     
# MAGIC     results.append({
# MAGIC         'threshold': threshold,
# MAGIC         'n_features': X_selected.shape[1],
# MAGIC         'mae': -scores.mean()
# MAGIC     })
# MAGIC
# MAGIC results_df = pd.DataFrame(results)
# MAGIC
# MAGIC # Encontrar óptimo
# MAGIC best_idx = results_df['mae'].idxmin()
# MAGIC best_result = results_df.loc[best_idx]
# MAGIC
# MAGIC print(f"Threshold óptimo: {best_result['threshold']:.4f}")
# MAGIC print(f"Número óptimo de features: {int(best_result['n_features'])}")
# MAGIC print(f"MAE: ${best_result['mae']:.2f}")
# MAGIC
# MAGIC # Visualizar
# MAGIC plt.figure(figsize=(10, 6))
# MAGIC plt.plot(results_df['n_features'], results_df['mae'], marker='o')
# MAGIC plt.xlabel('Número de Features')
# MAGIC plt.ylabel('MAE (CV)')
# MAGIC plt.title('MAE vs. Número de Features')
# MAGIC plt.axvline(x=best_result['n_features'], color='red', linestyle='--', label='Optimal')
# MAGIC plt.legend()
# MAGIC plt.grid(True)
# MAGIC plt.show()
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 📊 Método 3: Nested CV para Selección + Evaluación
# MAGIC
# MAGIC **Problema**: Usar el mismo CV para selección y evaluación → **overfitting**.
# MAGIC
# MAGIC **Solución**: **Nested CV** (CV anidado).
# MAGIC
# MAGIC **Estructura**:
# MAGIC - **Outer CV**: Evaluación del rendimiento
# MAGIC - **Inner CV**: Selección de features
# MAGIC
# MAGIC ```python
# MAGIC from sklearn.model_selection import cross_val_score, KFold
# MAGIC from sklearn.feature_selection import RFECV
# MAGIC from sklearn.ensemble import RandomForestRegressor
# MAGIC
# MAGIC # Outer CV (evaluación)
# MAGIC outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
# MAGIC
# MAGIC # Inner CV (selección)
# MAGIC inner_cv = KFold(n_splits=3, shuffle=True, random_state=42)
# MAGIC
# MAGIC scores = []
# MAGIC
# MAGIC for train_idx, test_idx in outer_cv.split(X):
# MAGIC     X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
# MAGIC     y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
# MAGIC     
# MAGIC     # Selección en train (inner CV)
# MAGIC     model = RandomForestRegressor(n_estimators=50, random_state=42)
# MAGIC     selector = RFECV(estimator=model, cv=inner_cv, scoring='r2', n_jobs=-1)
# MAGIC     selector.fit(X_train, y_train)
# MAGIC     
# MAGIC     X_train_selected = selector.transform(X_train)
# MAGIC     X_test_selected = selector.transform(X_test)
# MAGIC     
# MAGIC     # Entrenar modelo final con features seleccionadas
# MAGIC     model_final = RandomForestRegressor(n_estimators=100, random_state=42)
# MAGIC     model_final.fit(X_train_selected, y_train)
# MAGIC     
# MAGIC     # Evaluar en test
# MAGIC     score = model_final.score(X_test_selected, y_test)
# MAGIC     scores.append(score)
# MAGIC     
# MAGIC     print(f"Fold: R² = {score:.4f}, Features = {selector.n_features_}")
# MAGIC
# MAGIC print(f"\nR² promedio: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
# MAGIC ```
# MAGIC
# MAGIC 💡 **Nested CV** da una estimación **más realista** del rendimiento.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### ⚠️ Errores Comunes
# MAGIC
# MAGIC ❌ **Error 1**: Seleccionar features en todo el dataset antes de CV
# MAGIC ```python
# MAGIC # MAL
# MAGIC selector = SelectKBest(k=10)
# MAGIC X_selected = selector.fit_transform(X, y)  # Usa TODO el dataset
# MAGIC scores = cross_val_score(model, X_selected, y, cv=5)  # Data leakage!
# MAGIC ```
# MAGIC
# MAGIC ✅ **Correcto**: Seleccionar dentro del CV
# MAGIC ```python
# MAGIC # BIEN
# MAGIC from sklearn.pipeline import Pipeline
# MAGIC
# MAGIC pipeline = Pipeline([
# MAGIC     ('selector', SelectKBest(k=10)),
# MAGIC     ('model', RandomForestRegressor())
# MAGIC ])
# MAGIC
# MAGIC scores = cross_val_score(pipeline, X, y, cv=5)  # Selección en cada fold
# MAGIC ```
# MAGIC
# MAGIC ---

# COMMAND ----------

# DBTITLE 1,3. Métodos Wrapper
# MAGIC %md
# MAGIC ## 3️⃣ Métodos Wrapper (Wrapper Methods)
# MAGIC
# MAGIC ### 📖 Concepto
# MAGIC
# MAGIC **Métodos wrapper** evalúan **subconjuntos de features** entrenando un modelo y midiendo su rendimiento.
# MAGIC
# MAGIC **Proceso**:
# MAGIC 1. Entrenar modelo con subconjunto de features
# MAGIC 2. Medir rendimiento (accuracy, MAE, etc.)
# MAGIC 3. Probar otro subconjunto
# MAGIC 4. Repetir hasta encontrar el mejor
# MAGIC
# MAGIC **Ventajas**:
# MAGIC - ✅ **Considera interacciones**: Entre features
# MAGIC - ✅ **Optimiza para el modelo específico**
# MAGIC - ✅ **Mejor rendimiento**: Que métodos de filtro
# MAGIC
# MAGIC **Desventajas**:
# MAGIC - ❌ **Muy lento**: Entrenan muchos modelos
# MAGIC - ❌ **Riesgo de overfitting**: Sobre el conjunto de validación
# MAGIC - ❌ **No escalable**: Impracticable con miles de features
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 🔼 Método 1: Forward Selection (Selección Hacia Adelante)
# MAGIC
# MAGIC **Proceso**:
# MAGIC 1. Empezar sin features
# MAGIC 2. Probar **agregar cada feature** restante
# MAGIC 3. Seleccionar la que **mejora más** el rendimiento
# MAGIC 4. Repetir hasta que no haya mejora
# MAGIC
# MAGIC **Visualización**:
# MAGIC ```
# MAGIC Iteración 1: [] → [F1] (mejor)
# MAGIC Iteración 2: [F1] → [F1, F3] (mejor)
# MAGIC Iteración 3: [F1, F3] → [F1, F3, F7] (mejor)
# MAGIC Iteración 4: [F1, F3, F7] → No mejora → PARAR
# MAGIC ```
# MAGIC
# MAGIC **Implementación (manual)**:
# MAGIC ```python
# MAGIC from sklearn.model_selection import cross_val_score
# MAGIC from sklearn.ensemble import RandomForestRegressor
# MAGIC
# MAGIC def forward_selection(X, y, max_features=10):
# MAGIC     selected_features = []
# MAGIC     remaining_features = list(X.columns)
# MAGIC     
# MAGIC     model = RandomForestRegressor(n_estimators=50, random_state=42)
# MAGIC     best_score = -float('inf')
# MAGIC     
# MAGIC     for i in range(max_features):
# MAGIC         scores = []
# MAGIC         
# MAGIC         # Probar cada feature restante
# MAGIC         for feature in remaining_features:
# MAGIC             features_to_test = selected_features + [feature]
# MAGIC             X_subset = X[features_to_test]
# MAGIC             
# MAGIC             # Evaluar con CV
# MAGIC             score = cross_val_score(model, X_subset, y, cv=5, scoring='r2').mean()
# MAGIC             scores.append((feature, score))
# MAGIC         
# MAGIC         # Seleccionar mejor feature
# MAGIC         best_feature, current_best_score = max(scores, key=lambda x: x[1])
# MAGIC         
# MAGIC         # Si no mejora, parar
# MAGIC         if current_best_score <= best_score:
# MAGIC             break
# MAGIC         
# MAGIC         # Agregar feature
# MAGIC         selected_features.append(best_feature)
# MAGIC         remaining_features.remove(best_feature)
# MAGIC         best_score = current_best_score
# MAGIC         
# MAGIC         print(f"Iteración {i+1}: Agregada {best_feature}, R² = {best_score:.4f}")
# MAGIC     
# MAGIC     return selected_features
# MAGIC
# MAGIC # Usar
# MAGIC selected = forward_selection(X, y, max_features=5)
# MAGIC print(f"\nFeatures seleccionadas: {selected}")
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 🔽 Método 2: Backward Elimination (Eliminación Hacia Atrás)
# MAGIC
# MAGIC **Proceso**:
# MAGIC 1. Empezar con **todas las features**
# MAGIC 2. Probar **eliminar cada feature**
# MAGIC 3. Eliminar la que **afecta menos** al rendimiento
# MAGIC 4. Repetir hasta que eliminar empeore mucho
# MAGIC
# MAGIC **Visualización**:
# MAGIC ```
# MAGIC Iteración 1: [F1, F2, F3, F4, F5] → [F1, F2, F3, F5] (eliminar F4)
# MAGIC Iteración 2: [F1, F2, F3, F5] → [F1, F3, F5] (eliminar F2)
# MAGIC Iteración 3: [F1, F3, F5] → Eliminar empeora mucho → PARAR
# MAGIC ```
# MAGIC
# MAGIC ⚠️ **Problema**: Más lento que forward con muchas features.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 🔄 Método 3: Recursive Feature Elimination (RFE)
# MAGIC
# MAGIC **Proceso**:
# MAGIC 1. Entrenar modelo con todas las features
# MAGIC 2. Calcular **importancia** de cada feature
# MAGIC 3. **Eliminar la menos importante**
# MAGIC 4. Repetir hasta tener N features deseadas
# MAGIC
# MAGIC **Ventaja**: Más eficiente que backward elimination.
# MAGIC
# MAGIC **Implementación**:
# MAGIC ```python
# MAGIC from sklearn.feature_selection import RFE
# MAGIC from sklearn.ensemble import RandomForestRegressor
# MAGIC
# MAGIC # RFE para seleccionar top 10 features
# MAGIC model = RandomForestRegressor(n_estimators=100, random_state=42)
# MAGIC selector = RFE(estimator=model, n_features_to_select=10, step=1)
# MAGIC
# MAGIC X_selected = selector.fit_transform(X, y)
# MAGIC
# MAGIC # Ver features seleccionadas
# MAGIC selected_features = X.columns[selector.support_].tolist()
# MAGIC print(f"Features seleccionadas: {selected_features}")
# MAGIC
# MAGIC # Ver ranking de features
# MAGIC ranking = pd.DataFrame({
# MAGIC     'feature': X.columns,
# MAGIC     'ranking': selector.ranking_
# MAGIC }).sort_values('ranking')
# MAGIC
# MAGIC print("\nRanking de features:")
# MAGIC print(ranking)
# MAGIC ```
# MAGIC
# MAGIC **RFE con Validación Cruzada (RFECV)**:
# MAGIC ```python
# MAGIC from sklearn.feature_selection import RFECV
# MAGIC
# MAGIC # RFECV encuentra automáticamente el número óptimo
# MAGIC selector = RFECV(
# MAGIC     estimator=model, 
# MAGIC     step=1, 
# MAGIC     cv=5,  # 5-fold CV
# MAGIC     scoring='r2',
# MAGIC     n_jobs=-1
# MAGIC )
# MAGIC
# MAGIC selector.fit(X, y)
# MAGIC
# MAGIC print(f"Número óptimo de features: {selector.n_features_}")
# MAGIC print(f"Features seleccionadas: {X.columns[selector.support_].tolist()}")
# MAGIC
# MAGIC # Visualizar
# MAGIC import matplotlib.pyplot as plt
# MAGIC
# MAGIC plt.figure(figsize=(10, 6))
# MAGIC plt.plot(range(1, len(selector.cv_results_['mean_test_score']) + 1), 
# MAGIC          selector.cv_results_['mean_test_score'])
# MAGIC plt.xlabel('Número de Features')
# MAGIC plt.ylabel('R² Score (CV)')
# MAGIC plt.title('RFECV: Rendimiento vs. Número de Features')
# MAGIC plt.grid(True)
# MAGIC plt.show()
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### ⚖️ Comparación de Métodos Wrapper
# MAGIC
# MAGIC | Método | Complejidad | Resultado | Cuándo Usar |
# MAGIC |---------|-------------|-----------|---------------|
# MAGIC | **Forward Selection** | O(N²) | Bueno | Pocas features esperadas |
# MAGIC | **Backward Elimination** | O(N²) | Bueno | Muchas features importantes |
# MAGIC | **RFE** | O(N) | Muy bueno | **Recomendado** (más eficiente) |
# MAGIC | **RFECV** | O(N × K) | Excelente | Mejor opción (encuentra N óptimo) |
# MAGIC
# MAGIC N = número de features, K = número de folds en CV
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 🎯 Cuándo Usar Métodos Wrapper
# MAGIC
# MAGIC ✅ **Usar cuando**:
# MAGIC - Dataset **mediano** (< 1000 features)
# MAGIC - Tiempo de entrenamiento **no es crítico**
# MAGIC - Quieres **máximo rendimiento**
# MAGIC - Tienes recursos computacionales
# MAGIC
# MAGIC ❌ **No usar cuando**:
# MAGIC - Dataset **muy grande** (> 10,000 features)
# MAGIC - Tiempo es **crítico**
# MAGIC - Recursos computacionales **limitados**
# MAGIC
# MAGIC ---

# COMMAND ----------

# DBTITLE 1,4. Métodos Embedded
# MAGIC %md
# MAGIC ## 4️⃣ Métodos Embedded (Embedded Methods)
# MAGIC
# MAGIC ### 📖 Concepto
# MAGIC
# MAGIC **Métodos embedded** realizan selección de features **durante el entrenamiento del modelo**.
# MAGIC
# MAGIC **Ventaja clave**: Balance entre **velocidad** (más rápido que wrapper) y **rendimiento** (mejor que filtro).
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 🎯 Método 1: Regularización Lasso (L1)
# MAGIC
# MAGIC **Idea**: Lasso **penaliza coeficientes** y los **fuerza a cero**.
# MAGIC
# MAGIC **Fórmula**:
# MAGIC ```
# MAGIC Loss = MSE + α × ∑|wᵢ|
# MAGIC ```
# MAGIC
# MAGIC - α alto → más coeficientes = 0 → menos features
# MAGIC - α bajo → menos penalización → más features
# MAGIC
# MAGIC **Implementación**:
# MAGIC ```python
# MAGIC from sklearn.linear_model import Lasso
# MAGIC from sklearn.preprocessing import StandardScaler
# MAGIC
# MAGIC # Escalar datos (importante para Lasso)
# MAGIC scaler = StandardScaler()
# MAGIC X_scaled = scaler.fit_transform(X)
# MAGIC
# MAGIC # Lasso con penalización fuerte
# MAGIC model = Lasso(alpha=1.0, random_state=42)
# MAGIC model.fit(X_scaled, y)
# MAGIC
# MAGIC # Features con coeficiente != 0
# MAGIC selected_features = X.columns[model.coef_ != 0].tolist()
# MAGIC print(f"Features seleccionadas: {selected_features}")
# MAGIC print(f"Número: {len(selected_features)}")
# MAGIC
# MAGIC # Visualizar coeficientes
# MAGIC import matplotlib.pyplot as plt
# MAGIC
# MAGIC plt.figure(figsize=(10, 6))
# MAGIC plt.barh(X.columns, model.coef_)
# MAGIC plt.xlabel('Coeficiente')
# MAGIC plt.title('Coeficientes Lasso')
# MAGIC plt.axvline(x=0, color='red', linestyle='--')
# MAGIC plt.show()
# MAGIC ```
# MAGIC
# MAGIC **LassoCV** (encuentra α óptimo con CV):
# MAGIC ```python
# MAGIC from sklearn.linear_model import LassoCV
# MAGIC
# MAGIC # LassoCV prueba múltiples alphas
# MAGIC model = LassoCV(cv=5, random_state=42, n_jobs=-1)
# MAGIC model.fit(X_scaled, y)
# MAGIC
# MAGIC print(f"Alpha óptimo: {model.alpha_}")
# MAGIC print(f"R² Score: {model.score(X_scaled, y):.4f}")
# MAGIC
# MAGIC selected_features = X.columns[model.coef_ != 0].tolist()
# MAGIC print(f"Features seleccionadas: {len(selected_features)}")
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 🎯 Método 2: Ridge (L2) vs. ElasticNet
# MAGIC
# MAGIC **Ridge (L2)**:
# MAGIC - **No elimina features** (coeficientes pequeños pero != 0)
# MAGIC - Útil para **regularización**, no para selección
# MAGIC
# MAGIC **ElasticNet** (L1 + L2):
# MAGIC - Combina Lasso y Ridge
# MAGIC - Balance entre **selección** y **estabilidad**
# MAGIC
# MAGIC ```python
# MAGIC from sklearn.linear_model import ElasticNet
# MAGIC
# MAGIC model = ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42)
# MAGIC model.fit(X_scaled, y)
# MAGIC
# MAGIC selected_features = X.columns[model.coef_ != 0].tolist()
# MAGIC print(f"Features seleccionadas: {selected_features}")
# MAGIC ```
# MAGIC
# MAGIC **l1_ratio**:
# MAGIC - `l1_ratio=1`: Solo L1 (Lasso)
# MAGIC - `l1_ratio=0`: Solo L2 (Ridge)
# MAGIC - `l1_ratio=0.5`: 50% L1 + 50% L2
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 🎯 Método 3: Tree-Based Feature Importance
# MAGIC
# MAGIC **Idea**: Árboles de decisión calculan **importancia** de features automáticamente.
# MAGIC
# MAGIC **Métrica**: Reducción promedio de impureza (Gini o entropía).
# MAGIC
# MAGIC #### Random Forest Feature Importance
# MAGIC
# MAGIC ```python
# MAGIC from sklearn.ensemble import RandomForestRegressor
# MAGIC
# MAGIC model = RandomForestRegressor(n_estimators=100, random_state=42)
# MAGIC model.fit(X, y)
# MAGIC
# MAGIC # Feature importance
# MAGIC importances = pd.DataFrame({
# MAGIC     'feature': X.columns,
# MAGIC     'importance': model.feature_importances_
# MAGIC }).sort_values('importance', ascending=False)
# MAGIC
# MAGIC print(importances)
# MAGIC
# MAGIC # Visualizar
# MAGIC import matplotlib.pyplot as plt
# MAGIC
# MAGIC plt.figure(figsize=(10, 6))
# MAGIC plt.barh(importances['feature'][:15], importances['importance'][:15])
# MAGIC plt.xlabel('Importancia')
# MAGIC plt.title('Top 15 Features por Importancia (Random Forest)')
# MAGIC plt.gca().invert_yaxis()
# MAGIC plt.show()
# MAGIC
# MAGIC # Seleccionar top 10
# MAGIC top_features = importances['feature'][:10].tolist()
# MAGIC X_selected = X[top_features]
# MAGIC ```
# MAGIC
# MAGIC #### XGBoost Feature Importance
# MAGIC
# MAGIC ```python
# MAGIC import xgboost as xgb
# MAGIC
# MAGIC model = xgb.XGBRegressor(n_estimators=100, random_state=42)
# MAGIC model.fit(X, y)
# MAGIC
# MAGIC # Feature importance
# MAGIC importances = pd.DataFrame({
# MAGIC     'feature': X.columns,
# MAGIC     'importance': model.feature_importances_
# MAGIC }).sort_values('importance', ascending=False)
# MAGIC
# MAGIC print(importances.head(15))
# MAGIC
# MAGIC # XGBoost tiene plot_importance built-in
# MAGIC xgb.plot_importance(model, max_num_features=15, importance_type='weight')
# MAGIC plt.show()
# MAGIC ```
# MAGIC
# MAGIC **Tipos de importancia en XGBoost**:
# MAGIC - `weight`: Número de veces que se usa la feature
# MAGIC - `gain`: Ganancia promedio al usar la feature
# MAGIC - `cover`: Número promedio de muestras afectadas
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 🎯 Método 4: SelectFromModel
# MAGIC
# MAGIC **Idea**: Seleccionar features usando **cualquier modelo con feature importance**.
# MAGIC
# MAGIC ```python
# MAGIC from sklearn.feature_selection import SelectFromModel
# MAGIC from sklearn.ensemble import RandomForestRegressor
# MAGIC
# MAGIC # Entrenar modelo
# MAGIC model = RandomForestRegressor(n_estimators=100, random_state=42)
# MAGIC model.fit(X, y)
# MAGIC
# MAGIC # Seleccionar features con importancia > threshold
# MAGIC selector = SelectFromModel(model, threshold='median')  # median, mean, o valor específico
# MAGIC selector.fit(X, y)
# MAGIC
# MAGIC X_selected = selector.transform(X)
# MAGIC
# MAGIC # Ver features seleccionadas
# MAGIC selected_features = X.columns[selector.get_support()].tolist()
# MAGIC print(f"Features seleccionadas: {selected_features}")
# MAGIC print(f"Número: {len(selected_features)}")
# MAGIC ```
# MAGIC
# MAGIC **Thresholds comunes**:
# MAGIC - `'median'`: Selecciona 50% de features
# MAGIC - `'mean'`: Selecciona features sobre la media
# MAGIC - `'1.5*mean'`: Selecciona solo las muy importantes
# MAGIC - `0.01`: Importancia > 0.01
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### ⚖️ Comparación de Métodos Embedded
# MAGIC
# MAGIC | Método | Tipo Modelo | Velocidad | Interpretabilidad | Cuándo Usar |
# MAGIC |---------|-------------|-----------|-------------------|---------------|
# MAGIC | **Lasso** | Lineal | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Relaciones lineales |
# MAGIC | **ElasticNet** | Lineal | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Features correlacionadas |
# MAGIC | **RF Importance** | Tree-based | ⭐⭐⭐⭐ | ⭐⭐⭐ | Relaciones no lineales |
# MAGIC | **XGBoost Importance** | Tree-based | ⭐⭐⭐ | ⭐⭐⭐ | Máximo rendimiento |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 🎯 Cuándo Usar Métodos Embedded
# MAGIC
# MAGIC ✅ **Usar cuando**:
# MAGIC - Quieres **balance** entre velocidad y rendimiento
# MAGIC - Tienes modelo con **feature importance** (RF, XGBoost)
# MAGIC - Dataset **mediano a grande** (1000-10000 features)
# MAGIC - Buscas **interpretabilidad** (Lasso)
# MAGIC
# MAGIC ❌ **No usar cuando**:
# MAGIC - Modelo no tiene feature importance (KNN, SVM sin kernel)
# MAGIC - Quieres máximo rendimiento (usar wrapper)
# MAGIC
# MAGIC ---

# COMMAND ----------

# DBTITLE 1,Título del notebook
# MAGIC %md
# MAGIC # 🎯 Selección de Características (Feature Selection)
# MAGIC ## Material Complementario - Laboratorio (Herramientas)
# MAGIC ### Universidad del Aconcagua - Mendoza, Argentina
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 🎯 Objetivos de Aprendizaje
# MAGIC
# MAGIC 1. Comprender **por qué seleccionar features** (maldición de la dimensionalidad)
# MAGIC 2. Dominar **métodos de filtro** (varianza, correlación, MI)
# MAGIC 3. Aplicar **métodos wrapper** (RFE, forward/backward selection)
# MAGIC 4. Usar **métodos embedded** (Lasso, tree-based importance)
# MAGIC 5. Analizar **importancia de features** (permutation, SHAP)
# MAGIC 6. Integrar selección con **validación cruzada**
# MAGIC 7. Aplicar técnicas a **features H3 y temporales**
# MAGIC
# MAGIC ### 📁 Contenido
# MAGIC
# MAGIC 1. ¿Por qué Selección de Características?
# MAGIC 2. Métodos de Filtro (Filter Methods)
# MAGIC 3. Métodos Wrapper (Wrapper Methods)
# MAGIC 4. Métodos Embedded (Embedded Methods)
# MAGIC 5. Feature Importance y Permutation Importance
# MAGIC 6. Selección con Validación Cruzada
# MAGIC 7. Casos Prácticos: Features H3 y Temporales
# MAGIC 8. Comparación de Métodos
# MAGIC 9. Mejores Prácticas
# MAGIC
# MAGIC ### ⏱️ Duración Estimada: 2-3 horas
# MAGIC
# MAGIC ---

# COMMAND ----------

# DBTITLE 1,1. ¿Por qué Selección de Características?
# MAGIC %md
# MAGIC ## 1️⃣ ¿Por qué Necesitamos Selección de Características?
# MAGIC
# MAGIC ### 🚨 Problemas con Muchas Features
# MAGIC
# MAGIC **Escenario típico**:
# MAGIC ```python
# MAGIC # Dataset con 100 features
# MAGIC X_train.shape  # (10000, 100)
# MAGIC ```
# MAGIC
# MAGIC ❌ **Problemas**:
# MAGIC
# MAGIC #### 1. **Maldición de la Dimensionalidad** (Curse of Dimensionality)
# MAGIC
# MAGIC - Más features → espacio de búsqueda **exponencialmente mayor**
# MAGIC - Datos se vuelven **dispersos** (sparse)
# MAGIC - Se necesita **mucho más datos** para entrenar bien
# MAGIC
# MAGIC **Ejemplo**:
# MAGIC ```
# MAGIC 10 features → necesitas ~1,000 registros
# MAGIC 100 features → necesitas ~100,000 registros
# MAGIC 1,000 features → necesitas ~10,000,000 registros
# MAGIC ```
# MAGIC
# MAGIC #### 2. **Overfitting**
# MAGIC
# MAGIC - Modelo aprende **ruido** en lugar de patrones reales
# MAGIC - Buen rendimiento en train, **mal rendimiento en test**
# MAGIC
# MAGIC #### 3. **Tiempo de Entrenamiento**
# MAGIC
# MAGIC - Más features → más tiempo de cómputo
# MAGIC - 100 features puede ser **10x más lento** que 10 features
# MAGIC
# MAGIC #### 4. **Interpretabilidad**
# MAGIC
# MAGIC - Difícil explicar un modelo con 100 features
# MAGIC - Stakeholders quieren modelos **simples y entendibles**
# MAGIC
# MAGIC #### 5. **Features Redundantes o Irrelevantes**
# MAGIC
# MAGIC - **Redundantes**: Correlacionadas entre sí (ej: peso en kg y peso en lb)
# MAGIC - **Irrelevantes**: No aportan información (ej: ID de cliente para predecir ventas)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### ✅ Beneficios de Selección de Features
# MAGIC
# MAGIC 1. ✅ **Mejor rendimiento**: Menos overfitting
# MAGIC 2. ✅ **Más rápido**: Menor tiempo de entrenamiento e inferencia
# MAGIC 3. ✅ **Más interpretable**: Modelos más simples
# MAGIC 4. ✅ **Menos datos necesarios**: Evita maldición de dimensionalidad
# MAGIC 5. ✅ **Menos almacenamiento**: Menos features en producción
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 📊 Ejemplo Visual
# MAGIC
# MAGIC **Dataset original**: 50 features
# MAGIC ```
# MAGIC [edad, ingreso, ciudad, producto_1, producto_2, ..., producto_45, ruido_1, ruido_2, ruido_3]
# MAGIC ```
# MAGIC
# MAGIC **Después de selección**: 10 features importantes
# MAGIC ```
# MAGIC [edad, ingreso, ciudad, producto_3, producto_7, producto_12, producto_20, producto_31, producto_40, producto_45]
# MAGIC ```
# MAGIC
# MAGIC **Resultados**:
# MAGIC - Accuracy: 85% → **87%** (✅ mejor)
# MAGIC - Tiempo de entrenamiento: 5 min → **30 seg** (✅ 5x más rápido)
# MAGIC - Interpretabilidad: ❌ Compleja → ✅ **Simple**
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 📦 Tipos de Features
# MAGIC
# MAGIC | Tipo | Descripción | Acción |
# MAGIC |------|-------------|----------|
# MAGIC | **⭐ Relevantes** | Correlacionadas con el target | ✅ **Mantener** |
# MAGIC | **🔄 Redundantes** | Correlacionadas entre sí | ❌ **Eliminar una** |
# MAGIC | **🚫 Irrelevantes** | No aportan información | ❌ **Eliminar** |
# MAGIC | **🎲 Ruido** | Patrones aleatorios sin sentido | ❌ **Eliminar** |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 🧠 ¿Selección vs. Extracción de Features?
# MAGIC
# MAGIC **Feature Selection** (lo que veremos):
# MAGIC - **Selecciona subconjunto** de features existentes
# MAGIC - Mantiene features **originales e interpretables**
# MAGIC - Métodos: filtro, wrapper, embedded
# MAGIC
# MAGIC **Feature Extraction** (PCA, t-SNE, etc.):
# MAGIC - **Crea nuevas features** combinando las originales
# MAGIC - Features transformadas (**menos interpretables**)
# MAGIC - Métodos: PCA, LDA, autoencoders
# MAGIC
# MAGIC 🎯 **Este módulo se enfoca en Feature Selection.**
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 📊 Cuántas Features Seleccionar?
# MAGIC
# MAGIC **Regla general**:
# MAGIC ```
# MAGIC Número de features ≈ sqrt(Número de registros) / 10
# MAGIC ```
# MAGIC
# MAGIC **Ejemplos**:
# MAGIC - 100 registros → ~1 feature
# MAGIC - 1,000 registros → ~3 features
# MAGIC - 10,000 registros → ~10 features
# MAGIC - 100,000 registros → ~30 features
# MAGIC
# MAGIC ⚠️ **No es una regla estricta**, solo una guía inicial.
# MAGIC
# MAGIC **Mejor enfoque**: Usar **validación cruzada** para encontrar el número óptimo.
# MAGIC
# MAGIC ---

# COMMAND ----------

# DBTITLE 1,2. Métodos de Filtro (Filter)
# MAGIC %md
# MAGIC ## 2️⃣ Métodos de Filtro (Filter Methods)
# MAGIC
# MAGIC ### 📖 Concepto
# MAGIC
# MAGIC **Métodos de filtro** evalúan cada feature **individualmente** usando métricas estadísticas, **independientes del modelo**.
# MAGIC
# MAGIC **Ventajas**:
# MAGIC - ✅ **Rápidos**: No entrenan modelos
# MAGIC - ✅ **Escalables**: Funcionan con miles de features
# MAGIC - ✅ **Independientes del modelo**: Funcionan con cualquier algoritmo
# MAGIC
# MAGIC **Desventajas**:
# MAGIC - ❌ **No consideran interacciones**: Evalúan features aisladamente
# MAGIC - ❌ **Pueden descartar features útiles**: En combinación con otras
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 📊 Método 1: Varianza Baja (Low Variance)
# MAGIC
# MAGIC **Idea**: Eliminar features con **poca variabilidad** (casi constantes).
# MAGIC
# MAGIC **¿Por qué?** Si una feature es casi constante, no aporta información.
# MAGIC
# MAGIC **Ejemplo**:
# MAGIC ```python
# MAGIC feature_A = [1, 1, 1, 1, 1, 1, 1, 1]     # Varianza = 0 → Eliminar
# MAGIC feature_B = [1, 2, 1, 2, 1, 2, 1, 2]     # Varianza > 0 → Mantener
# MAGIC ```
# MAGIC
# MAGIC **Implementación**:
# MAGIC ```python
# MAGIC from sklearn.feature_selection import VarianceThreshold
# MAGIC
# MAGIC # Eliminar features con varianza < 0.01
# MAGIC selector = VarianceThreshold(threshold=0.01)
# MAGIC X_selected = selector.fit_transform(X)
# MAGIC
# MAGIC print(f"Features originales: {X.shape[1]}")
# MAGIC print(f"Features seleccionadas: {X_selected.shape[1]}")
# MAGIC ```
# MAGIC
# MAGIC ⚠️ **Importante**: Escalar datos antes (StandardScaler) para comparar varianzas.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 📊 Método 2: Correlación con el Target
# MAGIC
# MAGIC **Idea**: Seleccionar features **más correlacionadas** con el target.
# MAGIC
# MAGIC **Métricas**:
# MAGIC - **Pearson correlation**: Para features **continuas** y relación **lineal**
# MAGIC - **Spearman correlation**: Para relaciones **monotónicas** (no lineales)
# MAGIC
# MAGIC **Implementación**:
# MAGIC ```python
# MAGIC import pandas as pd
# MAGIC
# MAGIC # Calcular correlaciones
# MAGIC correlations = df.corr()['target'].abs().sort_values(ascending=False)
# MAGIC
# MAGIC # Seleccionar top 10 features
# MAGIC top_features = correlations[1:11].index.tolist()  # Excluir el target mismo
# MAGIC
# MAGIC print("Top 10 features por correlación:")
# MAGIC print(correlations[1:11])
# MAGIC ```
# MAGIC
# MAGIC **Visualización**:
# MAGIC ```python
# MAGIC import seaborn as sns
# MAGIC import matplotlib.pyplot as plt
# MAGIC
# MAGIC # Heatmap de correlaciones
# MAGIC plt.figure(figsize=(12, 8))
# MAGIC sns.heatmap(df[top_features + ['target']].corr(), annot=True, cmap='coolwarm', center=0)
# MAGIC plt.title('Correlación de Top Features con Target')
# MAGIC plt.show()
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 📊 Método 3: Chi-Cuadrado (χ²)
# MAGIC
# MAGIC **Uso**: **Clasificación** con features **categóricas** o **binarias**.
# MAGIC
# MAGIC **Idea**: Mide **dependencia** entre feature y target.
# MAGIC
# MAGIC **Hipótesis**:
# MAGIC - H₀: Feature y target son **independientes**
# MAGIC - H₁: Feature y target **NO son independientes**
# MAGIC
# MAGIC **Implementación**:
# MAGIC ```python
# MAGIC from sklearn.feature_selection import SelectKBest, chi2
# MAGIC
# MAGIC # Seleccionar top 10 features por chi-cuadrado
# MAGIC selector = SelectKBest(score_func=chi2, k=10)
# MAGIC X_selected = selector.fit_transform(X, y)
# MAGIC
# MAGIC # Ver scores
# MAGIC scores = pd.DataFrame({
# MAGIC     'feature': X.columns,
# MAGIC     'chi2_score': selector.scores_
# MAGIC }).sort_values('chi2_score', ascending=False)
# MAGIC
# MAGIC print(scores.head(10))
# MAGIC ```
# MAGIC
# MAGIC ⚠️ **Importante**: Features deben ser **no negativas** para chi2.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 📊 Método 4: Mutual Information (MI)
# MAGIC
# MAGIC **Idea**: Mide **cuánta información** aporta una feature sobre el target.
# MAGIC
# MAGIC **Ventaja**: Captura relaciones **no lineales**.
# MAGIC
# MAGIC **Uso**:
# MAGIC - `mutual_info_classif`: Para **clasificación**
# MAGIC - `mutual_info_regression`: Para **regresión**
# MAGIC
# MAGIC **Implementación (Regresión)**:
# MAGIC ```python
# MAGIC from sklearn.feature_selection import SelectKBest, mutual_info_regression
# MAGIC
# MAGIC # Seleccionar top 15 features por MI
# MAGIC selector = SelectKBest(score_func=mutual_info_regression, k=15)
# MAGIC X_selected = selector.fit_transform(X, y)
# MAGIC
# MAGIC # Ver scores
# MAGIC scores = pd.DataFrame({
# MAGIC     'feature': X.columns,
# MAGIC     'mi_score': selector.scores_
# MAGIC }).sort_values('mi_score', ascending=False)
# MAGIC
# MAGIC print(scores.head(15))
# MAGIC ```
# MAGIC
# MAGIC **Visualización**:
# MAGIC ```python
# MAGIC plt.figure(figsize=(10, 6))
# MAGIC plt.barh(scores['feature'][:15], scores['mi_score'][:15])
# MAGIC plt.xlabel('Mutual Information Score')
# MAGIC plt.title('Top 15 Features por Mutual Information')
# MAGIC plt.gca().invert_yaxis()
# MAGIC plt.show()
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 📊 Método 5: ANOVA F-test
# MAGIC
# MAGIC **Uso**: **Clasificación** con features **continuas**.
# MAGIC
# MAGIC **Idea**: Mide si las medias de la feature son **diferentes** entre clases.
# MAGIC
# MAGIC **Implementación**:
# MAGIC ```python
# MAGIC from sklearn.feature_selection import SelectKBest, f_classif
# MAGIC
# MAGIC # Seleccionar top 10 features por F-score
# MAGIC selector = SelectKBest(score_func=f_classif, k=10)
# MAGIC X_selected = selector.fit_transform(X, y)
# MAGIC
# MAGIC # Ver scores
# MAGIC scores = pd.DataFrame({
# MAGIC     'feature': X.columns,
# MAGIC     'f_score': selector.scores_,
# MAGIC     'p_value': selector.pvalues_
# MAGIC }).sort_values('f_score', ascending=False)
# MAGIC
# MAGIC print(scores.head(10))
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### ⚖️ Comparación de Métodos de Filtro
# MAGIC
# MAGIC | Método | Tipo Target | Tipo Features | Captura No Linealidad | Velocidad |
# MAGIC |---------|-------------|---------------|------------------------|------------|
# MAGIC | **Varianza** | N/A | Numéricas | N/A | ⭐⭐⭐⭐⭐ |
# MAGIC | **Correlación** | Continuo | Numéricas | ❌ No | ⭐⭐⭐⭐⭐ |
# MAGIC | **Chi²** | Categórico | Categóricas | ✅ Sí | ⭐⭐⭐⭐ |
# MAGIC | **Mutual Info** | Ambos | Ambas | ✅ Sí | ⭐⭐⭐ |
# MAGIC | **ANOVA F** | Categórico | Numéricas | ❌ No | ⭐⭐⭐⭐ |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 💻 Ejemplo Completo: Dataset de Panadería
# MAGIC
# MAGIC ```python
# MAGIC import pandas as pd
# MAGIC from sklearn.feature_selection import SelectKBest, mutual_info_regression
# MAGIC
# MAGIC # Cargar datos
# MAGIC df = pd.read_csv('ventas.csv')
# MAGIC
# MAGIC # Preparar features y target
# MAGIC X = df[['dia_semana', 'mes', 'sucursal_id', 'h3_index_encoded', 'segmento_encoded']]
# MAGIC y = df['total']
# MAGIC
# MAGIC # Seleccionar top 3 features por MI
# MAGIC selector = SelectKBest(score_func=mutual_info_regression, k=3)
# MAGIC X_selected = selector.fit_transform(X, y)
# MAGIC
# MAGIC # Ver features seleccionadas
# MAGIC selected_features = X.columns[selector.get_support()].tolist()
# MAGIC print(f"Features seleccionadas: {selected_features}")
# MAGIC ```
# MAGIC
# MAGIC ---

# COMMAND ----------

