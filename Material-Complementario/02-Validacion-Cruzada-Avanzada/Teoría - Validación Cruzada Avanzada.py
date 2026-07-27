# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,8. Comparación de Estrategias
# MAGIC %md
# MAGIC ## 8️⃣ Comparación y Selección de Estrategia
# MAGIC
# MAGIC ### 📊 Tabla Comparativa
# MAGIC
# MAGIC | Estrategia | Ventajas | Desventajas | Cuándo Usar |
# MAGIC |------------|----------|-------------|---------------|
# MAGIC | **K-Fold** | Simple, rápido, general | No para datos especiales | Datos i.i.d. aleatorios |
# MAGIC | **Stratified K-Fold** | Preserva distribución | Solo clasificación/bins | Clases desbalanceadas |
# MAGIC | **Time Series CV** | Respeta temporalidad | Solo expanding window | Series temporales |
# MAGIC | **Group K-Fold** | Evita leakage grupos | Puede desbalancear folds | Datos agrupados (clientes) |
# MAGIC | **LOOCV** | Máxima precisión | Extremadamente lento | Datasets pequeños (<100) |
# MAGIC | **Spatial CV** | Evalúa generalización geográfica | Requiere coordenadas | Datos geoespaciales |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 🧩 Árbol de Decisión: ¿Qué Estrategia Usar?
# MAGIC
# MAGIC ```
# MAGIC ¿Tienes datos temporales (fechas)?
# MAGIC │
# MAGIC ├── SÍ → Time Series CV
# MAGIC │
# MAGIC └── NO → ¿Tienes grupos (clientes, usuarios, dispositivos)?
# MAGIC     │
# MAGIC     ├── SÍ → Group K-Fold
# MAGIC     │
# MAGIC     └── NO → ¿Tienes coordenadas geográficas?
# MAGIC         │
# MAGIC         ├── SÍ → Spatial CV (Group K-Fold con H3)
# MAGIC         │
# MAGIC         └── NO → ¿Es clasificación con clases desbalanceadas?
# MAGIC             │
# MAGIC             ├── SÍ → Stratified K-Fold
# MAGIC             │
# MAGIC             └── NO → ¿Dataset muy pequeño (<100 registros)?
# MAGIC                 │
# MAGIC                 ├── SÍ → 10-Fold CV o LOOCV
# MAGIC                 │
# MAGIC                 └── NO → K-Fold estándar (5 o 10 folds)
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 📊 Ejemplo Comparativo: Dataset de Panadería
# MAGIC
# MAGIC **Dataset**: 50,000 ventas de 500 clientes en 3 sucursales, con fechas
# MAGIC
# MAGIC ```python
# MAGIC import pandas as pd
# MAGIC from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold, TimeSeriesSplit, GroupKFold
# MAGIC from sklearn.ensemble import RandomForestRegressor
# MAGIC
# MAGIC # Cargar datos
# MAGIC df = pd.read_csv('ventas.csv')
# MAGIC X = df[['dia_semana', 'mes', 'sucursal_id']]
# MAGIC y = df['total']
# MAGIC
# MAGIC model = RandomForestRegressor(n_estimators=100, random_state=42)
# MAGIC
# MAGIC # 1. K-Fold estándar
# MAGIC scores_kfold = cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error')
# MAGIC
# MAGIC # 2. Time Series CV
# MAGIC df_sorted = df.sort_values('fecha')
# MAGIC X_ts = df_sorted[['dia_semana', 'mes', 'sucursal_id']]
# MAGIC y_ts = df_sorted['total']
# MAGIC scores_ts = cross_val_score(model, X_ts, y_ts, cv=TimeSeriesSplit(n_splits=5), scoring='neg_mean_absolute_error')
# MAGIC
# MAGIC # 3. Group K-Fold (por cliente)
# MAGIC groups = df['cliente_id']
# MAGIC scores_group = cross_val_score(model, X, y, groups=groups, cv=GroupKFold(n_splits=5), scoring='neg_mean_absolute_error')
# MAGIC
# MAGIC # Comparar
# MAGIC print("Comparación de Estrategias de CV:")
# MAGIC print(f"K-Fold estándar:     MAE = ${-scores_kfold.mean():.2f}  (puede ser optimista)")
# MAGIC print(f"Time Series CV:      MAE = ${-scores_ts.mean():.2f}  (realista para forecasting)")
# MAGIC print(f"Group K-Fold:        MAE = ${-scores_group.mean():.2f}  (realista para clientes nuevos)")
# MAGIC ```
# MAGIC
# MAGIC **Salida esperada**:
# MAGIC ```
# MAGIC Comparación de Estrategias de CV:
# MAGIC K-Fold estándar:     MAE = $14.50  (puede ser optimista)
# MAGIC Time Series CV:      MAE = $17.20  (realista para forecasting)
# MAGIC Group K-Fold:        MAE = $18.80  (realista para clientes nuevos)
# MAGIC ```
# MAGIC
# MAGIC 💡 **Insights**:
# MAGIC - K-Fold estándar subestima el error (data leakage)
# MAGIC - Time Series CV es más realista para predicciones futuras
# MAGIC - Group K-Fold es más conservador (clientes nunca vistos)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### ⚖️ Trade-offs
# MAGIC
# MAGIC | Dimensión | K-Fold | Stratified | Time Series | Group | LOOCV | Spatial |
# MAGIC |-----------|--------|------------|-------------|-------|-------|----------|
# MAGIC | **Velocidad** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐ | ⭐⭐⭐⭐ |
# MAGIC | **Precisión** | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
# MAGIC | **Uso de datos** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
# MAGIC | **Generalidad** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
# MAGIC
# MAGIC ---

# COMMAND ----------

# DBTITLE 1,9. Mejores Prácticas
# MAGIC %md
# MAGIC ## 9️⃣ Mejores Prácticas de Cross-Validation
# MAGIC
# MAGIC ### ✅ DO: Buenas Prácticas
# MAGIC
# MAGIC #### 1. **Seleccionar Estrategia Según Tipo de Datos**
# MAGIC ```python
# MAGIC # ✅ BIEN: Time Series CV para datos temporales
# MAGIC df_sorted = df.sort_values('fecha')
# MAGIC scores = cross_val_score(model, X, y, cv=TimeSeriesSplit(n_splits=5))
# MAGIC
# MAGIC # ❌ MAL: K-Fold para datos temporales
# MAGIC scores = cross_val_score(model, X, y, cv=5)  # Data leakage!
# MAGIC ```
# MAGIC
# MAGIC #### 2. **Siempre Usar `random_state` para Reproducibilidad**
# MAGIC ```python
# MAGIC # ✅ BIEN
# MAGIC kf = KFold(n_splits=5, shuffle=True, random_state=42)
# MAGIC
# MAGIC # ❌ MAL (resultados no reproducibles)
# MAGIC kf = KFold(n_splits=5, shuffle=True)  # Diferente cada vez
# MAGIC ```
# MAGIC
# MAGIC #### 3. **Reportar Media ± Desviación Estándar**
# MAGIC ```python
# MAGIC # ✅ BIEN
# MAGIC scores = cross_val_score(model, X, y, cv=5)
# MAGIC print(f"MAE: {-scores.mean():.2f} ± {scores.std():.2f}")
# MAGIC
# MAGIC # ❌ MAL (solo media, sin varianza)
# MAGIC print(f"MAE: {-scores.mean():.2f}")
# MAGIC ```
# MAGIC
# MAGIC #### 4. **Usar Mismo CV para Comparar Modelos**
# MAGIC ```python
# MAGIC # ✅ BIEN: Mismo CV para comparación justa
# MAGIC kf = KFold(n_splits=5, shuffle=True, random_state=42)
# MAGIC
# MAGIC scores_rf = cross_val_score(model_rf, X, y, cv=kf)
# MAGIC scores_gb = cross_val_score(model_gb, X, y, cv=kf)
# MAGIC
# MAGIC # ❌ MAL: CVs diferentes
# MAGIC scores_rf = cross_val_score(model_rf, X, y, cv=5)
# MAGIC scores_gb = cross_val_score(model_gb, X, y, cv=3)  # No comparable
# MAGIC ```
# MAGIC
# MAGIC #### 5. **Pre-procesamiento Dentro del Fold**
# MAGIC ```python
# MAGIC from sklearn.pipeline import Pipeline
# MAGIC from sklearn.preprocessing import StandardScaler
# MAGIC
# MAGIC # ✅ BIEN: Scaler dentro del pipeline (se ajusta en cada fold)
# MAGIC pipeline = Pipeline([
# MAGIC     ('scaler', StandardScaler()),
# MAGIC     ('model', RandomForestRegressor())
# MAGIC ])
# MAGIC scores = cross_val_score(pipeline, X, y, cv=5)
# MAGIC
# MAGIC # ❌ MAL: Scaler fuera (usa info de test en train)
# MAGIC X_scaled = StandardScaler().fit_transform(X)  # Data leakage!
# MAGIC scores = cross_val_score(model, X_scaled, y, cv=5)
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### ❌ DON'T: Errores Comunes
# MAGIC
# MAGIC #### 1. **NO Usar K-Fold para Series Temporales**
# MAGIC ```python
# MAGIC # ❌ MAL
# MAGIC df_temporal = pd.read_csv('ventas_diarias.csv')
# MAGIC scores = cross_val_score(model, X, y, cv=5)  # Entrena con futuro!
# MAGIC
# MAGIC # ✅ BIEN
# MAGIC df_temporal = df_temporal.sort_values('fecha')
# MAGIC scores = cross_val_score(model, X, y, cv=TimeSeriesSplit(n_splits=5))
# MAGIC ```
# MAGIC
# MAGIC #### 2. **NO Olvidar `groups=` en Group K-Fold**
# MAGIC ```python
# MAGIC # ❌ MAL
# MAGIC gkf = GroupKFold(n_splits=5)
# MAGIC scores = cross_val_score(model, X, y, cv=gkf)  # Error o leakage!
# MAGIC
# MAGIC # ✅ BIEN
# MAGIC scores = cross_val_score(model, X, y, groups=df['cliente_id'], cv=gkf)
# MAGIC ```
# MAGIC
# MAGIC #### 3. **NO Hacer Feature Engineering con Datos de Test**
# MAGIC ```python
# MAGIC # ❌ MAL: Crear features usando TODO el dataset
# MAGIC df['mean_by_client'] = df.groupby('cliente_id')['total'].transform('mean')  # Leakage!
# MAGIC scores = cross_val_score(model, X, y, cv=5)
# MAGIC
# MAGIC # ✅ BIEN: Crear features dentro del fold (usar Pipeline o FunctionTransformer)
# MAGIC ```
# MAGIC
# MAGIC #### 4. **NO Usar LOOCV con Datasets Grandes**
# MAGIC ```python
# MAGIC # ❌ MAL: 10,000 modelos a entrenar
# MAGIC scores = cross_val_score(model, X, y, cv=LeaveOneOut())  # Muy lento!
# MAGIC
# MAGIC # ✅ BIEN: 5 o 10 folds es suficiente
# MAGIC scores = cross_val_score(model, X, y, cv=10)
# MAGIC ```
# MAGIC
# MAGIC #### 5. **NO Ignorar Advertencias de Clases Faltantes**
# MAGIC ```python
# MAGIC # ❌ MAL: Ignorar warning de clases desbalanceadas
# MAGIC scores = cross_val_score(model, X, y, cv=5)  # Algunos folds sin clase minoritaria
# MAGIC
# MAGIC # ✅ BIEN: Usar Stratified K-Fold
# MAGIC scores = cross_val_score(model, X, y, cv=StratifiedKFold(n_splits=5))
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 📊 Cuántos Folds Usar?
# MAGIC
# MAGIC **Recomendaciones generales**:
# MAGIC
# MAGIC | Tamaño Dataset | Nº Folds | Razón |
# MAGIC |-----------------|-----------|--------|
# MAGIC | < 100 | 10 o LOOCV | Maximizar datos de entrenamiento |
# MAGIC | 100 - 1,000 | 10 | Buen balance |
# MAGIC | 1,000 - 10,000 | 5 | Velocidad vs. precisión |
# MAGIC | > 10,000 | 3-5 | Suficiente para estimación robusta |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### ⚙️ Parámetros Importantes
# MAGIC
# MAGIC ```python
# MAGIC from sklearn.model_selection import cross_val_score
# MAGIC
# MAGIC scores = cross_val_score(
# MAGIC     estimator=model,           # Modelo a evaluar
# MAGIC     X=X,                       # Features
# MAGIC     y=y,                       # Target
# MAGIC     groups=groups,             # Grupos (para GroupKFold)
# MAGIC     cv=5,                      # Estrategia CV (int o objeto)
# MAGIC     scoring='neg_mean_absolute_error',  # Métrica
# MAGIC     n_jobs=-1,                 # Paralelización (-1 = todos los cores)
# MAGIC     verbose=1,                 # Mostrar progreso
# MAGIC     fit_params=None,           # Parámetros extras para fit()
# MAGIC     pre_dispatch='2*n_jobs'    # Control de memoria
# MAGIC )
# MAGIC ```
# MAGIC
# MAGIC **Métricas comunes**:
# MAGIC - Regresión: `'neg_mean_absolute_error'`, `'neg_mean_squared_error'`, `'r2'`
# MAGIC - Clasificación: `'accuracy'`, `'f1'`, `'roc_auc'`, `'precision'`, `'recall'`
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 💡 Tips Finales
# MAGIC
# MAGIC 1. ✅ **Siempre visualiza la distribución** de train/test en cada fold
# MAGIC 2. ✅ **Verifica que no haya data leakage** (revisa features y timestamps)
# MAGIC 3. ✅ **Usa nested CV** para optimización de hiperparámetros + evaluación
# MAGIC 4. ✅ **Documenta la estrategia CV** usada en tus experimentos
# MAGIC 5. ✅ **Considera el costo computacional** (LOOCV vs. 5-Fold)
# MAGIC 6. ✅ **Reporta intervalos de confianza** (media ± std)
# MAGIC
# MAGIC ---

# COMMAND ----------

# DBTITLE 1,Setup y datos
# Instalar dependencias
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    cross_val_score, cross_validate,
    KFold, StratifiedKFold, TimeSeriesSplit, GroupKFold, LeaveOneOut
)
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, accuracy_score

print("✅ Librerías importadas")

# COMMAND ----------

# DBTITLE 1,Cargar datos de la panadería
# Cargar datasets
ruta_datos = '/Workspace/Users/cortega@uda.edu.ar/Laboratorio/Datasets/'

df_ventas = pd.read_csv(ruta_datos + 'ventas.csv')
df_clientes = pd.read_csv(ruta_datos + 'clientes.csv')

print("✅ Datasets cargados")
print(f"   Ventas: {len(df_ventas):,}")
print(f"   Clientes: {len(df_clientes):,}")

# Preparar datos
df_ventas['fecha'] = pd.to_datetime(df_ventas['fecha'])
df_ventas = df_ventas.sort_values('fecha').reset_index(drop=True)

df_ventas['dia_semana'] = df_ventas['fecha'].dt.dayofweek
df_ventas['mes'] = df_ventas['fecha'].dt.month
df_ventas['dia_mes'] = df_ventas['fecha'].dt.day

print("\n✅ Features temporales creadas")

# COMMAND ----------

# DBTITLE 1,Comparación práctica de estrategias
print("="*80)
print("COMPARACIÓN DE ESTRATEGIAS DE CROSS-VALIDATION")
print("="*80)

# Preparar datos
df_ml = df_ventas[df_ventas['cliente_id'].notna()].merge(
    df_clientes[['cliente_id', 'segmento']], 
    on='cliente_id'
)
df_ml['segmento_encoded'] = df_ml['segmento'].astype('category').cat.codes

features = ['sucursal_id', 'dia_semana', 'mes', 'segmento_encoded']
X = df_ml[features]
y = df_ml['total']

model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)

print(f"\n📊 Dataset: {len(X):,} registros, {len(features)} features")
print(f"\nProbando diferentes estrategias de CV...\n")

# 1. K-Fold estándar
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores_kf = cross_val_score(model, X, y, cv=kf, scoring='neg_mean_absolute_error', n_jobs=-1)

# 2. Time Series CV
scores_ts = cross_val_score(model, X, y, cv=TimeSeriesSplit(n_splits=5), scoring='neg_mean_absolute_error', n_jobs=-1)

# 3. Group K-Fold (por cliente)
groups = df_ml['cliente_id'].values
gkf = GroupKFold(n_splits=5)
scores_gkf = cross_val_score(model, X, y, groups=groups, cv=gkf, scoring='neg_mean_absolute_error', n_jobs=-1)

# Resultados
print("Resultados:")
print(f"\n1️⃣ K-Fold estándar (5 folds):")
print(f"   MAE: ${-scores_kf.mean():.2f} ± ${scores_kf.std():.2f}")
print(f"   💡 Puede ser optimista (no considera estructura de datos)")

print(f"\n2️⃣ Time Series CV (5 folds):")
print(f"   MAE: ${-scores_ts.mean():.2f} ± ${scores_ts.std():.2f}")
print(f"   💡 Realista para forecasting (respeta orden temporal)")

print(f"\n3️⃣ Group K-Fold (por cliente):")
print(f"   MAE: ${-scores_gkf.mean():.2f} ± ${scores_gkf.std():.2f}")
print(f"   💡 Realista para clientes nuevos (evita data leakage)")

print(f"\n" + "="*80)
print("CONCLUSIÓN")
print("="*80)
diff_ts = (-scores_ts.mean()) - (-scores_kf.mean())
diff_gkf = (-scores_gkf.mean()) - (-scores_kf.mean())

print(f"\nDiferencia Time Series vs. K-Fold: ${diff_ts:.2f} ({(diff_ts/-scores_kf.mean()*100):.1f}% peor)")
print(f"Diferencia Group K-Fold vs. K-Fold: ${diff_gkf:.2f} ({(diff_gkf/-scores_kf.mean()*100):.1f}% peor)")
print(f"\n💡 La estrategia correcta revela el VERDADERO rendimiento del modelo.")

# COMMAND ----------

# DBTITLE 1,Conclusiones
# MAGIC %md
# MAGIC ## ✅ Conclusiones
# MAGIC
# MAGIC ### 🎯 Resumen del Módulo
# MAGIC
# MAGIC **Lo que aprendimos**:
# MAGIC
# MAGIC 1. ✅ **Limitaciones de train/test split simple** y por qué usar CV
# MAGIC 2. ✅ **K-Fold** para datos aleatorios (i.i.d.)
# MAGIC 3. ✅ **Stratified K-Fold** para clases desbalanceadas
# MAGIC 4. ✅ **Time Series CV** para respetar dependencia temporal
# MAGIC 5. ✅ **Group K-Fold** para evitar data leakage por agrupamiento
# MAGIC 6. ✅ **LOOCV** para datasets muy pequeños
# MAGIC 7. ✅ **Validación espacial** con features H3
# MAGIC 8. ✅ **Cómo seleccionar** la estrategia correcta
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 💡 Mensajes Clave
# MAGIC
# MAGIC 1. 🔑 **No hay una estrategia "mejor"** - depende de tus datos
# MAGIC 2. ⚠️ **La estrategia incorrecta** puede dar resultados demasiado optimistas
# MAGIC 3. 🎯 **Siempre considera la estructura** de tus datos (temporal, grupos, espacial)
# MAGIC 4. ⏱️ **Balance entre precisión y velocidad**: 5-Fold es generalmente suficiente
# MAGIC 5. 🔄 **Reproducibilidad**: Siempre usar `random_state`
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 📦 Guía Rápida de Selección
# MAGIC **¿Qué CV usar?**
# MAGIC
# MAGIC - 📅 **Datos con fechas** → Time Series CV
# MAGIC - 👥 **Múltiples registros por entidad** → Group K-Fold
# MAGIC - 🗺️ **Datos geoespaciales** → Spatial CV (Group K-Fold con H3)
# MAGIC - ⚖️ **Clases desbalanceadas** → Stratified K-Fold
# MAGIC - 🔬 **Dataset < 100 registros** → 10-Fold o LOOCV
# MAGIC - 🎲 **Datos aleatorios sin estructura especial** → K-Fold (5 o 10 folds)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 📚 Recursos Adicionales
# MAGIC
# MAGIC - [Scikit-learn Cross-Validation](https://scikit-learn.org/stable/modules/cross_validation.html)
# MAGIC - [Time Series Cross-Validation Best Practices](https://towardsdatascience.com/time-series-cross-validation)
# MAGIC - [Spatial Cross-Validation in R and Python](https://geocompr.robinlovelace.net/)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🎓 ¡Felicitaciones!
# MAGIC
# MAGIC **Has completado el módulo de Validación Cruzada Avanzada.**
# MAGIC
# MAGIC Ahora puedes:
# MAGIC - ✅ Seleccionar la estrategia de CV correcta según tipo de datos
# MAGIC - ✅ Evitar data leakage en evaluación de modelos
# MAGIC - ✅ Obtener estimaciones realistas de rendimiento
# MAGIC - ✅ Aplicar CV espacial con features H3
# MAGIC - ✅ Comparar modelos de forma justa
# MAGIC
# MAGIC **Próximo paso**: Notebook Práctico con ejercicios hands-on.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC **Universidad del Aconcagua**  
# MAGIC **Laboratorio (Herramientas)**  
# MAGIC **Mendoza, Argentina**

# COMMAND ----------

# DBTITLE 1,5. Group K-Fold
# MAGIC %md
# MAGIC ## 5️⃣ Group K-Fold (Evitar Data Leakage)
# MAGIC
# MAGIC ### 🚨 Problema: Data Leakage por Agrupamiento
# MAGIC
# MAGIC **Escenario**: Dataset con **múltiples registros del mismo cliente/usuario/entidad**.
# MAGIC
# MAGIC ```
# MAGIC Dataset de ventas de panadería:
# MAGIC ├── Cliente A: 50 compras
# MAGIC ├── Cliente B: 30 compras
# MAGIC ├── Cliente C: 20 compras
# MAGIC └── ...
# MAGIC ```
# MAGIC
# MAGIC ❌ **Con K-Fold estándar**:
# MAGIC ```
# MAGIC Train: [Cliente A - compra 1, Cliente A - compra 2, ...]
# MAGIC Test:  [Cliente A - compra 45, Cliente A - compra 46, ...]
# MAGIC ```
# MAGIC
# MAGIC **Problema**: El modelo **aprende sobre Cliente A** en training y lo **evalúa en test** → **data leakage**
# MAGIC
# MAGIC ✅ **Rendimiento inflado artificialmente** porque el modelo ya "conoce" al cliente.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### ✅ Solución: Group K-Fold
# MAGIC
# MAGIC **Concepto**: Asegurar que **todos los registros de un grupo estén en el mismo fold**.
# MAGIC
# MAGIC ```
# MAGIC Fold 1:
# MAGIC   Train: [Cliente B, Cliente C, Cliente D, ...]
# MAGIC   Test:  [Cliente A - todas sus compras]
# MAGIC   
# MAGIC Fold 2:
# MAGIC   Train: [Cliente A, Cliente C, Cliente D, ...]
# MAGIC   Test:  [Cliente B - todas sus compras]
# MAGIC ```
# MAGIC
# MAGIC ✅ **Ventaja**: Evalúa la capacidad del modelo de **generalizar a nuevos grupos** (clientes nunca vistos).
# MAGIC
# MAGIC ---
# MAGIC ### 💻 Implementación
# MAGIC
# MAGIC ```python
# MAGIC from sklearn.model_selection import GroupKFold, cross_val_score
# MAGIC from sklearn.ensemble import RandomForestRegressor
# MAGIC
# MAGIC # Definir grupos (IDs de clientes)
# MAGIC groups = df['cliente_id'].values
# MAGIC
# MAGIC # Group K-Fold
# MAGIC gkf = GroupKFold(n_splits=5)
# MAGIC
# MAGIC model = RandomForestRegressor(n_estimators=100, random_state=42)
# MAGIC
# MAGIC # Cross-validation con grupos
# MAGIC scores = cross_val_score(
# MAGIC     model, X, y, 
# MAGIC     groups=groups,  # 🔑 Importante: pasar grupos
# MAGIC     cv=gkf,
# MAGIC     scoring='neg_mean_absolute_error'
# MAGIC )
# MAGIC
# MAGIC print(f"MAE: {-scores.mean():.2f} ± {scores.std():.2f}")
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 📊 Ejemplo Visual
# MAGIC
# MAGIC **Dataset**: 1000 compras de 100 clientes
# MAGIC
# MAGIC **K-Fold estándar** (INCORRECTO):
# MAGIC ```
# MAGIC Train:  [C1-compra1, C1-compra2, C2-compra1, C3-compra1, ...]
# MAGIC Test:   [C1-compra3, C2-compra2, C3-compra2, ...]  ❌ Leakage!
# MAGIC ```
# MAGIC
# MAGIC **Group K-Fold** (CORRECTO):
# MAGIC ```
# MAGIC Train:  [C1-todas, C2-todas, C3-todas, ..., C80-todas]
# MAGIC Test:   [C81-todas, C82-todas, ..., C100-todas]     ✅ Sin leakage
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 🎯 Casos de Uso
# MAGIC
# MAGIC ✅ **Usar Group K-Fold cuando**:
# MAGIC
# MAGIC 1. **Múltiples transacciones por cliente**:
# MAGIC    - Predicción de churn
# MAGIC    - Recomendaciones
# MAGIC    - Credit scoring
# MAGIC
# MAGIC 2. **Múltiples mediciones por sujeto**:
# MAGIC    - Datos médicos (pacientes)
# MAGIC    - Experimentos (participantes)
# MAGIC    - Sensores (dispositivos)
# MAGIC
# MAGIC 3. **Datos jerárquicos**:
# MAGIC    - Estudiantes en escuelas
# MAGIC    - Empleados en empresas
# MAGIC    - Productos en categorías
# MAGIC
# MAGIC 4. **Datos geoespaciales agrupados**:
# MAGIC    - Ventas por sucursal
# MAGIC    - Mediciones por región
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### ⚠️ Consideraciones
# MAGIC
# MAGIC **1. Distribución desigual de grupos**:
# MAGIC ```python
# MAGIC # Algunos clientes tienen 1 compra, otros 100
# MAGIC # Los folds pueden ser desbalanceados en tamaño
# MAGIC ```
# MAGIC
# MAGIC **Solución**: Verificar distribución antes de CV:
# MAGIC ```python
# MAGIC print(df.groupby('cliente_id').size().describe())
# MAGIC ```
# MAGIC
# MAGIC **2. Grupos muy pequeños o muy grandes**:
# MAGIC - Grupo con 1 registro → poca información
# MAGIC - Grupo con 1000 registros → domina el fold
# MAGIC
# MAGIC **Solución**: Filtrar o balancear grupos si es necesario.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 🔄 Comparación: K-Fold vs. Group K-Fold
# MAGIC
# MAGIC **Experimento**: Predicción de ventas por cliente
# MAGIC
# MAGIC | Método | MAE Test | Comentario |
# MAGIC |--------|----------|------------|
# MAGIC | K-Fold estándar | $12.50 | ❌ **Optimista** (data leakage) |
# MAGIC | Group K-Fold | $18.30 | ✅ **Realista** (clientes nuevos) |
# MAGIC
# MAGIC 💡 **Diferencia de $5.80**: El impacto del data leakage!
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 💻 Ejemplo Completo: Panadería
# MAGIC
# MAGIC ```python
# MAGIC import pandas as pd
# MAGIC from sklearn.model_selection import GroupKFold, cross_val_score
# MAGIC from sklearn.ensemble import RandomForestRegressor
# MAGIC
# MAGIC # Cargar datos
# MAGIC df_ventas = pd.read_csv('ventas.csv')
# MAGIC
# MAGIC # Features y target
# MAGIC X = df_ventas[['dia_semana', 'mes', 'sucursal_id', 'segmento_encoded']]
# MAGIC y = df_ventas['total']
# MAGIC groups = df_ventas['cliente_id']  # Agrupar por cliente
# MAGIC
# MAGIC # Group K-Fold
# MAGIC gkf = GroupKFold(n_splits=5)
# MAGIC model = RandomForestRegressor(n_estimators=100, random_state=42)
# MAGIC
# MAGIC scores = cross_val_score(model, X, y, groups=groups, cv=gkf, scoring='neg_mean_absolute_error')
# MAGIC
# MAGIC print(f"MAE promedio: ${-scores.mean():.2f}")
# MAGIC print(f"Desviación estándar: ${scores.std():.2f}")
# MAGIC print(f"\n💡 Este MAE es realista para clientes NUEVOS")
# MAGIC ```
# MAGIC ---

# COMMAND ----------

# DBTITLE 1,6. Leave-One-Out CV
# MAGIC %md
# MAGIC ## 6️⃣ Leave-One-Out Cross-Validation (LOOCV)
# MAGIC
# MAGIC ### 📖 Concepto
# MAGIC
# MAGIC **LOOCV** es el caso extremo de K-Fold donde **K = N** (número de registros).
# MAGIC
# MAGIC **Proceso**:
# MAGIC - Para cada registro i:
# MAGIC   - Train: Todos los demás registros (N-1)
# MAGIC   - Test: Solo el registro i
# MAGIC - Entrenar **N modelos**
# MAGIC
# MAGIC **Visualización** (Dataset con 5 registros):
# MAGIC ```
# MAGIC Fold 1: [TEST] [TRAIN] [TRAIN] [TRAIN] [TRAIN]
# MAGIC Fold 2: [TRAIN] [TEST] [TRAIN] [TRAIN] [TRAIN]
# MAGIC Fold 3: [TRAIN] [TRAIN] [TEST] [TRAIN] [TRAIN]
# MAGIC Fold 4: [TRAIN] [TRAIN] [TRAIN] [TEST] [TRAIN]
# MAGIC Fold 5: [TRAIN] [TRAIN] [TRAIN] [TRAIN] [TEST]
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### ⚖️ Ventajas y Desventajas
# MAGIC
# MAGIC ✅ **Ventajas**:
# MAGIC 1. **Usa casi todos los datos** para entrenamiento (N-1)
# MAGIC 2. **Estimación casi insesgada** del error
# MAGIC 3. **Determinístico**: no depende de random splits
# MAGIC 4. **Útil con datasets pequeños** (< 100 registros)
# MAGIC
# MAGIC ❌ **Desventajas**:
# MAGIC 1. **Extremadamente lento**: N entrenamientos
# MAGIC    - 1000 registros = 1000 modelos a entrenar
# MAGIC 2. **Alta varianza** en la estimación
# MAGIC 3. **No funciona con algoritmos estocásticos** (redes neuronales)
# MAGIC 4. **Desperdicio computacional** con datasets grandes
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 💻 Implementación
# MAGIC
# MAGIC ```python
# MAGIC from sklearn.model_selection import LeaveOneOut, cross_val_score
# MAGIC from sklearn.linear_model import Ridge
# MAGIC
# MAGIC # LOOCV
# MAGIC loo = LeaveOneOut()
# MAGIC
# MAGIC # Usar modelo rápido (regresión lineal)
# MAGIC model = Ridge(alpha=1.0)
# MAGIC
# MAGIC # Cross-validation (puede ser MUY lento)
# MAGIC scores = cross_val_score(
# MAGIC     model, X, y, 
# MAGIC     cv=loo,
# MAGIC     scoring='neg_mean_squared_error'
# MAGIC )
# MAGIC
# MAGIC print(f"Número de folds: {len(scores)}  # = N registros")
# MAGIC print(f"MSE: {-scores.mean():.2f} ± {scores.std():.2f}")
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### ⏱️ Comparación de Tiempo de Ejecución
# MAGIC
# MAGIC **Dataset**: 1000 registros, Random Forest con 100 árboles
# MAGIC
# MAGIC | Método | Nº Modelos | Tiempo Estimado |
# MAGIC |---------|-------------|------------------|
# MAGIC | Train/Test Split | 1 | 1 segundo |
# MAGIC | 5-Fold CV | 5 | 5 segundos |
# MAGIC | 10-Fold CV | 10 | 10 segundos |
# MAGIC | **LOOCV** | **1000** | **1000 segundos (16 min)** |
# MAGIC
# MAGIC 🐢 **LOOCV es 200x más lento que 5-Fold!**
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 🎯 Cuándo Usar LOOCV
# MAGIC
# MAGIC ✅ **Usar cuando**:
# MAGIC - Dataset **muy pequeño** (N < 100)
# MAGIC - Modelo **rápido de entrenar** (regresión lineal, KNN)
# MAGIC - Necesitas **máxima precisión** en estimación
# MAGIC - No tienes restricciones de tiempo
# MAGIC
# MAGIC ❌ **NO usar cuando**:
# MAGIC - Dataset **grande** (N > 500)
# MAGIC - Modelo **lento** (RF, GB, redes neuronales)
# MAGIC - Tienes **tiempo limitado**
# MAGIC - **Casi siempre**: 5-Fold o 10-Fold es mejor opción
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 💡 Alternativa: Leave-P-Out
# MAGIC
# MAGIC **Leave-P-Out CV**: Dejar **P registros** fuera en cada fold.
# MAGIC
# MAGIC ```python
# MAGIC from sklearn.model_selection import LeavePOut
# MAGIC
# MAGIC # Dejar 2 registros fuera
# MAGIC lpo = LeavePOut(p=2)
# MAGIC
# MAGIC # Número de combinaciones = C(N, 2)
# MAGIC # Para N=100: 4,950 combinaciones!
# MAGIC ```
# MAGIC
# MAGIC ⚠️ **Aún más lento que LOOCV**. Raramente usado en práctica.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 📊 Ejemplo Práctico: Dataset Pequeño
# MAGIC
# MAGIC ```python
# MAGIC import numpy as np
# MAGIC from sklearn.model_selection import LeaveOneOut, cross_val_score
# MAGIC from sklearn.linear_model import LinearRegression
# MAGIC
# MAGIC # Dataset pequeño (50 registros)
# MAGIC np.random.seed(42)
# MAGIC X = np.random.randn(50, 5)
# MAGIC y = np.random.randn(50)
# MAGIC
# MAGIC # LOOCV
# MAGIC loo = LeaveOneOut()
# MAGIC model = LinearRegression()
# MAGIC
# MAGIC scores = cross_val_score(model, X, y, cv=loo, scoring='r2')
# MAGIC
# MAGIC print(f"R² promedio: {scores.mean():.3f}")
# MAGIC print(f"R² por observación: min={scores.min():.3f}, max={scores.max():.3f}")
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### ⚖️ Conclusión
# MAGIC
# MAGIC **LOOCV** es **teóricamente elegante** pero **prácticamente poco usado**.
# MAGIC
# MAGIC 🏆 **Recomendación general**: 
# MAGIC - Dataset pequeño (N < 100): **10-Fold CV**
# MAGIC - Dataset mediano/grande: **5-Fold CV**
# MAGIC - LOOCV: Solo si tienes **muy pocos datos** y modelo **muy rápido**
# MAGIC
# MAGIC ---

# COMMAND ----------

# DBTITLE 1,7. Validación Espacial con H3
# MAGIC %md
# MAGIC ## 7️⃣ Validación Espacial con H3
# MAGIC
# MAGIC ### 🗺️ Problema: Autocorrelación Espacial
# MAGIC
# MAGIC **Concepto**: Observaciones **cercanas geográficamente** tienden a ser **similares**.
# MAGIC
# MAGIC ```
# MAGIC Ejemplo: Ventas de sucursales
# MAGIC ├── Sucursal A (zona norte): $10,000/día
# MAGIC ├── Sucursal B (zona norte, 1km de A): $9,500/día  ⭐ Similar
# MAGIC └── Sucursal C (zona sur, 20km): $15,000/día        ❌ Diferente
# MAGIC ```
# MAGIC
# MAGIC ❌ **Con K-Fold estándar**:
# MAGIC ```
# MAGIC Train: [Zona Norte, Zona Norte, Zona Sur]
# MAGIC Test:  [Zona Norte, Zona Sur]
# MAGIC ```
# MAGIC
# MAGIC **Problema**: El modelo aprende sobre **Zona Norte** en train y predice **Zona Norte** en test → rendimiento inflado.
# MAGIC
# MAGIC ✅ **En producción**: Predecirás zonas **nuevas/lejanas** → peor rendimiento real.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### ✅ Solución: Spatial Cross-Validation
# MAGIC
# MAGIC **Estrategia**: Asegurar que train y test estén **espacialmente separados**.
# MAGIC
# MAGIC **Visualización** (hexágonos H3):
# MAGIC ```
# MAGIC Fold 1:
# MAGIC   Train: [⬣ ⬣] [ ] [ ] [ ]  (Zona Oeste)
# MAGIC   Test:  [ ] [ ] [⬢ ⬢ ⬢]  (Zona Este)
# MAGIC   
# MAGIC Fold 2:
# MAGIC   Train: [ ] [ ] [⬣ ⬣ ⬣]  (Zona Este)
# MAGIC   Test:  [⬢ ⬢] [ ] [ ] [ ]  (Zona Oeste)
# MAGIC ```
# MAGIC
# MAGIC ✅ **Ventaja**: Evalúa la capacidad del modelo de **generalizar a nuevas zonas geográficas**.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 💻 Implementación con H3
# MAGIC
# MAGIC #### Opción 1: Group K-Fold por Zona H3
# MAGIC
# MAGIC ```python
# MAGIC import h3
# MAGIC from sklearn.model_selection import GroupKFold, cross_val_score
# MAGIC from sklearn.ensemble import RandomForestRegressor
# MAGIC
# MAGIC # Agrupar por hexágono H3 (resolución 7)
# MAGIC df['h3_zona'] = df.apply(lambda row: h3.geo_to_h3(row['lat'], row['lon'], 7), axis=1)
# MAGIC
# MAGIC X = df[['dia_semana', 'mes', 'temp_celsius']]
# MAGIC y = df['ventas']
# MAGIC groups = df['h3_zona']  # Agrupar por zona
# MAGIC
# MAGIC # Group K-Fold espacial
# MAGIC gkf = GroupKFold(n_splits=5)
# MAGIC model = RandomForestRegressor(n_estimators=100, random_state=42)
# MAGIC
# MAGIC scores = cross_val_score(model, X, y, groups=groups, cv=gkf, scoring='neg_mean_absolute_error')
# MAGIC
# MAGIC print(f"MAE (validación espacial): ${-scores.mean():.2f}")
# MAGIC ```
# MAGIC
# MAGIC #### Opción 2: Split por Distancia (Custom)
# MAGIC
# MAGIC ```python
# MAGIC import numpy as np
# MAGIC from sklearn.model_selection import BaseCrossValidator
# MAGIC
# MAGIC class SpatialKFold(BaseCrossValidator):
# MAGIC     """Cross-validation espacial custom."""
# MAGIC     
# MAGIC     def __init__(self, n_splits=5, buffer_distance=3):
# MAGIC         self.n_splits = n_splits
# MAGIC         self.buffer_distance = buffer_distance  # Distancia mínima H3
# MAGIC     
# MAGIC     def split(self, X, y=None, groups=None):
# MAGIC         # groups = lista de h3_index
# MAGIC         unique_zones = np.unique(groups)
# MAGIC         np.random.shuffle(unique_zones)
# MAGIC         
# MAGIC         fold_size = len(unique_zones) // self.n_splits
# MAGIC         
# MAGIC         for i in range(self.n_splits):
# MAGIC             # Test zones
# MAGIC             test_zones = unique_zones[i*fold_size:(i+1)*fold_size]
# MAGIC             
# MAGIC             # Train zones (excluir buffer alrededor de test)
# MAGIC             train_zones = []
# MAGIC             for zone in unique_zones:
# MAGIC                 if zone not in test_zones:
# MAGIC                     # Verificar distancia mínima a zonas test
# MAGIC                     min_dist = min([h3.h3_distance(zone, tz) for tz in test_zones])
# MAGIC                     if min_dist >= self.buffer_distance:
# MAGIC                         train_zones.append(zone)
# MAGIC             
# MAGIC             train_idx = np.where(np.isin(groups, train_zones))[0]
# MAGIC             test_idx = np.where(np.isin(groups, test_zones))[0]
# MAGIC             
# MAGIC             yield train_idx, test_idx
# MAGIC     
# MAGIC     def get_n_splits(self, X=None, y=None, groups=None):
# MAGIC         return self.n_splits
# MAGIC
# MAGIC # Usar spatial CV custom
# MAGIC spatial_cv = SpatialKFold(n_splits=5, buffer_distance=3)
# MAGIC scores = cross_val_score(model, X, y, groups=df['h3_index'], cv=spatial_cv, scoring='neg_mean_absolute_error')
# MAGIC
# MAGIC print(f"MAE (con buffer espacial): ${-scores.mean():.2f}")
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 📊 Ejemplo: Predicción de Ventas por Sucursal
# MAGIC
# MAGIC **Dataset**: Ventas de 500 clientes en 3 sucursales (zonas H3 diferentes)
# MAGIC
# MAGIC ```python
# MAGIC import pandas as pd
# MAGIC import h3
# MAGIC from sklearn.model_selection import GroupKFold, cross_val_score
# MAGIC from sklearn.ensemble import RandomForestRegressor
# MAGIC
# MAGIC # Cargar datos
# MAGIC df_ventas = pd.read_csv('ventas.csv')
# MAGIC df_clientes = pd.read_csv('clientes.csv')
# MAGIC
# MAGIC # Merge para obtener H3
# MAGIC df = df_ventas.merge(df_clientes[['cliente_id', 'h3_index']], on='cliente_id')
# MAGIC
# MAGIC # Features
# MAGIC X = df[['dia_semana', 'mes', 'segmento_encoded']]
# MAGIC y = df['total']
# MAGIC
# MAGIC # Validación espacial
# MAGIC groups = df['h3_index']
# MAGIC gkf = GroupKFold(n_splits=5)
# MAGIC model = RandomForestRegressor(n_estimators=100, random_state=42)
# MAGIC
# MAGIC scores_spatial = cross_val_score(model, X, y, groups=groups, cv=gkf, scoring='neg_mean_absolute_error')
# MAGIC scores_standard = cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error')
# MAGIC
# MAGIC print(f"MAE (CV estándar):  ${-scores_standard.mean():.2f}  ❌ Optimista")
# MAGIC print(f"MAE (CV espacial):  ${-scores_spatial.mean():.2f}  ✅ Realista")
# MAGIC print(f"Diferencia:         ${(-scores_spatial.mean()) - (-scores_standard.mean()):.2f}")
# MAGIC ```
# MAGIC
# MAGIC **Salida esperada**:
# MAGIC ```
# MAGIC MAE (CV estándar):  $15.20  ❌ Optimista
# MAGIC MAE (CV espacial):  $18.90  ✅ Realista
# MAGIC Diferencia:         $3.70
# MAGIC ```
# MAGIC
# MAGIC 💡 **La validación espacial revela el verdadero rendimiento en zonas nuevas.**
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 🎯 Cuándo Usar Validación Espacial
# MAGIC
# MAGIC ✅ **Usar cuando**:
# MAGIC - Predicciones **geográficamente distribuidas**
# MAGIC - Datos con **autocorrelación espacial**
# MAGIC - Planeas **expandir a nuevas zonas**
# MAGIC - Tienes features H3 o coordenadas
# MAGIC
# MAGIC **Ejemplos**:
# MAGIC - Predicción de precios inmobiliarios
# MAGIC - Forecasting de ventas por sucursal
# MAGIC - Modelos de transporte/logística
# MAGIC - Análisis de sensores distribuidos
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### ⚠️ Consideraciones
# MAGIC
# MAGIC 1. **Resolución H3**: Ajustar según escala del problema
# MAGIC    - Resolución 5: ~250 km² (ciudad)
# MAGIC    - Resolución 7: ~5 km² (barrio)
# MAGIC    - Resolución 9: ~0.1 km² (cuadra)
# MAGIC
# MAGIC 2. **Buffer de distancia**: Evitar zonas "frontera" entre train/test
# MAGIC
# MAGIC 3. **Distribución desigual**: Algunas zonas tienen más datos que otras
# MAGIC
# MAGIC ---

# COMMAND ----------

# DBTITLE 1,Título del notebook
# MAGIC %md
# MAGIC # 🎯 Validación Cruzada Avanzada
# MAGIC ## Material Complementario - Laboratorio (Herramientas)
# MAGIC ### Universidad del Aconcagua - Mendoza, Argentina
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 🎯 Objetivos de Aprendizaje
# MAGIC
# MAGIC 1. Comprender **limitaciones de validación simple** (train/test split)
# MAGIC 2. Dominar **K-Fold Cross-Validation** y sus variantes
# MAGIC 3. Aplicar **Stratified K-Fold** para datos desbalanceados
# MAGIC 4. Usar **Time Series CV** para datos temporales
# MAGIC 5. Implementar **Group K-Fold** para evitar data leakage
# MAGIC 6. Explorar **validación espacial** con features H3
# MAGIC 7. Seleccionar estrategia correcta según tipo de datos
# MAGIC
# MAGIC ### 📁 Contenido
# MAGIC
# MAGIC 1. ¿Por qué Cross-Validation?
# MAGIC 2. K-Fold Cross-Validation (Revisión)
# MAGIC 3. Stratified K-Fold (Datos Desbalanceados)
# MAGIC 4. Time Series Cross-Validation
# MAGIC 5. Group K-Fold (Evitar Data Leakage)
# MAGIC 6. Leave-One-Out CV (LOOCV)
# MAGIC 7. Validación Espacial con H3
# MAGIC 8. Comparación y Selección de Estrategia
# MAGIC 9. Mejores Prácticas
# MAGIC
# MAGIC ### ⏱️ Duración Estimada: 2-3 horas
# MAGIC
# MAGIC ---

# COMMAND ----------

# DBTITLE 1,1. ¿Por qué Cross-Validation?
# MAGIC %md
# MAGIC ## 1️⃣ ¿Por qué Necesitamos Cross-Validation?
# MAGIC
# MAGIC ### 🚨 Problema con Train/Test Split Simple
# MAGIC
# MAGIC **Escenario típico**:
# MAGIC ```python
# MAGIC X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# MAGIC model.fit(X_train, y_train)
# MAGIC score = model.score(X_test, y_test)
# MAGIC ```
# MAGIC
# MAGIC ❌ **Problemas**:
# MAGIC
# MAGIC 1. **Varianza alta**: Resultado depende de **cómo se dividieron los datos**
# MAGIC    - Split 1: Accuracy = 85%
# MAGIC    - Split 2: Accuracy = 78%
# MAGIC    - Split 3: Accuracy = 91%
# MAGIC    - **¿Cuál es el verdadero rendimiento?** 🤔
# MAGIC
# MAGIC 2. **Desperdicio de datos**: Solo 80% se usa para entrenamiento
# MAGIC
# MAGIC 3. **Sobreajuste al test set**: Si evaluamos múltiples modelos con el mismo test set
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### ✅ Solución: Cross-Validation
# MAGIC
# MAGIC **Concepto**: Dividir datos en **K folds** (particiones) y entrenar K veces, usando cada fold como test una vez.
# MAGIC
# MAGIC **Ventajas**:
# MAGIC - ✅ **Estimación más robusta** del rendimiento
# MAGIC - ✅ **Usa todos los datos** para entrenamiento y evaluación
# MAGIC - ✅ **Reduce varianza** de la métrica
# MAGIC - ✅ **Detecta overfitting** más confiablemente
# MAGIC
# MAGIC **Ejemplo visual (5-Fold CV)**:
# MAGIC ```
# MAGIC Fold 1: [TEST] [TRAIN] [TRAIN] [TRAIN] [TRAIN]  → Score 1
# MAGIC Fold 2: [TRAIN] [TEST] [TRAIN] [TRAIN] [TRAIN]  → Score 2
# MAGIC Fold 3: [TRAIN] [TRAIN] [TEST] [TRAIN] [TRAIN]  → Score 3
# MAGIC Fold 4: [TRAIN] [TRAIN] [TRAIN] [TEST] [TRAIN]  → Score 4
# MAGIC Fold 5: [TRAIN] [TRAIN] [TRAIN] [TRAIN] [TEST]  → Score 5
# MAGIC
# MAGIC Score final = Mean(Score 1, ..., Score 5) ± Std
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 📊 ¿Cuándo Usar Cada Tipo?
# MAGIC
# MAGIC | Tipo de Datos | Estrategia CV Recomendada |
# MAGIC |---------------|---------------------------|
# MAGIC | 🎲 Datos aleatorios (i.i.d.) | K-Fold estándar |
# MAGIC | ⚖️ Datos desbalanceados | **Stratified K-Fold** |
# MAGIC | 📅 Series temporales | **Time Series CV** |
# MAGIC | 👥 Datos agrupados (clientes, usuarios) | **Group K-Fold** |
# MAGIC | 🗺️ Datos geoespaciales | **Spatial CV** (custom) |
# MAGIC | 🔬 Datasets muy pequeños | **Leave-One-Out CV** |
# MAGIC
# MAGIC ---

# COMMAND ----------

# DBTITLE 1,2. K-Fold Estándar (Revisión)
# MAGIC %md
# MAGIC ## 2️⃣ K-Fold Cross-Validation (Revisión)
# MAGIC
# MAGIC ### 📖 Concepto
# MAGIC
# MAGIC **K-Fold CV** divide el dataset en **K particiones (folds)** de tamaño similar y entrena K modelos.
# MAGIC
# MAGIC **Proceso**:
# MAGIC 1. Dividir datos en K folds (ej: K=5)
# MAGIC 2. Para cada fold i:
# MAGIC    - Usar fold i como **test set**
# MAGIC    - Usar folds restantes como **training set**
# MAGIC    - Entrenar modelo y calcular métrica
# MAGIC 3. Promediar las K métricas
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 💻 Implementación Básica
# MAGIC
# MAGIC ```python
# MAGIC from sklearn.model_selection import cross_val_score, KFold
# MAGIC from sklearn.ensemble import RandomForestRegressor
# MAGIC
# MAGIC # Crear estrategia K-Fold
# MAGIC kf = KFold(n_splits=5, shuffle=True, random_state=42)
# MAGIC
# MAGIC # Modelo
# MAGIC model = RandomForestRegressor(n_estimators=100, random_state=42)
# MAGIC
# MAGIC # Cross-validation
# MAGIC scores = cross_val_score(
# MAGIC     model, 
# MAGIC     X, 
# MAGIC     y, 
# MAGIC     cv=kf,  # Estrategia de validación
# MAGIC     scoring='neg_mean_absolute_error',  # Métrica
# MAGIC     n_jobs=-1  # Paralelizar
# MAGIC )
# MAGIC
# MAGIC # Resultados
# MAGIC print(f"MAE por fold: {-scores}")
# MAGIC print(f"MAE promedio: {-scores.mean():.2f} ± {scores.std():.2f}")
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### ⚙️ Parámetros Importantes
# MAGIC
# MAGIC **`n_splits`**: Número de folds (típicamente 5 o 10)
# MAGIC - Más folds = más tiempo de entrenamiento
# MAGIC - Menos folds = más varianza en estimación
# MAGIC - **Recomendado**: 5 o 10 folds
# MAGIC
# MAGIC **`shuffle`**: Mezclar datos antes de dividir
# MAGIC - `True`: Recomendado para datos i.i.d.
# MAGIC - `False`: Para series temporales (mantener orden)
# MAGIC
# MAGIC **`random_state`**: Semilla para reproducibilidad
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 📊 ¿Cuántos Folds Usar?
# MAGIC
# MAGIC | K | Train Size | Test Size | Tiempo | Varianza | Uso |
# MAGIC |---|------------|-----------|--------|----------|-----|
# MAGIC | 3 | 67% | 33% | Rápido | Alta | Experimentación |
# MAGIC | 5 | 80% | 20% | Medio | Media | **Estándar** |
# MAGIC | 10 | 90% | 10% | Lento | Baja | Evaluación final |
# MAGIC | N | N-1 | 1 | Muy lento | Muy baja | Datasets pequeños |
# MAGIC
# MAGIC ⚖️ **Trade-off**: Más folds = mejor estimación pero más tiempo
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### ⚠️ Limitaciones de K-Fold Estándar
# MAGIC
# MAGIC ❌ **No funciona bien con**:
# MAGIC 1. **Datos desbalanceados**: Algunos folds pueden no tener clases minoritarias
# MAGIC 2. **Series temporales**: Rompe dependencia temporal (entrena con futuro)
# MAGIC 3. **Datos agrupados**: Puede haber data leakage (mismo cliente en train y test)
# MAGIC 4. **Datos espaciales**: No considera autocorrelación espacial
# MAGIC
# MAGIC ✅ **Solución**: Usar variantes especializadas (próximas secciones)
# MAGIC
# MAGIC ---

# COMMAND ----------

# DBTITLE 1,3. Stratified K-Fold
# MAGIC %md
# MAGIC ## 3️⃣ Stratified K-Fold (Datos Desbalanceados)
# MAGIC
# MAGIC ### 🎯 Problema que Resuelve
# MAGIC
# MAGIC **Escenario**: Dataset de clasificación con clases **desbalanceadas**.
# MAGIC
# MAGIC ```
# MAGIC Dataset: 1000 registros
# MAGIC ├── Clase A: 900 registros (90%)
# MAGIC └── Clase B: 100 registros (10%)
# MAGIC ```
# MAGIC
# MAGIC ❌ **Con K-Fold estándar**:
# MAGIC - Algunos folds pueden tener **0% de clase B**
# MAGIC - Modelo no puede aprender clase minoritaria en esos folds
# MAGIC - Métricas sesgadas
# MAGIC
# MAGIC ✅ **Con Stratified K-Fold**:
# MAGIC - **Mantiene la proporción** de clases en cada fold
# MAGIC - Cada fold tiene ~90% clase A y ~10% clase B
# MAGIC - Evaluación más justa
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 📊 Visualización
# MAGIC
# MAGIC **K-Fold estándar** (distribución aleatoria):
# MAGIC ```
# MAGIC Fold 1: A=950, B=50   (95% A, 5% B)   ❌ Desbalanceado
# MAGIC Fold 2: A=850, B=150  (85% A, 15% B)  ❌ Desbalanceado
# MAGIC Fold 3: A=920, B=80   (92% A, 8% B)   ❌ Desbalanceado
# MAGIC ```
# MAGIC
# MAGIC **Stratified K-Fold** (proporción preservada):
# MAGIC ```
# MAGIC Fold 1: A=900, B=100  (90% A, 10% B)  ✅ Balanceado
# MAGIC Fold 2: A=900, B=100  (90% A, 10% B)  ✅ Balanceado
# MAGIC Fold 3: A=900, B=100  (90% A, 10% B)  ✅ Balanceado
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 💻 Implementación
# MAGIC
# MAGIC #### Clasificación (Estándar)
# MAGIC ```python
# MAGIC from sklearn.model_selection import StratifiedKFold, cross_val_score
# MAGIC from sklearn.ensemble import RandomForestClassifier
# MAGIC
# MAGIC # Crear estrategia Stratified K-Fold
# MAGIC skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# MAGIC
# MAGIC # Modelo
# MAGIC model = RandomForestClassifier(n_estimators=100, random_state=42)
# MAGIC
# MAGIC # Cross-validation
# MAGIC scores = cross_val_score(
# MAGIC     model, X, y, 
# MAGIC     cv=skf,  # Estrategia stratified
# MAGIC     scoring='f1_weighted'
# MAGIC )
# MAGIC
# MAGIC print(f"F1-Score: {scores.mean():.3f} ± {scores.std():.3f}")
# MAGIC ```
# MAGIC
# MAGIC #### Regresión (Binning Manual)
# MAGIC ```python
# MAGIC import numpy as np
# MAGIC from sklearn.model_selection import StratifiedKFold
# MAGIC
# MAGIC # Para regresión, crear bins del target
# MAGIC y_binned = pd.qcut(y, q=5, labels=False, duplicates='drop')
# MAGIC
# MAGIC skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# MAGIC
# MAGIC # Iterar manualmente
# MAGIC for train_idx, test_idx in skf.split(X, y_binned):
# MAGIC     X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
# MAGIC     y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
# MAGIC     
# MAGIC     model.fit(X_train, y_train)
# MAGIC     score = model.score(X_test, y_test)
# MAGIC     print(f"Score: {score:.3f}")
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 📈 Caso de Uso: Segmentación de Clientes
# MAGIC
# MAGIC **Dataset de Panadería**:
# MAGIC - Segmento Premium: 5% de clientes
# MAGIC - Segmento Regular: 70% de clientes
# MAGIC - Segmento Ocasional: 25% de clientes
# MAGIC
# MAGIC ✅ **Stratified K-Fold** asegura que cada fold tenga la misma distribución.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### ⚙️ Parámetros Importantes
# MAGIC
# MAGIC **Igual que K-Fold**, pero con estratificación automática:
# MAGIC - `n_splits`: Número de folds
# MAGIC - `shuffle`: Mezclar datos (recomendado `True`)
# MAGIC - `random_state`: Reproducibilidad
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 🎯 Cuándo Usar Stratified K-Fold
# MAGIC
# MAGIC ✅ **Usar cuando**:
# MAGIC - Clasificación con **clases desbalanceadas**
# MAGIC - Regresión con **distribución sesgada** del target
# MAGIC - Quieres **garantizar representatividad** en cada fold
# MAGIC
# MAGIC ❌ **No usar cuando**:
# MAGIC - Datos temporales (usar Time Series CV)
# MAGIC - Datos agrupados (usar Group K-Fold)
# MAGIC
# MAGIC ---

# COMMAND ----------

# DBTITLE 1,4. Time Series Cross-Validation
# MAGIC %md
# MAGIC ## 4️⃣ Time Series Cross-Validation
# MAGIC
# MAGIC ### 🚨 Problema con CV Estándar en Series Temporales
# MAGIC
# MAGIC ❌ **K-Fold estándar ROMPE la dependencia temporal**:
# MAGIC
# MAGIC ```
# MAGIC Datos: [Ene, Feb, Mar, Abr, May, Jun, Jul, Ago, Sep]
# MAGIC
# MAGIC K-Fold estándar (INCORRECTO):
# MAGIC Fold 1: Train=[Feb,Mar,May,Jun,Ago,Sep]  Test=[Ene,Abr,Jul]  ⛔ Entrena con futuro!
# MAGIC Fold 2: Train=[Ene,Mar,Abr,Jun,Jul,Sep]  Test=[Feb,May,Ago]  ⛔ Entrena con futuro!
# MAGIC ```
# MAGIC
# MAGIC **Problema**: El modelo **ve el futuro** durante entrenamiento → **data leakage**
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### ✅ Solución: Time Series Split (Forward Chaining)
# MAGIC
# MAGIC **Concepto**: Entrenar solo con **datos pasados**, nunca con datos futuros.
# MAGIC
# MAGIC **Visualización**:
# MAGIC ```
# MAGIC Fold 1: [TRAIN] [TEST] [ ] [ ] [ ]      → Predice paso 2
# MAGIC Fold 2: [TRAIN] [TRAIN] [TEST] [ ] [ ]  → Predice paso 3
# MAGIC Fold 3: [TRAIN] [TRAIN] [TRAIN] [TEST] [ ] → Predice paso 4
# MAGIC Fold 4: [TRAIN] [TRAIN] [TRAIN] [TRAIN] [TEST] → Predice paso 5
# MAGIC ```
# MAGIC
# MAGIC **Características**:
# MAGIC - ✅ Respeta **orden temporal**
# MAGIC - ✅ Training set **crece** con cada fold
# MAGIC - ✅ Test set siempre **después** de train
# MAGIC - ✅ Simula **producción real** (predecir futuro)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 💻 Implementación con TimeSeriesSplit
# MAGIC
# MAGIC ```python
# MAGIC from sklearn.model_selection import TimeSeriesSplit, cross_val_score
# MAGIC from sklearn.ensemble import RandomForestRegressor
# MAGIC
# MAGIC # Time Series CV con 5 splits
# MAGIC tscv = TimeSeriesSplit(n_splits=5)
# MAGIC
# MAGIC model = RandomForestRegressor(n_estimators=100, random_state=42)
# MAGIC
# MAGIC # Cross-validation
# MAGIC scores = cross_val_score(
# MAGIC     model, X, y, 
# MAGIC     cv=tscv,  # Time Series strategy
# MAGIC     scoring='neg_mean_absolute_error'
# MAGIC )
# MAGIC
# MAGIC print(f"MAE por fold: {-scores}")
# MAGIC print(f"MAE promedio: {-scores.mean():.2f} ± {scores.std():.2f}")
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### ⚙️ Variantes de Time Series CV
# MAGIC
# MAGIC #### 1. **Expanding Window** (Ventana Creciente) - Default
# MAGIC
# MAGIC ```
# MAGIC Fold 1: [█] [T]
# MAGIC Fold 2: [█ █] [T]
# MAGIC Fold 3: [█ █ █] [T]
# MAGIC Fold 4: [█ █ █ █] [T]
# MAGIC ```
# MAGIC
# MAGIC ✅ Usa **todos los datos históricos**  
# MAGIC ⚠️ Puede ser lento con datasets grandes
# MAGIC
# MAGIC #### 2. **Rolling Window** (Ventana Deslizante)
# MAGIC
# MAGIC ```
# MAGIC Fold 1: [█ █] [T] [ ]
# MAGIC Fold 2: [ ] [█ █] [T]
# MAGIC Fold 3: [ ] [ ] [█ █] [T]
# MAGIC ```
# MAGIC
# MAGIC ✅ **Tamaño fijo** de entrenamiento  
# MAGIC ✅ Más rápido  
# MAGIC ⚠️ Descarta datos antiguos
# MAGIC
# MAGIC **Implementación manual**:
# MAGIC ```python
# MAGIC from sklearn.model_selection import TimeSeriesSplit
# MAGIC
# MAGIC # Rolling window custom
# MAGIC window_size = 1000  # Últimos 1000 registros
# MAGIC
# MAGIC for train_idx, test_idx in TimeSeriesSplit(n_splits=5).split(X):
# MAGIC     # Limitar train a últimos `window_size` registros
# MAGIC     train_idx = train_idx[-window_size:]
# MAGIC     
# MAGIC     X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
# MAGIC     y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
# MAGIC     
# MAGIC     model.fit(X_train, y_train)
# MAGIC     score = model.score(X_test, y_test)
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 📊 Ejemplo: Ventas de Panadería por Día
# MAGIC
# MAGIC **Dataset**: Ventas diarias de enero a diciembre (365 días)
# MAGIC
# MAGIC ```python
# MAGIC # Ordenar por fecha (IMPORTANTE)
# MAGIC df = df.sort_values('fecha').reset_index(drop=True)
# MAGIC
# MAGIC # Preparar features y target
# MAGIC X = df[['dia_semana', 'mes', 'es_feriado', 'temp_celsius']]
# MAGIC y = df['ventas_totales']
# MAGIC
# MAGIC # Time Series CV
# MAGIC tscv = TimeSeriesSplit(n_splits=6)
# MAGIC
# MAGIC for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
# MAGIC     print(f"Fold {fold}:")
# MAGIC     print(f"  Train: días {train_idx[0]} a {train_idx[-1]}")
# MAGIC     print(f"  Test:  días {test_idx[0]} a {test_idx[-1]}")
# MAGIC ```
# MAGIC
# MAGIC **Output esperado**:
# MAGIC ```
# MAGIC Fold 1:
# MAGIC   Train: días 0 a 59    (Ene-Feb)
# MAGIC   Test:  días 60 a 119  (Mar-Abr)
# MAGIC   
# MAGIC Fold 2:
# MAGIC   Train: días 0 a 119   (Ene-Abr)
# MAGIC   Test:  días 120 a 179 (May-Jun)
# MAGIC ...
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 🎯 Cuándo Usar Time Series CV
# MAGIC
# MAGIC ✅ **Siempre que tengas dependencia temporal**:
# MAGIC - Predicción de ventas
# MAGIC - Forecasting de demanda
# MAGIC - Predicción de precios
# MAGIC - Series temporales en general
# MAGIC
# MAGIC ❌ **No usar cuando**:
# MAGIC - Datos NO tienen orden temporal
# MAGIC - Quieres aprovechar datos futuros (no es realista)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### ⚠️ Errores Comunes
# MAGIC
# MAGIC 1. **No ordenar datos por fecha** antes de split
# MAGIC 2. **Usar K-Fold estándar** en series temporales
# MAGIC 3. **Incluir features con "fuga del futuro"** (ej: precio_futuro)
# MAGIC 4. **No considerar estacionalidad** en los folds
# MAGIC
# MAGIC ---