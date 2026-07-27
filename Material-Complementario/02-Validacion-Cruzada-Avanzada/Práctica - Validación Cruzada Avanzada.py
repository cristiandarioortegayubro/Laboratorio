# Databricks notebook source
# DBTITLE 1,Título
# MAGIC %md
# MAGIC # 🎯 Práctica: Validación Cruzada Avanzada
# MAGIC ## Material Complementario - Laboratorio (Herramientas)
# MAGIC ### Universidad del Aconcagua - Mendoza, Argentina
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 📝 Objetivos
# MAGIC
# MAGIC En este notebook práctico, aplicarás:
# MAGIC
# MAGIC 1. ✅ **Stratified K-Fold** en clasificación de segmentos de clientes
# MAGIC 2. ✅ **Time Series CV** para forecasting de ventas
# MAGIC 3. ✅ **Group K-Fold** para evitar leakage por cliente
# MAGIC 4. ✅ **Validación espacial** con features H3
# MAGIC 5. ✅ **Comparación** de todas las estrategias
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 📊 Ejercicios
# MAGIC
# MAGIC | Ejercicio | Descripción | Estrategia CV | Dificultad |
# MAGIC |-----------|-------------|---------------|------------|
# MAGIC | **1** | Clasificación de segmentos (desbalanceados) | Stratified K-Fold | 🟢 Fácil |
# MAGIC | **2** | Forecasting de ventas diarias | Time Series CV | 🟡 Medio |
# MAGIC | **3** | Predicción por cliente (evitar leakage) | Group K-Fold | 🟡 Medio |
# MAGIC | **4** | Predicción geoespacial por zona H3 | Spatial CV | 🔴 Difícil |
# MAGIC | **5** | Comparación completa de estrategias | Todas | 🔴 Difícil |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### ⏱️ Duración Estimada: 2-3 horas
# MAGIC
# MAGIC ---

# COMMAND ----------

# DBTITLE 1,Setup
# Importar librerías
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import h3

from sklearn.model_selection import (
    cross_val_score, cross_validate,
    KFold, StratifiedKFold, TimeSeriesSplit, GroupKFold
)
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error, accuracy_score, f1_score, 
    classification_report, confusion_matrix
)

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
df_ventas['dia_mes'] = df_ventas['fecha'].dt.day
df_ventas['semana_anio'] = df_ventas['fecha'].dt.isocalendar().week

# Merge con clientes
df = df_ventas.merge(df_clientes, on='cliente_id', how='left')
df['segmento_encoded'] = df['segmento'].astype('category').cat.codes

print("\n✅ Features temporales creadas")
print(f"\nColumnas disponibles: {list(df.columns)}")

# COMMAND ----------

# DBTITLE 1,Ejercicio 1: Stratified K-Fold (Clasificación)
# MAGIC %md
# MAGIC ## 🎯 Ejercicio 1: Stratified K-Fold - Clasificación de Segmentos
# MAGIC
# MAGIC ### 📝 Problema
# MAGIC
# MAGIC **Objetivo**: Predecir el **segmento de cliente** (Premium, Regular, Ocasional) basándose en su comportamiento de compra.
# MAGIC
# MAGIC **Datos**:
# MAGIC - Features: total de compra, día de semana, mes, sucursal
# MAGIC - Target: segmento (clasificación)
# MAGIC - **Desbalance**: ~5% Premium, ~70% Regular, ~25% Ocasional
# MAGIC
# MAGIC ### ❓ Pregunta
# MAGIC
# MAGIC ¿Por qué usar **Stratified K-Fold** en lugar de K-Fold estándar?
# MAGIC
# MAGIC **Respuesta**: Porque el dataset tiene **clases desbalanceadas**. Stratified K-Fold **preserva la proporción** de cada clase en cada fold, asegurando que todos los folds tengan ejemplos de la clase minoritaria (Premium).
# MAGIC
# MAGIC ---

# COMMAND ----------

# DBTITLE 1,Ejercicio 1: Implementación
print("="*80)
print("EJERCICIO 1: STRATIFIED K-FOLD - CLASIFICACIÓN DE SEGMENTOS")
print("="*80)

# Preparar datos para clasificación
df_clf = df[df['segmento'].notna()].copy()

# Features
features_clf = ['total', 'dia_semana', 'mes', 'sucursal_id']
X_clf = df_clf[features_clf]
y_clf = df_clf['segmento']

print(f"\n📊 Dataset: {len(X_clf):,} registros")
print(f"\nDistribución de clases:")
print(y_clf.value_counts(normalize=True).apply(lambda x: f"{x*100:.1f}%"))

# Comparar K-Fold vs. Stratified K-Fold
model_clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)

print("\n\n🔄 Evaluando con K-Fold estándar...")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores_kf = cross_val_score(model_clf, X_clf, y_clf, cv=kf, scoring='f1_weighted', n_jobs=-1)

print("🔄 Evaluando con Stratified K-Fold...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores_skf = cross_val_score(model_clf, X_clf, y_clf, cv=skf, scoring='f1_weighted', n_jobs=-1)

# Resultados
print("\n" + "="*80)
print("RESULTADOS")
print("="*80)

print(f"\nK-Fold estándar:")
print(f"   F1-Score: {scores_kf.mean():.4f} ± {scores_kf.std():.4f}")
print(f"   Scores por fold: {[f'{s:.4f}' for s in scores_kf]}")

print(f"\nStratified K-Fold:")
print(f"   F1-Score: {scores_skf.mean():.4f} ± {scores_skf.std():.4f}")
print(f"   Scores por fold: {[f'{s:.4f}' for s in scores_skf]}")

# Comparación
mejora = scores_skf.mean() - scores_kf.mean()
print(f"\n💡 Mejora con Stratified: {mejora:+.4f} ({mejora/scores_kf.mean()*100:+.2f}%)")
print(f"\n✅ Stratified K-Fold tiene **menor varianza** entre folds ({scores_skf.std():.4f} vs. {scores_kf.std():.4f})")
print(f"✅ Garantiza que cada fold tenga todas las clases representadas")

# COMMAND ----------

# DBTITLE 1,Ejercicio 2: Time Series CV (Forecasting)
# MAGIC %md
# MAGIC ## 📅 Ejercicio 2: Time Series CV - Forecasting de Ventas
# MAGIC
# MAGIC ### 📝 Problema
# MAGIC
# MAGIC **Objetivo**: Predecir las **ventas totales diarias** de la panadería para los próximos días.
# MAGIC
# MAGIC **Datos**:
# MAGIC - Features: día de semana, mes, ventas promedio de últimos 7 días
# MAGIC - Target: ventas totales del día
# MAGIC - **Dependencia temporal**: Ventas de hoy dependen de ventas pasadas
# MAGIC
# MAGIC ### ❓ Pregunta
# MAGIC
# MAGIC ¿Por qué NO usar K-Fold estándar para series temporales?
# MAGIC
# MAGIC **Respuesta**: K-Fold estándar **mezcla datos** y puede entrenar con datos **futuros**, causando **data leakage**. En producción, solo conocemos el pasado, no el futuro.
# MAGIC
# MAGIC **Time Series CV** respeta el orden temporal: siempre entrena con datos pasados y evalúa en datos futuros.
# MAGIC
# MAGIC ---

# COMMAND ----------

# DBTITLE 1,Ejercicio 2: Implementación
print("="*80)
print("EJERCICIO 2: TIME SERIES CV - FORECASTING DE VENTAS")
print("="*80)

# Agregar ventas por día
df_diario = df.groupby('fecha').agg({
    'total': 'sum',
    'cliente_id': 'count'
}).rename(columns={'cliente_id': 'num_ventas'}).reset_index()

# Features temporales
df_diario['dia_semana'] = df_diario['fecha'].dt.dayofweek
df_diario['mes'] = df_diario['fecha'].dt.month
df_diario['dia_mes'] = df_diario['fecha'].dt.day

# Feature: Ventas promedio de últimos 7 días (lag)
df_diario['ventas_lag_7d'] = df_diario['total'].rolling(window=7, min_periods=1).mean().shift(1)

# Eliminar primer registro (sin lag)
df_diario = df_diario[1:].reset_index(drop=True)

features_ts = ['dia_semana', 'mes', 'dia_mes', 'ventas_lag_7d']
X_ts = df_diario[features_ts]
y_ts = df_diario['total']

print(f"\n📊 Dataset: {len(X_ts)} días")
print(f"Rango de fechas: {df_diario['fecha'].min()} a {df_diario['fecha'].max()}")

# Modelo
model_ts = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)

# Comparar K-Fold vs. Time Series CV
print("\n🔄 Evaluando con K-Fold estándar (INCORRECTO)...")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores_kf_ts = cross_val_score(model_ts, X_ts, y_ts, cv=kf, scoring='neg_mean_absolute_error', n_jobs=-1)

print("🔄 Evaluando con Time Series CV (CORRECTO)...")
tscv = TimeSeriesSplit(n_splits=5)
scores_tscv = cross_val_score(model_ts, X_ts, y_ts, cv=tscv, scoring='neg_mean_absolute_error', n_jobs=-1)

# Resultados
print("\n" + "="*80)
print("RESULTADOS")
print("="*80)

print(f"\nK-Fold estándar (INCORRECTO):")
print(f"   MAE: ${-scores_kf_ts.mean():.2f} ± ${scores_kf_ts.std():.2f}")
print(f"   ❌ Data leakage: entrena con datos futuros")

print(f"\nTime Series CV (CORRECTO):")
print(f"   MAE: ${-scores_tscv.mean():.2f} ± ${scores_tscv.std():.2f}")
print(f"   ✅ Realista: solo usa datos pasados")

diff = (-scores_tscv.mean()) - (-scores_kf_ts.mean())
print(f"\n💡 Diferencia: ${diff:.2f} ({diff/-scores_kf_ts.mean()*100:.1f}% peor)")
print(f"\n⚠️ K-Fold subestima el error porque 've el futuro' durante entrenamiento.")
print(f"✅ Time Series CV es la estimación REALISTA para forecasting.")

# COMMAND ----------

# DBTITLE 1,Ejercicio 3: Group K-Fold
# MAGIC %md
# MAGIC ## 👥 Ejercicio 3: Group K-Fold - Evitar Data Leakage por Cliente
# MAGIC
# MAGIC ### 📝 Problema
# MAGIC
# MAGIC **Objetivo**: Predecir el **total de compra** de transacciones futuras, considerando que:
# MAGIC - Cada cliente tiene **múltiples compras**
# MAGIC - Queremos predecir compras de **clientes NUEVOS** (nunca vistos)
# MAGIC
# MAGIC **Datos**:
# MAGIC - Dataset: 50,000 ventas de 500 clientes
# MAGIC - Promedio: 100 compras por cliente
# MAGIC
# MAGIC ### ❓ Pregunta
# MAGIC
# MAGIC ¿Qué pasa si usamos K-Fold estándar?
# MAGIC
# MAGIC **Respuesta**: K-Fold puede poner **compras del mismo cliente** en train y test → **data leakage**. El modelo "conoce" al cliente en train, por lo que su rendimiento en test será **artificialmente alto**.
# MAGIC
# MAGIC **Group K-Fold** asegura que **todas las compras de un cliente** estén en el mismo fold (train O test, nunca ambos).
# MAGIC
# MAGIC ---

# COMMAND ----------

# DBTITLE 1,Ejercicio 3: Implementación
print("="*80)
print("EJERCICIO 3: GROUP K-FOLD - PREDICCIÓN POR CLIENTE")
print("="*80)

# Preparar datos
df_group = df[df['cliente_id'].notna()].copy()

# Features
features_group = ['sucursal_id', 'dia_semana', 'mes', 'segmento_encoded']
X_group = df_group[features_group]
y_group = df_group['total']
groups = df_group['cliente_id'].values

print(f"\n📊 Dataset: {len(X_group):,} ventas")
print(f"Número de clientes: {len(np.unique(groups))}")
print(f"Promedio ventas por cliente: {len(X_group) / len(np.unique(groups)):.1f}")

# Modelo
model_group = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)

# Comparar K-Fold vs. Group K-Fold
print("\n🔄 Evaluando con K-Fold estándar (con leakage)...")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores_kf_group = cross_val_score(model_group, X_group, y_group, cv=kf, scoring='neg_mean_absolute_error', n_jobs=-1)

print("🔄 Evaluando con Group K-Fold (sin leakage)...")
gkf = GroupKFold(n_splits=5)
scores_gkf = cross_val_score(model_group, X_group, y_group, groups=groups, cv=gkf, scoring='neg_mean_absolute_error', n_jobs=-1)

# Resultados
print("\n" + "="*80)
print("RESULTADOS")
print("="*80)

print(f"\nK-Fold estándar:")
print(f"   MAE: ${-scores_kf_group.mean():.2f} ± ${scores_kf_group.std():.2f}")
print(f"   ❌ Optimista (data leakage: mismo cliente en train y test)")

print(f"\nGroup K-Fold:")
print(f"   MAE: ${-scores_gkf.mean():.2f} ± ${scores_gkf.std():.2f}")
print(f"   ✅ Realista (clientes NUEVOS nunca vistos)")

diff = (-scores_gkf.mean()) - (-scores_kf_group.mean())
print(f"\n💡 Diferencia: ${diff:.2f} ({diff/-scores_kf_group.mean()*100:.1f}% peor)")
print(f"\n⚠️ K-Fold subestima el error porque el modelo 've' al cliente en train.")
print(f"✅ Group K-Fold es la estimación REALISTA para clientes nuevos.")
print(f"\n💡 Si tu objetivo es predecir compras de clientes NUEVOS, usa Group K-Fold!")

# COMMAND ----------

# DBTITLE 1,Ejercicio 4: Validación Espacial
# MAGIC %md
# MAGIC ## 🗺️ Ejercicio 4: Validación Espacial con H3
# MAGIC
# MAGIC ### 📝 Problema
# MAGIC
# MAGIC **Objetivo**: Predecir **ventas por zona geográfica** (hexágonos H3) y evaluar capacidad de generalizar a **zonas nuevas**.
# MAGIC
# MAGIC **Datos**:
# MAGIC - Clientes tienen coordenadas geográficas
# MAGIC - Features H3 por zona
# MAGIC - **Autocorrelación espacial**: Zonas cercanas tienen ventas similares
# MAGIC
# MAGIC ### ❓ Pregunta
# MAGIC
# MAGIC ¿Por qué usar validación espacial en lugar de K-Fold?
# MAGIC
# MAGIC **Respuesta**: K-Fold puede entrenar con zonas **cercanas** a las zonas de test → el modelo aprende patrones locales. En producción, puede predecir en **zonas lejanas/nuevas** donde estos patrones no aplican.
# MAGIC
# MAGIC **Spatial CV** (Group K-Fold por zona H3) asegura que train y test estén **espacialmente separados**.
# MAGIC
# MAGIC ---

# COMMAND ----------

# DBTITLE 1,Ejercicio 4: Implementación
print("="*80)
print("EJERCICIO 4: VALIDACIÓN ESPACIAL CON H3")
print("="*80)

# Agregar zona H3 (resolución 7)
df_spatial = df[df['h3_index'].notna()].copy()

# Features (incluyendo H3)
features_spatial = ['dia_semana', 'mes', 'segmento_encoded']
X_spatial = df_spatial[features_spatial]
y_spatial = df_spatial['total']
groups_h3 = df_spatial['h3_index'].values

print(f"\n📊 Dataset: {len(X_spatial):,} ventas")
print(f"Número de zonas H3: {len(np.unique(groups_h3))}")
print(f"Promedio ventas por zona: {len(X_spatial) / len(np.unique(groups_h3)):.1f}")

# Modelo
model_spatial = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)

# Comparar K-Fold vs. Spatial CV
print("\n🔄 Evaluando con K-Fold estándar (puede tener autocorrelación)...")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores_kf_spatial = cross_val_score(model_spatial, X_spatial, y_spatial, cv=kf, scoring='neg_mean_absolute_error', n_jobs=-1)

print("🔄 Evaluando con Spatial CV (Group K-Fold por H3)...")
gkf_h3 = GroupKFold(n_splits=5)
scores_spatial = cross_val_score(model_spatial, X_spatial, y_spatial, groups=groups_h3, cv=gkf_h3, scoring='neg_mean_absolute_error', n_jobs=-1)

# Resultados
print("\n" + "="*80)
print("RESULTADOS")
print("="*80)

print(f"\nK-Fold estándar:")
print(f"   MAE: ${-scores_kf_spatial.mean():.2f} ± ${scores_kf_spatial.std():.2f}")
print(f"   ❌ Puede ser optimista (autocorrelación espacial)")

print(f"\nSpatial CV (por zona H3):")
print(f"   MAE: ${-scores_spatial.mean():.2f} ± ${scores_spatial.std():.2f}")
print(f"   ✅ Realista (zonas NUEVAS separadas espacialmente)")

diff = (-scores_spatial.mean()) - (-scores_kf_spatial.mean())
print(f"\n💡 Diferencia: ${diff:.2f} ({diff/-scores_kf_spatial.mean()*100:.1f}% peor)")
print(f"\n✅ Spatial CV revela el rendimiento REAL en zonas nuevas/lejanas.")
print(f"🗺️ Útil para: expansión geográfica, nuevas sucursales, zonas sin historial.")

# COMMAND ----------

# DBTITLE 1,Ejercicio 5: Comparación Completa
# MAGIC %md
# MAGIC ## 📊 Ejercicio 5: Comparación Completa de Estrategias
# MAGIC
# MAGIC ### 📝 Problema
# MAGIC
# MAGIC **Objetivo**: Comparar **todas las estrategias** en el mismo dataset y entender cuándo usar cada una.
# MAGIC
# MAGIC Vamos a evaluar:
# MAGIC 1. K-Fold estándar
# MAGIC 2. Stratified K-Fold (para clasificación)
# MAGIC 3. Time Series CV
# MAGIC 4. Group K-Fold
# MAGIC 5. Spatial CV
# MAGIC
# MAGIC Y responder: **¿Cuál estrategia elegirías para tu problema?**
# MAGIC
# MAGIC ---

# COMMAND ----------

# DBTITLE 1,Ejercicio 5: Comparación
print("="*80)
print("EJERCICIO 5: COMPARACIÓN COMPLETA DE ESTRATEGIAS")
print("="*80)

# Usar dataset con todos los datos
df_comp = df[df['cliente_id'].notna() & df['h3_index'].notna()].copy()

features_comp = ['sucursal_id', 'dia_semana', 'mes', 'segmento_encoded']
X_comp = df_comp[features_comp]
y_comp = df_comp['total']

model_comp = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)

print(f"\n📊 Dataset: {len(X_comp):,} registros")
print(f"\nProbando todas las estrategias...\n")

# 1. K-Fold estándar
scores_1 = cross_val_score(model_comp, X_comp, y_comp, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)

# 2. Time Series CV
scores_2 = cross_val_score(model_comp, X_comp, y_comp, cv=TimeSeriesSplit(n_splits=5), scoring='neg_mean_absolute_error', n_jobs=-1)

# 3. Group K-Fold (por cliente)
groups_cliente = df_comp['cliente_id'].values
scores_3 = cross_val_score(model_comp, X_comp, y_comp, groups=groups_cliente, cv=GroupKFold(n_splits=5), scoring='neg_mean_absolute_error', n_jobs=-1)

# 4. Spatial CV (por H3)
groups_h3_comp = df_comp['h3_index'].values
scores_4 = cross_val_score(model_comp, X_comp, y_comp, groups=groups_h3_comp, cv=GroupKFold(n_splits=5), scoring='neg_mean_absolute_error', n_jobs=-1)

# Resultados
print("="*80)
print("TABLA COMPARATIVA")
print("="*80)

resultados = pd.DataFrame({
    'Estrategia': ['K-Fold', 'Time Series CV', 'Group K-Fold (Cliente)', 'Spatial CV (H3)'],
    'MAE': [-scores_1.mean(), -scores_2.mean(), -scores_3.mean(), -scores_4.mean()],
    'Std': [scores_1.std(), scores_2.std(), scores_3.std(), scores_4.std()],
    'Cuándo Usar': [
        'Datos aleatorios i.i.d.',
        'Series temporales / Forecasting',
        'Clientes nuevos / Grupos',
        'Zonas nuevas / Geoespacial'
    ]
})

resultados['MAE_str'] = resultados['MAE'].apply(lambda x: f'${x:.2f}')
resultados['Std_str'] = resultados['Std'].apply(lambda x: f'${x:.2f}')
resultados = resultados.sort_values('MAE')

print("\n")
print(resultados[['Estrategia', 'MAE_str', 'Std_str', 'Cuándo Usar']].to_string(index=False))

print("\n" + "="*80)
print("CONCLUSIONES")
print("="*80)

print(f"\n🏆 K-Fold tiene el MAE más bajo (${-scores_1.mean():.2f})")
print(f"   ⚠️ Pero puede ser OPTIMISTA (no considera estructura de datos)")

print(f"\n🔴 Spatial CV tiene el MAE más alto (${-scores_4.mean():.2f})")
print(f"   ✅ Pero es el más REALISTA para zonas nuevas")

print(f"\n💡 LA ESTRATEGIA CORRECTA DEPENDE DE TU OBJETIVO:")
print(f"   - Predecir futuro? → Time Series CV")
print(f"   - Clientes nuevos? → Group K-Fold")
print(f"   - Zonas nuevas? → Spatial CV")
print(f"   - Datos aleatorios? → K-Fold")

print(f"\n⚠️ NO uses la estrategia con MAE más bajo si no coincide con tu problema real.")

# COMMAND ----------

# DBTITLE 1,Conclusiones finales
# MAGIC %md
# MAGIC ## ✅ Conclusiones del Notebook Práctico
# MAGIC
# MAGIC ### 🎯 Resumen de Ejercicios
# MAGIC
# MAGIC | Ejercicio | Estrategia | Aprendizaje Clave |
# MAGIC |-----------|------------|-------------------|
# MAGIC | **1** | Stratified K-Fold | Preserva distribución de clases desbalanceadas |
# MAGIC | **2** | Time Series CV | Respeta orden temporal, evita ver el futuro |
# MAGIC | **3** | Group K-Fold | Evita leakage por agrupamiento (clientes) |
# MAGIC | **4** | Spatial CV | Evalúa generalización a zonas nuevas |
# MAGIC | **5** | Comparación | La estrategia correcta depende del problema |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 💡 Mensajes Clave
# MAGIC
# MAGIC 1. ❌ **K-Fold NO es siempre la mejor opción**
# MAGIC    - Puede dar resultados **optimistas** si tus datos tienen estructura especial
# MAGIC
# MAGIC 2. ✅ **Selecciona la estrategia según tipo de datos**:
# MAGIC    - 📅 Temporal → Time Series CV
# MAGIC    - 👥 Agrupados → Group K-Fold
# MAGIC    - 🗺️ Geoespacial → Spatial CV
# MAGIC    - ⚖️ Desbalanceados → Stratified K-Fold
# MAGIC    - 🎲 Aleatorios → K-Fold estándar
# MAGIC
# MAGIC 3. 🔑 **El MAE más bajo NO significa mejor estrategia**
# MAGIC    - Una estrategia con MAE alto pero **realista** es mejor que una con MAE bajo pero **optimista**
# MAGIC
# MAGIC 4. 🎯 **Piensa en producción**:
# MAGIC    - ¿Predecirás el futuro? → Time Series CV
# MAGIC    - ¿Clientes nuevos? → Group K-Fold
# MAGIC    - ¿Zonas nuevas? → Spatial CV
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 📚 Próximos Pasos
# MAGIC
# MAGIC Ahora que dominas validación cruzada avanzada:
# MAGIC
# MAGIC 1. ✅ Aplica estas técnicas en tus **proyectos del curso** (TP04-TP08)
# MAGIC 2. ✅ Usa la estrategia correcta en **trabajos finales**
# MAGIC 3. ✅ Combina con **optimización de hiperparámetros** (Módulo 01)
# MAGIC 4. ✅ Continuar con **Módulo 03: Selección de Características**
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🎓 ¡Felicitaciones!
# MAGIC
# MAGIC **Has completado el Módulo 02: Validación Cruzada Avanzada.**
# MAGIC
# MAGIC Ahora puedes:
# MAGIC - ✅ Implementar 6 estrategias diferentes de CV
# MAGIC - ✅ Seleccionar la estrategia correcta según tus datos
# MAGIC - ✅ Evitar data leakage en evaluación
# MAGIC - ✅ Obtener estimaciones realistas de rendimiento
# MAGIC - ✅ Validar modelos espacialmente con H3
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC **Universidad del Aconcagua**  
# MAGIC **Laboratorio (Herramientas)**  
# MAGIC **Mendoza, Argentina**

# COMMAND ----------

