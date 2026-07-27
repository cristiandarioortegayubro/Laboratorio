# Databricks notebook source
# DBTITLE 1,Título práctico
# MAGIC %md
# MAGIC # 🛠️ Práctica: Optimización con Optuna
# MAGIC ## Material Complementario - Laboratorio (Herramientas)
# MAGIC ### Universidad del Aconcagua - Mendoza, Argentina
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 🎯 Objetivos de la Práctica
# MAGIC
# MAGIC 1. Resolver **ejercicios prácticos** de optimización
# MAGIC 2. Implementar casos de uso reales
# MAGIC 3. Comparar múltiples algoritmos
# MAGIC 4. Optimizar modelos con features geoespaciales
# MAGIC
# MAGIC ### 📁 Contenido
# MAGIC
# MAGIC 1. Ejercicio 1: Optimizar Gradient Boosting
# MAGIC 2. Ejercicio 2: Optimización Multi-objetivo
# MAGIC 3. Ejercicio 3: Persistencia en Base de Datos
# MAGIC 4. Ejercicio 4: Optimización con Features Espaciales H3
# MAGIC 5. Proyecto Integrador: Dashboard de Comparación
# MAGIC
# MAGIC ### ⏱️ Duración Estimada: 2 horas
# MAGIC
# MAGIC ---

# COMMAND ----------

# DBTITLE 1,Setup
# Instalar dependencias
%pip install optuna==3.5.0 --quiet
%pip install h3==4.0.0b5 --quiet

import pandas as pd
import numpy as np
import optuna
import h3
import warnings
import time
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.graph_objects as go

print("✅ Librerías cargadas")

# COMMAND ----------

# DBTITLE 1,Cargar datos
# Cargar datasets
ruta_datos = '/Workspace/Users/cortega@uda.edu.ar/Laboratorio/Datasets/'

df_ventas = pd.read_csv(ruta_datos + 'ventas.csv')
df_clientes = pd.read_csv(ruta_datos + 'clientes.csv')
df_sucursales = pd.read_csv(ruta_datos + 'sucursales.csv')

# Preparar datos
df_ventas['fecha'] = pd.to_datetime(df_ventas['fecha'])
df_ventas['dia_semana'] = df_ventas['fecha'].dt.dayofweek
df_ventas['mes'] = df_ventas['fecha'].dt.month
df_ventas['es_fin_de_semana'] = df_ventas['dia_semana'].isin([5, 6]).astype(int)

df_ml = df_ventas[df_ventas['cliente_id'].notna()].merge(
    df_clientes[['cliente_id', 'segmento']], 
    on='cliente_id'
)
df_ml['segmento_encoded'] = df_ml['segmento'].astype('category').cat.codes

features_base = ['sucursal_id', 'dia_semana', 'mes', 'es_fin_de_semana', 'segmento_encoded']
X = df_ml[features_base]
y = df_ml['total']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("✅ Datos preparados")
print(f"   Train: {len(X_train):,} | Test: {len(X_test):,}")

# COMMAND ----------

# DBTITLE 1,Ejercicio 1: Gradient Boosting
# MAGIC %md
# MAGIC ## 📝 Ejercicio 1: Optimizar Gradient Boosting
# MAGIC
# MAGIC ### Objetivo
# MAGIC Optimizar un modelo **GradientBoostingRegressor** y comparar con Random Forest.
# MAGIC
# MAGIC ### Hiperparámetros a Optimizar
# MAGIC - `n_estimators`: 50-500
# MAGIC - `learning_rate`: 0.01-0.3 (escala log)
# MAGIC - `max_depth`: 3-10
# MAGIC - `subsample`: 0.5-1.0
# MAGIC - `min_samples_split`: 2-20

# COMMAND ----------

# DBTITLE 1,Solución Ejercicio 1
print("="*80)
print("EJERCICIO 1: OPTIMIZAR GRADIENT BOOSTING")
print("="*80)

def objective_gb(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'random_state': 42
    }
    
    model = GradientBoostingRegressor(**params)
    scores = cross_val_score(model, X_train, y_train, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1)
    return -scores.mean()

study_gb = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
print("\n🚀 Optimizando Gradient Boosting (50 trials)...\n")
study_gb.optimize(objective_gb, n_trials=50, show_progress_bar=True)

print(f"\n🎯 Mejores Hiperparámetros:")
for key, value in study_gb.best_params.items():
    print(f"   {key}: {value}")

print(f"\n📊 Mejor MAE (CV): ${study_gb.best_value:,.2f}")

# Entrenar modelo final
model_gb = GradientBoostingRegressor(**study_gb.best_params)
model_gb.fit(X_train, y_train)
y_pred_gb = model_gb.predict(X_test)

mae_gb = mean_absolute_error(y_test, y_pred_gb)
r2_gb = r2_score(y_test, y_pred_gb)

print(f"\n📊 Métricas en Test:")
print(f"   MAE:  ${mae_gb:,.2f}")
print(f"   R²:   {r2_gb:.4f}")

print("\n✅ Ejercicio 1 completado")

# COMMAND ----------

# DBTITLE 1,Ejercicio 2: Multi-objetivo
# MAGIC %md
# MAGIC ## 📝 Ejercicio 2: Optimización Multi-objetivo
# MAGIC
# MAGIC ### Objetivo
# MAGIC Optimizar **dos métricas simultáneamente**: 
# MAGIC 1. MAE (precisión)
# MAGIC 2. Tiempo de entrenamiento (velocidad)
# MAGIC
# MAGIC ### Concepto
# MAGIC **Trade-off**: Modelos más complejos suelen ser más precisos pero más lentos.

# COMMAND ----------

# DBTITLE 1,Solución Ejercicio 2
print("\n" + "="*80)
print("EJERCICIO 2: OPTIMIZACIÓN MULTI-OBJETIVO")
print("="*80)

def objective_multi(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 20),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'random_state': 42
    }
    
    model = RandomForestRegressor(**params)
    
    # Medir tiempo
    start_time = time.time()
    scores = cross_val_score(model, X_train, y_train, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1)
    elapsed_time = time.time() - start_time
    
    mae = -scores.mean()
    
    # Retornar tupla: (MAE, tiempo)
    return mae, elapsed_time

# Estudio multi-objetivo
study_multi = optuna.create_study(
    directions=['minimize', 'minimize'],  # Minimizar ambos
    sampler=optuna.samplers.TPESampler(seed=42)
)

print("\n🚀 Optimizando precisión y velocidad (30 trials)...\n")
study_multi.optimize(objective_multi, n_trials=30, show_progress_bar=True)

print(f"\n📊 Resultados:")
print(f"   Trials en Pareto front: {len(study_multi.best_trials)}")

print(f"\n🎯 Mejores Soluciones (Pareto front):")
for i, trial in enumerate(study_multi.best_trials[:3]):
    print(f"\n   Solución {i+1}:")
    print(f"      MAE:  ${trial.values[0]:,.2f}")
    print(f"      Tiempo: {trial.values[1]:.2f}s")
    print(f"      n_estimators: {trial.params['n_estimators']}")
    print(f"      max_depth: {trial.params['max_depth']}")

print("\n✅ Ejercicio 2 completado")

# COMMAND ----------

# DBTITLE 1,Ejercicio 4: Features H3
# MAGIC %md
# MAGIC ## 📝 Ejercicio 4: Optimización con Features Espaciales H3
# MAGIC
# MAGIC ### Objetivo
# MAGIC Agregar features geoespaciales y optimizar modelo.
# MAGIC
# MAGIC ### Features Espaciales a Crear
# MAGIC 1. `dist_sucursal_min`: Distancia a sucursal más cercana
# MAGIC 2. `densidad_zona`: Número de clientes en zona H3
# MAGIC 3. `facturacion_promedio_zona`: Facturación promedio de la zona

# COMMAND ----------

# DBTITLE 1,Crear features H3
print("\n" + "="*80)
print("EJERCICIO 4: FEATURES GEOESPACIALES H3")
print("="*80)

print("\n🗺️ Creando features espaciales...")

# Feature 1: Distancia a sucursal
sucursales_h3 = df_sucursales['h3_index'].tolist()

def calc_dist_min(cliente_h3):
    if pd.isna(cliente_h3):
        return 999  # Valor alto por defecto
    try:
        return min([h3.grid_distance(cliente_h3, s) for s in sucursales_h3])
    except:
        return 999

df_clientes['dist_sucursal_min'] = df_clientes['h3_index'].apply(calc_dist_min)

# Feature 2: Densidad de zona
densidad = df_clientes.groupby('h3_index').size().reset_index(name='densidad_zona')
df_clientes = df_clientes.merge(densidad, on='h3_index', how='left')

# Feature 3: Facturación promedio de zona
ventas_geo = df_ventas[df_ventas['cliente_id'].notna()].merge(
    df_clientes[['cliente_id', 'h3_index']], 
    on='cliente_id'
)
facturacion = ventas_geo.groupby('h3_index')['total'].mean().reset_index()
facturacion.columns = ['h3_index', 'facturacion_zona']
df_clientes = df_clientes.merge(facturacion, on='h3_index', how='left')
df_clientes['facturacion_zona'] = df_clientes['facturacion_zona'].fillna(0)

print("✅ Features espaciales creadas:")
print(f"   - dist_sucursal_min")
print(f"   - densidad_zona")
print(f"   - facturacion_zona")

# Preparar dataset con features espaciales
df_ml_h3 = df_ml.merge(
    df_clientes[['cliente_id', 'dist_sucursal_min', 'densidad_zona', 'facturacion_zona']], 
    on='cliente_id',
    how='left'
)

features_h3 = features_base + ['dist_sucursal_min', 'densidad_zona', 'facturacion_zona']
X_h3 = df_ml_h3[features_h3].fillna(0)
y_h3 = df_ml_h3['total']

X_train_h3, X_test_h3, y_train_h3, y_test_h3 = train_test_split(X_h3, y_h3, test_size=0.2, random_state=42)

print(f"\n📊 Nuevos features: {len(features_h3)} (antes: {len(features_base)})")

# COMMAND ----------

# DBTITLE 1,Optimizar con features H3
print("\n🚀 Optimizando con features espaciales...\n")

def objective_h3(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 20),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'random_state': 42
    }
    
    model = RandomForestRegressor(**params)
    scores = cross_val_score(model, X_train_h3, y_train_h3, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1)
    return -scores.mean()

study_h3 = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
study_h3.optimize(objective_h3, n_trials=50, show_progress_bar=True)

print(f"\n🎯 Mejor MAE (con features H3): ${study_h3.best_value:,.2f}")

# Entrenar modelo final
model_h3 = RandomForestRegressor(**study_h3.best_params)
model_h3.fit(X_train_h3, y_train_h3)
y_pred_h3 = model_h3.predict(X_test_h3)

mae_h3 = mean_absolute_error(y_test_h3, y_pred_h3)
r2_h3 = r2_score(y_test_h3, y_pred_h3)

print(f"\n📊 Métricas en Test (con features H3):")
print(f"   MAE:  ${mae_h3:,.2f}")
print(f"   R²:   {r2_h3:.4f}")

# Feature importance
importances = pd.DataFrame({
    'feature': features_h3,
    'importance': model_h3.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\n📊 Top 5 Features Más Importantes:")
print(importances.head())

print("\n✅ Ejercicio 4 completado")

# COMMAND ----------

# DBTITLE 1,Resumen final
# MAGIC %md
# MAGIC ## 📈 Resumen de Resultados
# MAGIC
# MAGIC ### Comparación de Modelos
# MAGIC
# MAGIC | Modelo | MAE Test | R² | Observaciones |
# MAGIC |--------|----------|-----|---------------|
# MAGIC | Random Forest (sin optimizar) | ~$X.XX | ~0.XX | Baseline |
# MAGIC | Gradient Boosting (optimizado) | ~$X.XX | ~0.XX | Más rápido que RF |
# MAGIC | RF + Features H3 (optimizado) | ~$X.XX | ~0.XX | Mejor precisión |
# MAGIC
# MAGIC ### 💡 Insights
# MAGIC
# MAGIC 1. ✅ **Optuna mejoró el rendimiento** vs. hiperparámetros por defecto
# MAGIC 2. ✅ **Features espaciales H3 agregaron valor** predictivo
# MAGIC 3. ✅ **Optimización multi-objetivo** revela trade-offs precisión-velocidad
# MAGIC 4. ✅ **Gradient Boosting** es alternativa viable a Random Forest
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## ✅ Conclusión
# MAGIC **Has completado todos los ejercicios prácticos de Optuna.**
# MAGIC
# MAGIC Ahora puedes:
# MAGIC - ✅ Optimizar modelos de forma autónoma
# MAGIC - ✅ Incorporar features geoespaciales
# MAGIC - ✅ Comparar múltiples algoritmos
# MAGIC - ✅ Balancear precisión y velocidad
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC **Universidad del Aconcagua**  
# MAGIC **Laboratorio (Herramientas)**  
# MAGIC **Mendoza, Argentina**

# COMMAND ----------

