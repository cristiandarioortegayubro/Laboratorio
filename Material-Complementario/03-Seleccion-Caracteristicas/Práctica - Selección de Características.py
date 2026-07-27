# Databricks notebook source
# DBTITLE 1,Conclusiones Finales
# MAGIC %md
# MAGIC ## 🎓 Conclusiones Finales
# MAGIC
# MAGIC ### 📊 Resumen de Ejercicios
# MAGIC
# MAGIC **Ejercicio 1: Eliminar Features Redundantes**
# MAGIC - ✅ Identificamos features altamente correlacionadas (> 0.9)
# MAGIC - ✅ Eliminamos features redundantes sin perder rendimiento
# MAGIC - 💡 **Lección**: Features correlacionadas aportan información duplicada
# MAGIC
# MAGIC **Ejercicio 2: Mutual Information**
# MAGIC - ✅ Calculamos MI scores para todas las features
# MAGIC - ✅ Seleccionamos top 3 features más informativas
# MAGIC - 💡 **Lección**: MI captura relaciones no lineales con el target
# MAGIC
# MAGIC **Ejercicio 3: RFECV**
# MAGIC - ✅ Encontramos el número óptimo de features automáticamente
# MAGIC - ✅ Visualizamos curva de rendimiento vs. número de features
# MAGIC - 💡 **Lección**: RFECV elimina la adivinanza - te dice cuántas features necesitas
# MAGIC
# MAGIC **Ejercicio 4: Lasso vs. RF Importance**
# MAGIC - ✅ Comparamos dos métodos embedded populares
# MAGIC - ✅ Observamos que seleccionan features DIFERENTES
# MAGIC - 💡 **Lección**: Lasso (lineal) vs. RF (no lineal) capturan patrones distintos
# MAGIC
# MAGIC **Ejercicio 5: Features H3**
# MAGIC - ✅ Evaluamos múltiples resoluciones H3
# MAGIC - ✅ Seleccionamos la resolución óptima por MI + CV
# MAGIC - 💡 **Lección**: NO usar múltiples resoluciones H3 simultáneamente
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 💡 Mensajes Clave
# MAGIC
# MAGIC 1. 🔑 **Menos features puede ser MEJOR** - Evita overfitting y simplifica modelos
# MAGIC 2. ⚠️ **Siempre usar Pipeline** - Evita data leakage en CV
# MAGIC 3. 🎯 **Probar múltiples métodos** - Lasso, RF, MI, RFECV
# MAGIC 4. ⏱️ **Filter → Embedded → Wrapper** - Workflow incremental
# MAGIC 5. 📊 **Validar con CV** - No confiar en feature importance solo
# MAGIC 6. 🗺️ **Features H3**: Seleccionar UNA resolución (res 7 para ventas urbanas)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 🚀 Workflow Recomendado
# MAGIC
# MAGIC **Paso 1: Limpieza Rápida** (⏱️ 5 minutos)
# MAGIC 1. Eliminar features con varianza cero
# MAGIC 2. Eliminar features altamente correlacionadas (> 0.95)
# MAGIC
# MAGIC **Paso 2: Selección Inicial** (⏱️ 15 minutos)
# MAGIC 3. Calcular Mutual Information
# MAGIC 4. Seleccionar top 20-30 features
# MAGIC
# MAGIC **Paso 3: Optimización** (⏱️ 1-2 horas)
# MAGIC 5. Usar RFECV para encontrar N óptimo
# MAGIC 6. Comparar con RF Importance y Lasso
# MAGIC 7. Validar con nested CV
# MAGIC
# MAGIC **Paso 4: Producción**
# MAGIC 8. Documentar features seleccionadas
# MAGIC 9. Monitorear importancia en producción
# MAGIC 10. Re-evaluar cada 3-6 meses
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### ✅ Checklist Final
# MAGIC
# MAGIC Antes de ir a producción, verifica:
# MAGIC
# MAGIC - [ ] Eliminaste features con varianza cero
# MAGIC - [ ] Eliminaste features altamente correlacionadas
# MAGIC - [ ] Probaste al menos 2 métodos de selección
# MAGIC - [ ] Usaste Pipeline para evitar data leakage
# MAGIC - [ ] Evaluaste con validación cruzada (no solo train/test split)
# MAGIC - [ ] Documentaste features seleccionadas y por qué
# MAGIC - [ ] Verificaste que el modelo es interpretable
# MAGIC - [ ] Comparaste rendimiento: todas las features vs. seleccionadas
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 📚 Recursos Adicionales
# MAGIC
# MAGIC - [Scikit-learn Feature Selection Guide](https://scikit-learn.org/stable/modules/feature_selection.html)
# MAGIC - [Feature Engineering and Selection Book](http://www.feat.engineering/)
# MAGIC - [Interpretable Machine Learning Book](https://christophm.github.io/interpretable-ml-book/)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🎉 ¡Felicitaciones!
# MAGIC
# MAGIC **Has completado el módulo práctico de Selección de Características.**
# MAGIC
# MAGIC Ahora puedes:
# MAGIC - ✅ Identificar y eliminar features redundantes
# MAGIC - ✅ Aplicar 5+ métodos de selección (filtro, wrapper, embedded)
# MAGIC - ✅ Usar RFECV para encontrar N óptimo
# MAGIC - ✅ Comparar Lasso vs. RF Importance
# MAGIC - ✅ Seleccionar features H3 correctamente
# MAGIC - ✅ Evitar data leakage con Pipelines
# MAGIC - ✅ Validar selección con CV
# MAGIC
# MAGIC **Próximo módulo**: Interpretabilidad de Modelos (SHAP, LIME)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC **Universidad del Aconcagua**  
# MAGIC **Laboratorio (Herramientas)**  
# MAGIC **Mendoza, Argentina**

# COMMAND ----------

# DBTITLE 1,Ejercicio 3: RFECV
# MAGIC %md
# MAGIC ## 💻 Ejercicio 3: RFECV - Encontrar N Óptimo Automáticamente
# MAGIC
# MAGIC ### 🎯 Problema
# MAGIC
# MAGIC ¿Cuántas features debemos seleccionar? **RFECV** lo decide automáticamente usando validación cruzada.
# MAGIC
# MAGIC **Objetivo**:
# MAGIC - Usar RFECV para encontrar el número óptimo de features
# MAGIC - Visualizar la curva de rendimiento vs. número de features
# MAGIC - Comparar con otros métodos
# MAGIC
# MAGIC ---

# COMMAND ----------

# DBTITLE 1,Ejercicio 3 - Solución
print("="*80)
print("EJERCICIO 3: RFECV - ENCONTRAR N ÓPTIMO")
print("="*80)

# Preparar datos
df_ml = df[df['cliente_id'].notna()].copy()

features = [
    'sucursal_id', 'dia_semana', 'mes', 'dia_mes', 'trimestre',
    'es_fin_semana', 'segmento_encoded'
]

X = df_ml[features]
y = df_ml['total']

print(f"\n📊 Dataset: {len(X):,} registros, {len(features)} features")
print(f"\n1️⃣ Ejecutando RFECV (puede tardar un poco)...\n")

# RFECV
model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
cv = KFold(n_splits=5, shuffle=True, random_state=42)

selector = RFECV(
    estimator=model,
    step=1,
    cv=cv,
    scoring='neg_mean_absolute_error',
    n_jobs=-1
)

selector.fit(X, y)

print(f"✅ RFECV completado")
print(f"\nNúmero óptimo de features: {selector.n_features_}")

selected_features = X.columns[selector.support_].tolist()
print(f"Features seleccionadas: {selected_features}")

# Ranking de features
ranking_df = pd.DataFrame({
    'feature': X.columns,
    'ranking': selector.ranking_,
    'selected': selector.support_
}).sort_values('ranking')

print("\nRanking de features:")
for idx, row in ranking_df.iterrows():
    status = "✅" if row['selected'] else "❌"
    print(f"  {status} {row['feature']:20s}: Rank {row['ranking']}")

print(f"\n2️⃣ Visualizando curva de rendimiento...\n")

# Visualizar curva
plt.figure(figsize=(12, 6))
plt.plot(
    range(1, len(selector.cv_results_['mean_test_score']) + 1),
    -selector.cv_results_['mean_test_score'],  # Negar porque es neg_mae
    marker='o'
)
plt.xlabel('Número de Features')
plt.ylabel('MAE (CV)')
plt.title('RFECV: MAE vs. Número de Features')
plt.axvline(x=selector.n_features_, color='red', linestyle='--', 
            label=f'Óptimo = {selector.n_features_} features')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print(f"\n3️⃣ Evaluando modelo final con features seleccionadas...\n")

# Evaluar
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modelo con todas las features
model_all = RandomForestRegressor(n_estimators=100, random_state=42)
model_all.fit(X_train, y_train)
y_pred_all = model_all.predict(X_test)
mae_all = mean_absolute_error(y_test, y_pred_all)

# Modelo con features seleccionadas
X_train_selected = selector.transform(X_train)
X_test_selected = selector.transform(X_test)

model_selected = RandomForestRegressor(n_estimators=100, random_state=42)
model_selected.fit(X_train_selected, y_train)
y_pred_selected = model_selected.predict(X_test_selected)
mae_selected = mean_absolute_error(y_test, y_pred_selected)

print("Resultados en test set:")
print(f"\nCon todas las features ({len(features)}):")
print(f"  MAE: ${mae_all:.2f}")

print(f"\nCon features seleccionadas por RFECV ({selector.n_features_}):")
print(f"  MAE: ${mae_selected:.2f}")

mejora = mae_all - mae_selected
print(f"\nMejora: ${mejora:.2f} ({mejora/mae_all*100:.1f}%)")

if mejora > 0:
    print("✅ RFECV encontró el subconjunto óptimo!")
else:
    print("⚠️ Rendimiento similar - todas las features eran importantes")

print(f"\n💡 Conclusión: RFECV encuentra automáticamente el número óptimo de features.")

# COMMAND ----------

# DBTITLE 1,Ejercicio 4: Lasso vs RF
# MAGIC %md
# MAGIC ## 💻 Ejercicio 4: Comparación Lasso vs. RF Importance
# MAGIC
# MAGIC ### 🎯 Problema
# MAGIC
# MAGIC Comparar dos métodos embedded populares: **Lasso** (modelo lineal) vs. **RF Importance** (modelo no lineal).
# MAGIC
# MAGIC **Objetivo**:
# MAGIC - Seleccionar top 5 features con cada método
# MAGIC - Comparar features seleccionadas
# MAGIC - Evaluar rendimiento
# MAGIC - Analizar diferencias
# MAGIC
# MAGIC ---

# COMMAND ----------

# DBTITLE 1,Ejercicio 4 - Solución
print("="*80)
print("EJERCICIO 4: LASSO VS. RF IMPORTANCE")
print("="*80)

# Preparar datos
df_ml = df[df['cliente_id'].notna()].copy()

features = [
    'sucursal_id', 'dia_semana', 'mes', 'dia_mes', 'trimestre',
    'es_fin_semana', 'segmento_encoded'
]

X = df_ml[features]
y = df_ml['total']

print(f"\n📊 Dataset: {len(X):,} registros, {len(features)} features")

# ========== MÉTODO 1: LASSO ==========
print(f"\n1️⃣ Selección con Lasso...\n")

# Escalar (importante para Lasso)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# LassoCV para encontrar alpha óptimo
model_lasso = LassoCV(cv=5, random_state=42, n_jobs=-1)
model_lasso.fit(X_scaled, y)

print(f"Alpha óptimo: {model_lasso.alpha_:.4f}")

# Coeficientes
coefs_lasso = pd.DataFrame({
    'feature': features,
    'coef': model_lasso.coef_
}).sort_values('coef', key=abs, ascending=False)

print("\nCoeficientes Lasso:")
for idx, row in coefs_lasso.iterrows():
    status = "✅" if row['coef'] != 0 else "❌"
    print(f"  {status} {row['feature']:20s}: {row['coef']:8.2f}")

# Seleccionar features con coef != 0
lasso_features = coefs_lasso[coefs_lasso['coef'] != 0]['feature'].head(5).tolist()
print(f"\nTop 5 features por Lasso: {lasso_features}")

# ========== MÉTODO 2: RF IMPORTANCE ==========
print(f"\n2️⃣ Selección con RF Importance...\n")

model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
model_rf.fit(X, y)

importances_rf = pd.DataFrame({
    'feature': features,
    'importance': model_rf.feature_importances_
}).sort_values('importance', ascending=False)

print("RF Feature Importances:")
for idx, row in importances_rf.iterrows():
    print(f"  {row['feature']:20s}: {row['importance']:.4f}")

rf_features = importances_rf['feature'].head(5).tolist()
print(f"\nTop 5 features por RF: {rf_features}")

# ========== COMPARACIÓN ==========
print(f"\n3️⃣ Comparando features seleccionadas...\n")

lasso_set = set(lasso_features)
rf_set = set(rf_features)

common = lasso_set & rf_set
only_lasso = lasso_set - rf_set
only_rf = rf_set - lasso_set

print(f"Features en ambos métodos: {list(common)}")
print(f"Solo en Lasso: {list(only_lasso)}")
print(f"Solo en RF: {list(only_rf)}")

print(f"\n4️⃣ Evaluando rendimiento...\n")

# Modelo base
model_eval = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)

# Con features de Lasso
if lasso_features:
    scores_lasso = cross_val_score(model_eval, X[lasso_features], y, cv=5, 
                                    scoring='neg_mean_absolute_error', n_jobs=-1)
else:
    scores_lasso = None

# Con features de RF
scores_rf = cross_val_score(model_eval, X[rf_features], y, cv=5, 
                            scoring='neg_mean_absolute_error', n_jobs=-1)

# Con todas
scores_all = cross_val_score(model_eval, X, y, cv=5, 
                             scoring='neg_mean_absolute_error', n_jobs=-1)

print("Resultados:")
print(f"\nTodas las features ({len(features)}):")
print(f"  MAE: ${-scores_all.mean():.2f} ± ${scores_all.std():.2f}")

if scores_lasso is not None:
    print(f"\nTop 5 Lasso:")
    print(f"  MAE: ${-scores_lasso.mean():.2f} ± ${scores_lasso.std():.2f}")

print(f"\nTop 5 RF:")
print(f"  MAE: ${-scores_rf.mean():.2f} ± ${scores_rf.std():.2f}")

print(f"\n" + "="*80)
print("CONCLUSIÓN")
print("="*80)
print(f"\n💡 Lasso y RF pueden seleccionar features DIFERENTES:")
print(f"    - Lasso: Penaliza features linealmente correlacionadas")
print(f"    - RF: Captura interacciones no lineales")
print(f"\n✅ Recomendación: Probar AMBOS métodos y elegir según rendimiento CV.")

# COMMAND ----------

# DBTITLE 1,Ejercicio 5: Features H3
# MAGIC %md
# MAGIC ## 💻 Ejercicio 5: Selección de Features H3 (Geoespaciales)
# MAGIC
# MAGIC ### 🎯 Problema
# MAGIC
# MAGIC Tenemos múltiples **resoluciones H3** (res 5, 7, 9) para las mismas coordenadas - **features redundantes**.
# MAGIC
# MAGIC **Objetivo**:
# MAGIC - Crear features H3 con 3 resoluciones diferentes
# MAGIC - Calcular Mutual Information para cada una
# MAGIC - Seleccionar la resolución óptima
# MAGIC - Evaluar rendimiento
# MAGIC
# MAGIC ---

# COMMAND ----------

# DBTITLE 1,Ejercicio 5 - Solución
print("="*80)
print("EJERCICIO 5: SELECCIÓN DE FEATURES H3")
print("="*80)

# Cargar datos con coordenadas
try:
    # Intentar cargar datos con H3 si existen
    df_h3 = df[df['cliente_id'].notna()].copy()
    
    # Simular diferentes resoluciones H3 (para demostrar el concepto)
    # En datos reales, usar h3.geo_to_h3(lat, lon, resolution)
    
    # Simular res 5 (regiones grandes, ~250 km²)
    # Pocas zonas distintas
    np.random.seed(42)
    df_h3['h3_res5'] = np.random.randint(0, 5, size=len(df_h3))  # 5 zonas grandes
    
    # Simular res 7 (zonas medianas, ~5 km²) - IDEAL para ventas urbanas
    df_h3['h3_res7'] = np.random.randint(0, 20, size=len(df_h3))  # 20 zonas medianas
    
    # Simular res 9 (zonas pequeñas, ~0.1 km²)
    # Muchas zonas, puede tener overfitting
    df_h3['h3_res9'] = np.random.randint(0, 100, size=len(df_h3))  # 100 zonas pequeñas
    
    print(f"\n📊 Dataset: {len(df_h3):,} registros")
    print(f"\n1️⃣ Features H3 creadas con 3 resoluciones...\n")
    
    print("Cardinalidad (número de valores únicos):")
    print(f"  H3 res 5: {df_h3['h3_res5'].nunique()} zonas (regiones grandes)")
    print(f"  H3 res 7: {df_h3['h3_res7'].nunique()} zonas (zonas urbanas)")
    print(f"  H3 res 9: {df_h3['h3_res9'].nunique()} zonas (bloques pequeños)")
    
    print(f"\n2️⃣ Calculando Mutual Information para cada resolución...\n")
    
    # Preparar datos
    h3_features = ['h3_res5', 'h3_res7', 'h3_res9']
    X_h3 = df_h3[h3_features]
    y = df_h3['total']
    
    # Calcular MI para cada resolución
    mi_scores = mutual_info_regression(X_h3, y, random_state=42)
    
    mi_df = pd.DataFrame({
        'resolucion': h3_features,
        'mi_score': mi_scores
    }).sort_values('mi_score', ascending=False)
    
    print("Mutual Information Scores:")
    for idx, row in mi_df.iterrows():
        print(f"  {row['resolucion']}: {row['mi_score']:.4f}")
    
    # Visualizar
    plt.figure(figsize=(10, 5))
    plt.bar(mi_df['resolucion'], mi_df['mi_score'])
    plt.xlabel('Resolución H3')
    plt.ylabel('Mutual Information Score')
    plt.title('MI Score por Resolución H3')
    plt.tight_layout()
    plt.show()
    
    print(f"\n3️⃣ Comparando rendimiento por resolución...\n")
    
    # Probar cada resolución
    model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
    
    results = []
    for feature in h3_features:
        X_single = df_h3[[feature]]
        scores = cross_val_score(model, X_single, y, cv=5, 
                                scoring='neg_mean_absolute_error', n_jobs=-1)
        results.append({
            'resolucion': feature,
            'mae': -scores.mean(),
            'std': scores.std()
        })
    
    results_df = pd.DataFrame(results).sort_values('mae')
    
    print("Rendimiento por resolución (solo feature H3):")
    for idx, row in results_df.iterrows():
        print(f"  {row['resolucion']}: MAE = ${row['mae']:.2f} ± ${row['std']:.2f}")
    
    best_resolution = results_df.iloc[0]['resolucion']
    print(f"\n✅ Mejor resolución: {best_resolution}")
    
    print(f"\n4️⃣ Combinando con otras features...\n")
    
    # Modelo con mejor resolución + otras features
    other_features = ['dia_semana', 'mes', 'segmento_encoded']
    all_features = [best_resolution] + other_features
    
    X_combined = df_h3[all_features]
    scores_combined = cross_val_score(model, X_combined, y, cv=5, 
                                     scoring='neg_mean_absolute_error', n_jobs=-1)
    
    print(f"Con {best_resolution} + otras features:")
    print(f"  MAE: ${-scores_combined.mean():.2f} ± ${scores_combined.std():.2f}")
    
    print(f"\n" + "="*80)
    print("CONCLUSIÓN")
    print("="*80)
    print(f"\n💡 Para features H3:")
    print(f"    1. NO uses múltiples resoluciones simultáneamente (redundantes)")
    print(f"    2. Resolución 7 (~5 km²) suele ser óptima para ventas urbanas")
    print(f"    3. Resolución 5 (~250 km²) para análisis regional")
    print(f"    4. Resolución 9 (~0.1 km²) puede causar overfitting (demasiado granular)")
    print(f"\n✅ Selecciona la resolución con mejor balance MI score + rendimiento CV.")
    
except Exception as e:
    print(f"\n⚠️ Error al crear features H3: {e}")
    print(f"\nNota: Este ejercicio requiere coordenadas geográficas y la librería h3.")
    print(f"Para datos reales, usa: h3.geo_to_h3(lat, lon, resolution)")

# COMMAND ----------

# DBTITLE 1,Título
# MAGIC %md
# MAGIC # 🏃 Práctica: Selección de Características
# MAGIC ## Material Complementario - Laboratorio (Herramientas)
# MAGIC ### Universidad del Aconcagua - Mendoza, Argentina
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 🎯 Objetivos
# MAGIC
# MAGIC Aplicar los métodos de selección de features del notebook teórico a **problemas reales** de la panadería.
# MAGIC
# MAGIC ### 📝 Ejercicios
# MAGIC
# MAGIC 1. **Eliminar Features Redundantes** (correlación alta)
# MAGIC 2. **Selección con Mutual Information** para predecir ventas
# MAGIC 3. **RFECV**: Encontrar N óptimo de features automáticamente
# MAGIC 4. **Comparación Lasso vs. RF Importance** 
# MAGIC 5. **Selección de Features H3** (múltiples resoluciones)
# MAGIC
# MAGIC ### ⏱️ Duración: 2-3 horas
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

from sklearn.feature_selection import (
    VarianceThreshold, SelectKBest, RFE, RFECV, SelectFromModel,
    chi2, f_classif, mutual_info_regression, mutual_info_classif
)
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import Lasso, LassoCV, Ridge
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score

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
df_ventas['dia_mes'] = df_ventas['fecha'].dt.day
df_ventas['trimestre'] = df_ventas['fecha'].dt.quarter
df_ventas['es_fin_semana'] = df_ventas['dia_semana'].isin([5, 6]).astype(int)

# Merge
df = df_ventas.merge(df_clientes, on='cliente_id', how='left')
df['segmento_encoded'] = df['segmento'].astype('category').cat.codes

print("\n✅ Features creadas")

# COMMAND ----------

# DBTITLE 1,Ejercicio 1: Eliminar Features Redundantes
# MAGIC %md
# MAGIC ## 💻 Ejercicio 1: Eliminar Features Redundantes por Correlación
# MAGIC
# MAGIC ### 🎯 Problema
# MAGIC
# MAGIC Tenemos features **altamente correlacionadas** que aportan información redundante.
# MAGIC
# MAGIC **Objetivo**: 
# MAGIC - Identificar pares de features con correlación > 0.9
# MAGIC - Eliminar una de cada par
# MAGIC - Comparar rendimiento antes y después
# MAGIC
# MAGIC ---

# COMMAND ----------

# DBTITLE 1,Ejercicio 1 - Solución
print("="*80)
print("EJERCICIO 1: ELIMINAR FEATURES REDUNDANTES")
print("="*80)

# Preparar datos con features correlacionadas
df_ml = df[df['cliente_id'].notna()].copy()

# Crear features "artificialmente" correlacionadas para demostrar
df_ml['dia_mes_x2'] = df_ml['dia_mes'] * 2  # Perfectamente correlacionada con dia_mes
df_ml['mes_x_dia'] = df_ml['mes'] * df_ml['dia_semana']  # Interacción

features = [
    'sucursal_id', 'dia_semana', 'mes', 'dia_mes', 'dia_mes_x2',  # dia_mes_x2 es redundante
    'trimestre', 'es_fin_semana', 'segmento_encoded', 'mes_x_dia'
]

X = df_ml[features]
y = df_ml['total']

print(f"\n📊 Dataset: {len(X):,} registros, {len(features)} features")
print(f"\n1️⃣ Calculando matriz de correlación...\n")

# Matriz de correlación
corr_matrix = X.corr().abs()

# Encontrar pares altamente correlacionados (> 0.9)
threshold = 0.9
correlated_pairs = []

for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        if corr_matrix.iloc[i, j] > threshold:
            correlated_pairs.append((
                corr_matrix.columns[i],
                corr_matrix.columns[j],
                corr_matrix.iloc[i, j]
            ))

if correlated_pairs:
    print("Features altamente correlacionadas (> 0.9):")
    for feat1, feat2, corr in correlated_pairs:
        print(f"  - {feat1} <-> {feat2}: {corr:.3f}")
else:
    print("✅ No hay features con correlación > 0.9")

# Eliminar features redundantes
to_drop = set()
for feat1, feat2, corr in correlated_pairs:
    # Eliminar la segunda feature del par
    to_drop.add(feat2)

features_reduced = [f for f in features if f not in to_drop]

print(f"\n2️⃣ Features eliminadas: {list(to_drop)}")
print(f"\n3️⃣ Comparando rendimiento...\n")

# Comparar rendimiento
model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)

# Con todas las features
scores_all = cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)

# Con features reducidas
X_reduced = X[features_reduced]
scores_reduced = cross_val_score(model, X_reduced, y, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)

print("Resultados:")
print(f"\nCon todas las features ({len(features)}):")
print(f"  MAE: ${-scores_all.mean():.2f} ± ${scores_all.std():.2f}")

print(f"\nSin features redundantes ({len(features_reduced)}):")
print(f"  MAE: ${-scores_reduced.mean():.2f} ± ${scores_reduced.std():.2f}")

diff = (-scores_reduced.mean()) - (-scores_all.mean())
print(f"\nDiferencia: ${diff:.2f} ({diff/-scores_all.mean()*100:.1f}%)")

if diff < 0:
    print("✅ Mejor rendimiento sin features redundantes!")
else:
    print("⚠️ Rendimiento similar - features eran redundantes pero no dañinas")

print(f"\n💡 Conclusión: Eliminar features redundantes simplifica el modelo sin perder rendimiento.")

# COMMAND ----------

# DBTITLE 1,Ejercicio 2: Mutual Information
# MAGIC %md
# MAGIC ## 💻 Ejercicio 2: Selección con Mutual Information
# MAGIC
# MAGIC ### 🎯 Problema
# MAGIC
# MAGIC Predicir **ventas totales** seleccionando las features más informativas usando **Mutual Information**.
# MAGIC
# MAGIC **Objetivo**:
# MAGIC - Calcular MI scores para todas las features
# MAGIC - Seleccionar top 5 features
# MAGIC - Comparar con usar todas las features
# MAGIC - Visualizar scores
# MAGIC
# MAGIC ---

# COMMAND ----------

# DBTITLE 1,Ejercicio 2 - Solución
print("="*80)
print("EJERCICIO 2: SELECCIÓN CON MUTUAL INFORMATION")
print("="*80)

# Preparar datos
df_ml = df[df['cliente_id'].notna()].copy()

features = [
    'sucursal_id', 'dia_semana', 'mes', 'trimestre', 'es_fin_semana',
    'segmento_encoded'
]

X = df_ml[features]
y = df_ml['total']

print(f"\n📊 Dataset: {len(X):,} registros, {len(features)} features")
print(f"\n1️⃣ Calculando Mutual Information scores...\n")

# Calcular MI
mi_scores = mutual_info_regression(X, y, random_state=42)

# Crear DataFrame con scores
mi_df = pd.DataFrame({
    'feature': features,
    'mi_score': mi_scores
}).sort_values('mi_score', ascending=False)

print("Mutual Information Scores:")
for idx, row in mi_df.iterrows():
    print(f"  {row['feature']:20s}: {row['mi_score']:.4f}")

# Visualizar
plt.figure(figsize=(10, 6))
plt.barh(mi_df['feature'], mi_df['mi_score'])
plt.xlabel('Mutual Information Score')
plt.title('Feature Importance por Mutual Information')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

print(f"\n2️⃣ Seleccionando top 3 features...\n")

# Seleccionar top 3
top_features = mi_df['feature'].head(3).tolist()
print(f"Features seleccionadas: {top_features}")

print(f"\n3️⃣ Comparando rendimiento...\n")

# Modelo
model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)

# Con todas las features
scores_all = cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)

# Con top 3 features
X_selected = X[top_features]
scores_selected = cross_val_score(model, X_selected, y, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)

print("Resultados:")
print(f"\nTodas las features ({len(features)}):")
print(f"  MAE: ${-scores_all.mean():.2f} ± ${scores_all.std():.2f}")

print(f"\nTop 3 por MI:")
print(f"  MAE: ${-scores_selected.mean():.2f} ± ${scores_selected.std():.2f}")

mejora = (-scores_all.mean()) - (-scores_selected.mean())
print(f"\nMejora: ${mejora:.2f} ({mejora/-scores_all.mean()*100:.1f}%)")

if mejora > 0:
    print("✅ Mejor rendimiento con menos features!")
else:
    print("⚠️ Rendimiento similar o ligeramente peor")

print(f"\n💡 Conclusión: Mutual Information identifica features con relaciones no lineales con el target.")

# COMMAND ----------

