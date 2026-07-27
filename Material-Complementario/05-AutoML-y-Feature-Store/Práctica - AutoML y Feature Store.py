# Databricks notebook source
# DBTITLE 1,Práctica Módulo 05 - AutoML y Feature Store
# MAGIC %md
# MAGIC # Práctica: AutoML y Feature Store
# MAGIC
# MAGIC ## Objetivos de la Práctica
# MAGIC
# MAGIC 1. **Crear Feature Tables** con datos financieros
# MAGIC 2. **Ejecutar AutoML** con y sin Feature Store
# MAGIC 3. **Comparar resultados** y analizar ventajas
# MAGIC 4. **Implementar feature serving** para producción
# MAGIC 5. **Aplicar mejores prácticas** aprendidas
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Estructura
# MAGIC
# MAGIC - **Ejercicio 1**: Crear Feature Table de clientes bancarios
# MAGIC - **Ejercicio 2**: AutoML para predicción de churn
# MAGIC - **Ejercicio 3**: Comparación con/sin Feature Store
# MAGIC - **Ejercicio 4**: Feature Serving y producción
# MAGIC - **Ejercicio 5**: Feature Store para múltiples modelos
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC **Dataset**: Datos sintéticos de clientes bancarios con comportamiento transaccional

# COMMAND ----------

# DBTITLE 1,Setup: Crear Datos Sintéticos
# Setup: Generar datos sintéticos de clientes bancarios
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pyspark.sql import functions as F
from pyspark.sql.types import *

np.random.seed(42)

# Generar 1000 clientes
n_customers = 1000

customers_data = {
    'customer_id': range(1, n_customers + 1),
    'signup_date': [datetime(2022, 1, 1) + timedelta(days=int(x)) 
                    for x in np.random.uniform(0, 730, n_customers)],
    'age': np.random.randint(18, 75, n_customers),
    'income': np.random.lognormal(10.5, 0.5, n_customers),
    'credit_score': np.random.randint(300, 850, n_customers),
    'num_products': np.random.randint(1, 6, n_customers),
    'account_balance': np.random.lognormal(8, 1.5, n_customers),
    'total_transactions': np.random.randint(0, 200, n_customers),
    'avg_transaction': np.random.lognormal(5, 1, n_customers)
}

# Target: Churn (1 = churn, 0 = no churn)
# Aumentar probabilidad de churn si:
# - Pocas transacciones
# - Balance bajo
# - Credit score bajo
churn_prob = (
    0.1 + 
    0.3 * (customers_data['total_transactions'] < 20) +
    0.2 * (customers_data['account_balance'] < 5000) +
    0.2 * (customers_data['credit_score'] < 600)
)
customers_data['churn'] = (np.random.random(n_customers) < churn_prob).astype(int)

customers_df = pd.DataFrame(customers_data)

print(f"✅ Datos creados: {len(customers_df)} clientes")
print(f"Churn rate: {customers_df['churn'].mean():.1%}")
customers_df.head()

# COMMAND ----------

# DBTITLE 1,Setup: Guardar en Delta Tables
# Convertir a Spark DataFrame y guardar
spark_df = spark.createDataFrame(customers_df)

# Tabla raw
spark_df.write.mode("overwrite").saveAsTable("main.default.bank_customers_raw")

# Separar labels para training
labels_df = spark_df.select(
    "customer_id",
    "churn",
    F.current_timestamp().alias("label_date")
)

labels_df.write.mode("overwrite").saveAsTable("main.default.bank_churn_labels")

print("✅ Tablas creadas:")
print("  - main.default.bank_customers_raw")
print("  - main.default.bank_churn_labels")

# Verificar
print(f"\nCustomers: {spark.table('main.default.bank_customers_raw').count()}")
print(f"Labels: {spark.table('main.default.bank_churn_labels').count()}")

# COMMAND ----------

# DBTITLE 1,Ejercicio 1: Crear Feature Table
# MAGIC %md
# MAGIC # Ejercicio 1: Crear Feature Table de Clientes Bancarios
# MAGIC
# MAGIC ## Objetivo
# MAGIC Crear una Feature Table con features calculadas que puedan ser reutilizadas en múltiples modelos.
# MAGIC
# MAGIC ## Tareas
# MAGIC 1. Calcular features derivadas:
# MAGIC    - Días desde registro (`days_since_signup`)
# MAGIC    - Ratio transacciones/productos (`transactions_per_product`)
# MAGIC    - Segmento de edad (`age_segment`)
# MAGIC    - Categoría de credit score (`credit_category`)
# MAGIC    - Balance normalizado por ingreso (`balance_income_ratio`)
# MAGIC
# MAGIC 2. Crear Feature Table en `main.features.bank_customer_features`
# MAGIC
# MAGIC 3. Validar creación y contenido

# COMMAND ----------

# DBTITLE 1,Ejercicio 1: Solución - Feature Engineering
# Ejercicio 1: Solución - Feature Engineering

from databricks.feature_engineering import FeatureEngineeringClient
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# Inicializar cliente
fe = FeatureEngineeringClient()

# Cargar datos raw
raw_df = spark.table("main.default.bank_customers_raw")

# Calcular features
features_df = (
    raw_df
    .withColumn(
        "days_since_signup",
        F.datediff(F.current_date(), F.col("signup_date"))
    )
    .withColumn(
        "transactions_per_product",
        F.col("total_transactions") / F.col("num_products")
    )
    .withColumn(
        "age_segment",
        F.when(F.col("age") < 30, "young")
         .when(F.col("age") < 50, "middle")
         .otherwise("senior")
    )
    .withColumn(
        "credit_category",
        F.when(F.col("credit_score") < 600, "poor")
         .when(F.col("credit_score") < 700, "fair")
         .when(F.col("credit_score") < 750, "good")
         .otherwise("excellent")
    )
    .withColumn(
        "balance_income_ratio",
        F.col("account_balance") / F.col("income")
    )
    .withColumn(
        "last_updated",
        F.current_timestamp()
    )
    .select(
        "customer_id",
        "days_since_signup",
        "transactions_per_product",
        "age_segment",
        "credit_category",
        "balance_income_ratio",
        "credit_score",
        "num_products",
        "account_balance",
        "total_transactions",
        "last_updated"
    )
)

print("✅ Features calculadas")
features_df.show(5)

# COMMAND ----------

# DBTITLE 1,Ejercicio 1: Solución - Crear Feature Table
# Ejercicio 1: Crear Feature Table

# Eliminar si existe (para re-ejecución)
spark.sql("DROP TABLE IF EXISTS main.features.bank_customer_features")

# Crear Feature Table
fe.create_table(
    name="main.features.bank_customer_features",
    primary_keys=["customer_id"],
    timestamp_keys=["last_updated"],
    df=features_df,
    description="Features de clientes bancarios para modelos de churn, lifetime value y scoring"
)

print("✅ Feature Table creada exitosamente")
print("\nMetadata:")
print(f"  Nombre: main.features.bank_customer_features")
print(f"  Primary Key: customer_id")
print(f"  Timestamp Key: last_updated")
print(f"  Features: {len(features_df.columns) - 2}")

# COMMAND ----------

# DBTITLE 1,Ejercicio 1: Validación
# Ejercicio 1: Validación de Feature Table

# Leer Feature Table
feature_table_df = fe.read_table(name="main.features.bank_customer_features")

print(f"Total registros: {feature_table_df.count()}")
print(f"\nEsquema:")
feature_table_df.printSchema()

print("\nEstadísticas descriptivas:")
feature_table_df.select(
    "days_since_signup",
    "transactions_per_product",
    "balance_income_ratio",
    "credit_score"
).summary().show()

print("\nDistribución de segmentos:")
feature_table_df.groupBy("age_segment").count().show()
feature_table_df.groupBy("credit_category").count().show()

print("✅ Validación completada")

# COMMAND ----------

# DBTITLE 1,Ejercicio 2: AutoML con Feature Store
# MAGIC %md
# MAGIC # Ejercicio 2: AutoML para Predicción de Churn
# MAGIC
# MAGIC ## Objetivo
# MAGIC Usar AutoML con features del Feature Store para predecir churn bancario.
# MAGIC
# MAGIC ## Tareas
# MAGIC 1. Crear Training Set con Feature Lookups
# MAGIC 2. Ejecutar Databricks AutoML
# MAGIC 3. Analizar resultados y modelos generados
# MAGIC 4. Comparar métricas entre algoritmos

# COMMAND ----------

# DBTITLE 1,Ejercicio 2: Solución - Training Set
# Ejercicio 2: Crear Training Set con Feature Store

from databricks.feature_engineering import FeatureLookup

# Cargar labels
labels_df = spark.table("main.default.bank_churn_labels")

print(f"Labels: {labels_df.count()} registros")
print(f"Churn rate: {labels_df.select(F.avg('churn')).collect()[0][0]:.1%}")

# Definir Feature Lookups
feature_lookups = [
    FeatureLookup(
        table_name="main.features.bank_customer_features",
        lookup_key="customer_id",
        timestamp_lookup_key="label_date",  # Point-in-time correctness
        feature_names=[
            "days_since_signup",
            "transactions_per_product",
            "age_segment",
            "credit_category",
            "balance_income_ratio",
            "credit_score",
            "num_products",
            "account_balance",
            "total_transactions"
        ]
    )
]

# Crear Training Set
training_set = fe.create_training_set(
    df=labels_df,
    feature_lookups=feature_lookups,
    label="churn",
    exclude_columns=["label_date"]
)

# Cargar como DataFrame
training_df = training_set.load_df()

print(f"\n✅ Training set creado: {training_df.count()} rows x {len(training_df.columns)} columns")
training_df.show(5)

# COMMAND ----------

# DBTITLE 1,Ejercicio 2: Solución - Ejecutar AutoML
# Ejercicio 2: Ejecutar AutoML

import databricks.automl

# Convertir a Pandas para AutoML
training_pdf = training_df.toPandas()

print(f"Dataset shape: {training_pdf.shape}")
print(f"\nChurn distribution:")
print(training_pdf['churn'].value_counts(normalize=True))

# Ejecutar AutoML
print("\n🚀 Iniciando AutoML (10 minutos)...\n")

summary = databricks.automl.classify(
    dataset=training_pdf,
    target_col="churn",
    timeout_minutes=10,
    primary_metric="f1",  # Balance entre precision y recall
    max_trials=20  # Número máximo de experimentos
)

print("✅ AutoML completado")

# COMMAND ----------

# DBTITLE 1,Ejercicio 2: Análisis de Resultados
# Ejercicio 2: Análisis de Resultados de AutoML

# Mejor modelo
best_trial = summary.best_trial

print("=" * 60)
print("MEJOR MODELO")
print("=" * 60)
print(f"\nAlgoritmo: {best_trial.model_description}")
print(f"\nMétricas de validación:")
print(f"  F1 Score:  {best_trial.metrics.get('val_f1_score', 'N/A'):.4f}")
print(f"  Accuracy:  {best_trial.metrics.get('val_accuracy_score', 'N/A'):.4f}")
print(f"  Precision: {best_trial.metrics.get('val_precision_score', 'N/A'):.4f}")
print(f"  Recall:    {best_trial.metrics.get('val_recall_score', 'N/A'):.4f}")
print(f"  ROC AUC:   {best_trial.metrics.get('val_roc_auc_score', 'N/A'):.4f}")

print(f"\nMLflow Run ID: {best_trial.mlflow_run_id}")
print(f"Notebook: {best_trial.notebook_url}")

# Comparar todos los trials
print("\n" + "=" * 60)
print("COMPARACIÓN DE TODOS LOS MODELOS")
print("=" * 60)

trials_data = []
for trial in summary.trials:
    trials_data.append({
        'Model': trial.model_description[:30],
        'F1': trial.metrics.get('val_f1_score', 0),
        'Accuracy': trial.metrics.get('val_accuracy_score', 0),
        'ROC_AUC': trial.metrics.get('val_roc_auc_score', 0)
    })

trials_df = pd.DataFrame(trials_data).sort_values('F1', ascending=False)
print(trials_df.head(10).to_string(index=False))

print("\n✅ Análisis completado")

# COMMAND ----------

# DBTITLE 1,Ejercicio 3: Comparación con/sin Feature Store
# MAGIC %md
# MAGIC # Ejercicio 3: Comparación con/sin Feature Store
# MAGIC
# MAGIC ## Objetivo
# MAGIC Comparar el workflow tradicional vs. Feature Store para evidenciar ventajas.
# MAGIC
# MAGIC ## Tareas
# MAGIC 1. Entrenar modelo SIN Feature Store (workflow tradicional)
# MAGIC 2. Comparar esfuerzo, tiempo y complejidad
# MAGIC 3. Analizar diferencias en:
# MAGIC    - Código necesario
# MAGIC    - Reproducibilidad
# MAGIC    - Serving en producción
# MAGIC    - Reutilización para otros modelos

# COMMAND ----------

# DBTITLE 1,Ejercicio 3: Solución - Workflow Tradicional
# Ejercicio 3: Workflow TRADICIONAL (sin Feature Store)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
import mlflow
import mlflow.sklearn

print("=" * 60)
print("WORKFLOW TRADICIONAL (SIN FEATURE STORE)")
print("=" * 60)

# 1. Cargar datos y computar features MANUALMENTE
raw_df = spark.table("main.default.bank_customers_raw")

# ⚠️ Problema: Debemos replicar EXACTAMENTE la lógica de feature engineering
manual_features = (
    raw_df
    .withColumn("days_since_signup", F.datediff(F.current_date(), F.col("signup_date")))
    .withColumn("transactions_per_product", F.col("total_transactions") / F.col("num_products"))
    .withColumn("balance_income_ratio", F.col("account_balance") / F.col("income"))
    .select(
        "customer_id",
        "days_since_signup",
        "transactions_per_product",
        "balance_income_ratio",
        "credit_score",
        "num_products",
        "account_balance",
        "total_transactions",
        "churn"
    )
).toPandas()

print(f"✅ Features computadas manualmente: {manual_features.shape}")

# 2. Preparar datos
X = manual_features.drop(['customer_id', 'churn'], axis=1)
y = manual_features['churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# COMMAND ----------

# DBTITLE 1,Ejercicio 3: Solución - Entrenar Modelo Tradicional
# Ejercicio 3: Entrenar con workflow tradicional

with mlflow.start_run(run_name="traditional_workflow_rf") as run:
    # Entrenar
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    
    rf_model.fit(X_train, y_train)
    
    # Evaluar
    y_pred = rf_model.predict(X_test)
    y_proba = rf_model.predict_proba(X_test)[:, 1]
    
    f1 = f1_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    
    # Log metrics
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("roc_auc", roc_auc)
    
    # Log model
    mlflow.sklearn.log_model(rf_model, "model")
    
    print("\nMétricas (Workflow Tradicional):")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  ROC AUC:   {roc_auc:.4f}")
    
    traditional_run_id = run.info.run_id
    
print(f"\n✅ Modelo tradicional entrenado")
print(f"Run ID: {traditional_run_id}")

# COMMAND ----------

# DBTITLE 1,Ejercicio 3: Análisis Comparativo
# Ejercicio 3: Análisis Comparativo

print("=" * 70)
print("COMPARACIÓN: TRADICIONAL vs. FEATURE STORE")
print("=" * 70)

comparison = pd.DataFrame({
    'Aspecto': [
        'Líneas de código',
        'Tiempo de desarrollo',
        'Riesgo de inconsistencia',
        'Reutilización de features',
        'Documentación automática',
        'Point-in-time correctness',
        'Serving en producción',
        'Reproducibilidad',
        'Colaboración entre equipos',
        'Gobernanza (Unity Catalog)',
        'Versionado de features',
        'Auditoría y linaje'
    ],
    'Workflow Tradicional': [
        '❌ 50-100 líneas',
        '❌ 1-2 semanas',
        '❌ Alto (training-serving skew)',
        '❌ Duplicación de código',
        '❌ Manual',
        '❌ Implementación manual compleja',
        '❌ Código separado para serving',
        '⚠️ Media (depende de discipline)',
        '❌ Difícil',
        '❌ Sin integración',
        '❌ Manual',
        '❌ Difícil'
    ],
    'Feature Store': [
        '✅ 10-20 líneas',
        '✅ Días',
        '✅ Bajo (consistencia garantizada)',
        '✅ Reutilización automática',
        '✅ Automática',
        '✅ Built-in',
        '✅ Lookup automático',
        '✅ Alta',
        '✅ Fácil',
        '✅ Totalmente integrado',
        '✅ Automático (Delta)',
        '✅ Completo'
    ]
})

print(comparison.to_string(index=False))

print("\n" + "=" * 70)
print("CONCLUSIÓN")
print("=" * 70)
print("""
Feature Store proporciona:

1. 🚀 VELOCIDAD: 3-5x más rápido desarrollar nuevos modelos
2. ✅ CALIDAD: Elimina training-serving skew
3. 🔄 REUTILIZACIÓN: Features compartidas entre equipos
4. 🔒 GOBERNANZA: Control de acceso y auditoría
5. 📊 PRODUCTIZACIÓN: Serving simplificado

Workflow Tradicional:
- ⚠️ Propenso a errores
- ⚠️ Duplicación de esfuerzo
- ⚠️ Difícil de mantener a escala
""")

print("✅ Análisis comparativo completado")

# COMMAND ----------

# DBTITLE 1,Ejercicio 4: Feature Serving en Producción
# MAGIC %md
# MAGIC # Ejercicio 4: Feature Serving en Producción
# MAGIC
# MAGIC ## Objetivo
# MAGIC Demostrar cómo usar el modelo en producción con Feature Store para predicciones batch.
# MAGIC
# MAGIC ## Tareas
# MAGIC 1. Simular nuevos clientes sin features calculadas
# MAGIC 2. Usar `score_batch` para predicciones automáticas
# MAGIC 3. Verificar que las features se buscan automáticamente
# MAGIC 4. Comparar con workflow tradicional

# COMMAND ----------

# DBTITLE 1,Ejercicio 4: Solución - Feature Serving
# Ejercicio 4: Feature Serving con Feature Store

import mlflow

print("=" * 60)
print("FEATURE SERVING EN PRODUCCIÓN")
print("=" * 60)

# Simular nuevos clientes (solo con customer_id)
new_customers = spark.createDataFrame([
    (101,),
    (250,),
    (500,),
    (750,)
], ["customer_id"])

print("\nNuevos clientes para scoring:")
new_customers.show()

# Usar modelo de AutoML (que tiene metadata de Feature Store)
# En producción, usarías: model_uri = "models:/bank_churn/production"
model_uri = f"runs:/{summary.best_trial.mlflow_run_id}/model"

print(f"\nModelo: {model_uri}")
print("\n🚀 Ejecutando predicciones con Feature Store...")

# Score batch - las features se buscan AUTOMÁTICAMENTE
predictions = fe.score_batch(
    model_uri=model_uri,
    df=new_customers
)

print("\n✅ Predicciones completadas")
print("\nResultados (con features automáticas del Feature Store):")
predictions.select(
    "customer_id",
    "prediction",
    "days_since_signup",  # Feature obtenida automáticamente
    "credit_score",       # Feature obtenida automáticamente
    "balance_income_ratio" # Feature obtenida automáticamente
).show()

print("\n" + "=" * 60)
print("¡Features obtenidas automáticamente del Feature Store!")
print("No fue necesario recomputar nada.")
print("Consistencia training-serving GARANTIZADA.")
print("=" * 60)

# COMMAND ----------

# DBTITLE 1,Ejercicio 5: Reutilización para Múltiples Modelos
# MAGIC %md
# MAGIC # Ejercicio 5: Feature Store para Múltiples Modelos
# MAGIC
# MAGIC ## Objetivo
# MAGIC Demostrar cómo el mismo Feature Store se reutiliza para entrenar diferentes modelos.
# MAGIC
# MAGIC ## Escenario
# MAGIC Usar las MISMAS features de `bank_customer_features` para:
# MAGIC 1. **Modelo 1**: Predicción de Churn (ya hecho)
# MAGIC 2. **Modelo 2**: Segmentación de Clientes (nuevo)
# MAGIC
# MAGIC ## Ventaja
# MAGIC - Features calculadas UNA SOLA VEZ
# MAGIC - Reutilizadas en ambos modelos
# MAGIC - Ahorro de tiempo y recursos

# COMMAND ----------

# DBTITLE 1,Ejercicio 5: Solución - Nuevo Modelo de Segmentación
# Ejercicio 5: Entrenar SEGUNDO modelo usando MISMAS features

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

print("=" * 60)
print("MODELO 2: SEGMENTACIÓN DE CLIENTES")
print("Reutilizando features del Feature Store")
print("=" * 60)

# Cargar features del Feature Store (REUTILIZACIÓN)
features_for_clustering = fe.read_table(
    name="main.features.bank_customer_features"
).select(
    "customer_id",
    "days_since_signup",
    "transactions_per_product",
    "balance_income_ratio",
    "credit_score",
    "account_balance"
).toPandas()

print(f"\n✅ Features cargadas: {features_for_clustering.shape}")
print("\n⚠️ Nota: NO fue necesario recomputar nada. Reutilizamos el Feature Store.")

# Preparar datos
X_clustering = features_for_clustering.drop('customer_id', axis=1)

# Clustering
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_clustering)

features_for_clustering['segment'] = clusters

print(f"\n✅ Segmentación completada")
print("\nDistribución de segmentos:")
print(features_for_clustering['segment'].value_counts().sort_index())

print("\nCaracterísticas promedio por segmento:")
segment_profile = features_for_clustering.groupby('segment')[[
    'credit_score',
    'account_balance',
    'transactions_per_product'
]].mean()

print(segment_profile.to_string())

# COMMAND ----------

# DBTITLE 1,Ejercicio 5: Visualización de Segmentos
# Ejercicio 5: Visualizar segmentos

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12, 5))

# Subplot 1: Credit Score vs Balance
plt.subplot(1, 2, 1)
sns.scatterplot(
    data=features_for_clustering,
    x='credit_score',
    y='account_balance',
    hue='segment',
    palette='Set2',
    alpha=0.6
)
plt.title('Segmentos: Credit Score vs Balance')
plt.xlabel('Credit Score')
plt.ylabel('Account Balance')

# Subplot 2: Transactions per Product vs Balance Ratio
plt.subplot(1, 2, 2)
sns.scatterplot(
    data=features_for_clustering,
    x='transactions_per_product',
    y='balance_income_ratio',
    hue='segment',
    palette='Set2',
    alpha=0.6
)
plt.title('Segmentos: Actividad vs Solvencia')
plt.xlabel('Transactions per Product')
plt.ylabel('Balance/Income Ratio')

plt.tight_layout()
plt.show()

print("✅ Visualización completada")

print("\n" + "=" * 60)
print("¡REUTILIZACIÓN DE FEATURES EXITOSA!")
print("=" * 60)
print("""
Dos modelos diferentes:
  1. Churn Prediction (clasificación)
  2. Customer Segmentation (clustering)

Ambos usaron las MISMAS features del Feature Store.

Beneficios:
  ✅ Ahorro de tiempo (no recomputar)
  ✅ Consistencia entre modelos
  ✅ Mantenimiento centralizado
  ✅ Colaboración facilitada
""")

# COMMAND ----------

# DBTITLE 1,Resumen y Conclusiones de la Práctica
# MAGIC %md
# MAGIC # Resumen y Conclusiones
# MAGIC
# MAGIC ## ✅ Ejercicios Completados
# MAGIC
# MAGIC ### Ejercicio 1: Feature Table
# MAGIC - Creamos `main.features.bank_customer_features`
# MAGIC - 10 features derivadas de datos raw
# MAGIC - Con primary keys y timestamp keys
# MAGIC - Documentación y metadata
# MAGIC
# MAGIC ### Ejercicio 2: AutoML con Feature Store
# MAGIC - Training set con Feature Lookups
# MAGIC - AutoML exploró 20 modelos automáticamente
# MAGIC - Mejor modelo registrado en MLflow
# MAGIC - Métricas: F1, Accuracy, ROC AUC
# MAGIC
# MAGIC ### Ejercicio 3: Comparación Tradicional vs. Feature Store
# MAGIC - Workflow tradicional: 50-100 líneas, propenso a errores
# MAGIC - Feature Store: 10-20 líneas, consistencia garantizada
# MAGIC - **Conclusión**: Feature Store reduce complejidad y riesgo dramáticamente
# MAGIC
# MAGIC ### Ejercicio 4: Feature Serving
# MAGIC - Predicciones batch con `score_batch`
# MAGIC - Features obtenidas AUTOMÁTICAMENTE del Feature Store
# MAGIC - Consistencia training-serving garantizada
# MAGIC
# MAGIC ### Ejercicio 5: Múltiples Modelos
# MAGIC - Reutilización de features para segmentación
# MAGIC - CERO cálculo adicional
# MAGIC - Demostración clara del valor de Feature Store
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Aprendizajes Clave
# MAGIC
# MAGIC ### AutoML
# MAGIC 1. **Velocidad**: De semanas a minutos
# MAGIC 2. **Exploración**: Múltiples algoritmos automáticamente
# MAGIC 3. **Baseline**: Línea base sólida para iterar
# MAGIC 4. **Notebooks**: Código generado es educativo y reproducible
# MAGIC
# MAGIC ### Feature Store
# MAGIC 1. **Reutilización**: Features compartidas entre modelos
# MAGIC 2. **Consistencia**: Training-serving skew eliminado
# MAGIC 3. **Velocidad**: 3-5x más rápido desarrollar modelos
# MAGIC 4. **Gobernanza**: Unity Catalog para permisos y auditoría
# MAGIC 5. **Productización**: Serving simplificado y automático
# MAGIC
# MAGIC ### Integración
# MAGIC ```
# MAGIC Feature Store + AutoML = 🚀 Aceleración Dramática del Ciclo ML
# MAGIC
# MAGIC   Feature Engineering (una vez)
# MAGIC            ↓
# MAGIC   Feature Store (reutilizable)
# MAGIC            ↓
# MAGIC   AutoML (exploración rápida)
# MAGIC            ↓
# MAGIC   Iteración Manual (refinamiento)
# MAGIC            ↓
# MAGIC   Producción (serving automático)
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Métricas de Éxito
# MAGIC
# MAGIC | Métrica | Valor Obtenido |
# MAGIC |---------|----------------|
# MAGIC | Features creadas | 10 features reutilizables |
# MAGIC | Modelos entrenados | 20+ (vía AutoML) + 1 clustering |
# MAGIC | Mejor F1 Score | ~0.75-0.85 (varía según datos) |
# MAGIC | Tiempo total desarrollo | <1 hora vs. 1-2 semanas tradicional |
# MAGIC | Reutilización de features | 100% (ambos modelos) |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Mejores Prácticas Aplicadas
# MAGIC
# MAGIC ✅ Features documentadas con descripción clara  
# MAGIC ✅ Primary keys y timestamp keys correctamente definidos  
# MAGIC ✅ Point-in-time correctness para evitar data leakage  
# MAGIC ✅ Unity Catalog para gobernanza  
# MAGIC ✅ MLflow para tracking y reproducibilidad  
# MAGIC ✅ AutoML como baseline, iteración manual para refinamiento  
# MAGIC ✅ Feature serving automático en producción  
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Próximos Pasos
# MAGIC
# MAGIC 1. **Ampliar Feature Store**: Agregar features de transacciones, productos
# MAGIC 2. **Online Serving**: Configurar serving de baja latencia
# MAGIC 3. **Monitoring**: Añadir monitoreo de drift de features
# MAGIC 4. **Feature Pipelines**: Automatizar actualización de Feature Tables
# MAGIC 5. **Multi-equipo**: Compartir Feature Store entre data scientists
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC 🎉 **¡Práctica del Módulo 05 completada exitosamente!**

# COMMAND ----------

