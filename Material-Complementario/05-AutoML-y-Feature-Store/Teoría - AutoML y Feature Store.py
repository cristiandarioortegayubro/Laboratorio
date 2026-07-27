# Databricks notebook source
# DBTITLE 1,Módulo 05 - AutoML y Feature Store
# MAGIC %md
# MAGIC # Módulo 05: AutoML y Feature Store
# MAGIC
# MAGIC ## Contenido del Módulo
# MAGIC
# MAGIC 1. **Introducción a AutoML**
# MAGIC    - ¿Qué es AutoML?
# MAGIC    - Ventajas y limitaciones
# MAGIC    - AutoML en Databricks
# MAGIC
# MAGIC 2. **Feature Store**
# MAGIC    - Motivación y conceptos clave
# MAGIC    - Arquitectura de Feature Store
# MAGIC    - Feature Tables y Feature Serving
# MAGIC
# MAGIC 3. **Integración AutoML + Feature Store**
# MAGIC    - Flujo de trabajo completo
# MAGIC    - Mejores prácticas
# MAGIC    - Casos de uso
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC **Objetivo del módulo**: Comprender las herramientas de automatización y gestión de features que aceleran el desarrollo de modelos ML en producción.

# COMMAND ----------

# DBTITLE 1,1. Introducción a AutoML
# MAGIC %md
# MAGIC # 1. Introducción a AutoML
# MAGIC
# MAGIC ## ¿Qué es AutoML?
# MAGIC
# MAGIC **AutoML (Automated Machine Learning)** automatiza tareas repetitivas del proceso de ML:
# MAGIC - Preprocesamiento de datos
# MAGIC - Selección de algoritmos
# MAGIC - Ingeniería de características
# MAGIC - Ajuste de hiperparámetros
# MAGIC - Validación cruzada
# MAGIC - Selección del modelo óptimo
# MAGIC
# MAGIC ## Ventajas de AutoML
# MAGIC
# MAGIC | Ventaja | Descripción |
# MAGIC |---------|-------------|
# MAGIC | **Velocidad** | Reduce el tiempo de desarrollo de semanas a minutos |
# MAGIC | **Democratización** | Permite a usuarios no expertos crear modelos de calidad |
# MAGIC | **Benchmark** | Proporciona una línea base sólida para comparar |
# MAGIC | **Exploración** | Prueba múltiples algoritmos y configuraciones automáticamente |
# MAGIC | **Reproducibilidad** | Genera código y notebooks reproducibles |
# MAGIC
# MAGIC ## Limitaciones de AutoML
# MAGIC
# MAGIC | Limitación | Descripción |
# MAGIC |------------|-------------|
# MAGIC | **Caja negra** | Menor control sobre el proceso de modelado |
# MAGIC | **Datos específicos** | Requiere datos bien preparados y representativos |
# MAGIC | **Interpretabilidad** | Puede sacrificar interpretabilidad por performance |
# MAGIC | **Recursos** | Consume recursos computacionales significativos |
# MAGIC | **Dominio** | No reemplaza el conocimiento del dominio del negocio |
# MAGIC
# MAGIC ## AutoML en Databricks
# MAGIC
# MAGIC Databricks AutoML ofrece:
# MAGIC - **Interfaz gráfica** para usuarios no técnicos
# MAGIC - **API de Python** para integración en workflows
# MAGIC - **Notebooks generados** con código explicativo
# MAGIC - **Registro automático en MLflow** para tracking
# MAGIC - **Soporte para**:
# MAGIC   - Clasificación (binaria y multiclase)
# MAGIC   - Regresión
# MAGIC   - Forecasting (series temporales)
# MAGIC
# MAGIC ```python
# MAGIC # Ejemplo básico de uso
# MAGIC import databricks.automl
# MAGIC
# MAGIC summary = databricks.automl.classify(
# MAGIC     dataset=train_df,
# MAGIC     target_col="target",
# MAGIC     timeout_minutes=15
# MAGIC )
# MAGIC ```

# COMMAND ----------

# DBTITLE 1,2. Feature Store - Conceptos Clave
# MAGIC %md
# MAGIC # 2. Feature Store - Conceptos Clave
# MAGIC
# MAGIC ## ¿Qué es Feature Store?
# MAGIC
# MAGIC **Feature Store** es un repositorio centralizado para features de ML que:
# MAGIC - **Almacena** features reutilizables
# MAGIC - **Versionado** features con control de cambios
# MAGIC - **Comparte** features entre equipos
# MAGIC - **Sirve** features en tiempo real y batch
# MAGIC - **Garantiza consistencia** entre entrenamiento e inferencia
# MAGIC
# MAGIC ## Motivación
# MAGIC
# MAGIC ### Problemas sin Feature Store
# MAGIC
# MAGIC | Problema | Impacto |
# MAGIC |----------|--------|
# MAGIC | **Duplicación** | Mismas features calculadas múltiples veces |
# MAGIC | **Inconsistencia** | Diferencias entre entrenamiento y producción |
# MAGIC | **Descubrimiento** | Difícil encontrar features existentes |
# MAGIC | **Documentación** | Features mal documentadas o sin documentar |
# MAGIC | **Gobernanza** | Difícil controlar acceso y calidad |
# MAGIC
# MAGIC ### Solución con Feature Store
# MAGIC
# MAGIC ```
# MAGIC ┌─────────────────────────────────────────────┐
# MAGIC │         Feature Store (Delta Tables)        │
# MAGIC ├─────────────────────────────────────────────┤
# MAGIC │  Feature Table 1: customer_features         │
# MAGIC │  Feature Table 2: transaction_features      │
# MAGIC │  Feature Table 3: product_features          │
# MAGIC └─────────────────────────────────────────────┘
# MAGIC          ↓                    ↓
# MAGIC     Training              Serving
# MAGIC     (Batch)            (Real-time)
# MAGIC ```
# MAGIC
# MAGIC ## Componentes Principales
# MAGIC
# MAGIC ### 1. Feature Table
# MAGIC
# MAGIC - **Tabla Delta** en Unity Catalog
# MAGIC - Contiene:
# MAGIC   - **Primary keys**: Identificadores únicos
# MAGIC   - **Features**: Columnas con features
# MAGIC   - **Timestamp** (opcional): Para point-in-time lookups
# MAGIC   - **Metadata**: Descripción, autor, versión
# MAGIC
# MAGIC ### 2. Feature Spec
# MAGIC
# MAGIC - Define qué features usar del Feature Store
# MAGIC - Especifica:
# MAGIC   - Tabla origen
# MAGIC   - Columnas a incluir
# MAGIC   - Claves de join
# MAGIC
# MAGIC ### 3. Training Set
# MAGIC
# MAGIC - Combina labels con features del Feature Store
# MAGIC - Garantiza consistencia temporal (point-in-time correctness)
# MAGIC
# MAGIC ### 4. Feature Serving
# MAGIC
# MAGIC - **Batch**: Para predicciones en lote
# MAGIC - **Online**: Para inferencia en tiempo real (low-latency)

# COMMAND ----------

# DBTITLE 1,3. Arquitectura de Feature Store
# MAGIC %md
# MAGIC # 3. Arquitectura de Feature Store
# MAGIC
# MAGIC ## Flujo Completo
# MAGIC
# MAGIC ```
# MAGIC ┌─────────────┐
# MAGIC │ Raw Data    │
# MAGIC └──────┬──────┘
# MAGIC        │
# MAGIC        ↓
# MAGIC ┌─────────────────────┐
# MAGIC │ Feature Engineering │ ← Spark / Pandas
# MAGIC └──────┬──────────────┘
# MAGIC        │
# MAGIC        ↓
# MAGIC ┌─────────────────────┐
# MAGIC │  Feature Store      │ ← Delta Tables
# MAGIC │  (Unity Catalog)    │
# MAGIC └──────┬──────────────┘
# MAGIC        │
# MAGIC        ├──────────────┐
# MAGIC        ↓              ↓
# MAGIC   ┌─────────┐   ┌──────────┐
# MAGIC   │Training │   │ Serving  │
# MAGIC   └─────────┘   └──────────┘
# MAGIC        │              │
# MAGIC        ↓              ↓
# MAGIC   ┌─────────┐   ┌──────────┐
# MAGIC   │  Model  │   │Real-time │
# MAGIC   │ Training│   │Inference │
# MAGIC   └─────────┘   └──────────┘
# MAGIC ```
# MAGIC
# MAGIC ## Ventajas de la Arquitectura
# MAGIC
# MAGIC 1. **Single Source of Truth**
# MAGIC    - Features definidas una sola vez
# MAGIC    - Usadas en múltiples modelos
# MAGIC
# MAGIC 2. **Consistencia Training-Serving**
# MAGIC    - Misma lógica en entrenamiento y producción
# MAGIC    - Elimina "training-serving skew"
# MAGIC
# MAGIC 3. **Point-in-Time Correctness**
# MAGIC    - Features históricas correctas para cada timestamp
# MAGIC    - Evita data leakage
# MAGIC
# MAGIC 4. **Reutilización**
# MAGIC    - Features compartidas entre equipos
# MAGIC    - Reduce tiempo de desarrollo
# MAGIC
# MAGIC 5. **Gobernanza**
# MAGIC    - Control de acceso vía Unity Catalog
# MAGIC    - Auditoría y linaje de datos
# MAGIC
# MAGIC ## Tecnologías en Databricks
# MAGIC
# MAGIC - **Storage**: Delta Lake (ACID, versionado, time travel)
# MAGIC - **Catalog**: Unity Catalog (gobernanza, permisos)
# MAGIC - **Compute**: Spark / Pandas on Spark
# MAGIC - **API**: Python Feature Store API
# MAGIC - **Integration**: MLflow para tracking

# COMMAND ----------

# DBTITLE 1,4. Creación de Feature Tables
# 4. Creación de Feature Tables

from databricks.feature_engineering import FeatureEngineeringClient
from pyspark.sql import functions as F

# Inicializar cliente
fe = FeatureEngineeringClient()

# Ejemplo: Crear features de clientes
customer_features = (
    spark.table("main.default.customers")
    .withColumn("days_since_signup", F.datediff(F.current_date(), F.col("signup_date")))
    .withColumn("total_orders", F.col("order_count"))
    .withColumn("avg_order_value", F.col("total_spent") / F.col("order_count"))
    .select(
        "customer_id",  # Primary key
        "days_since_signup",
        "total_orders",
        "avg_order_value",
        "preferred_category",
        "last_purchase_date"  # Timestamp column
    )
)

# Crear Feature Table
fe.create_table(
    name="main.features.customer_features",
    primary_keys=["customer_id"],
    timestamp_keys=["last_purchase_date"],
    df=customer_features,
    description="Customer behavioral features for ML models"
)

print("✅ Feature Table creada exitosamente")

# COMMAND ----------

# DBTITLE 1,5. Actualización de Feature Tables
# 5. Actualización de Feature Tables

# Opción 1: Sobrescribir completamente
fe.write_table(
    name="main.features.customer_features",
    df=customer_features_updated,
    mode="overwrite"
)

# Opción 2: Merge (upsert)
# Actualiza registros existentes e inserta nuevos
fe.write_table(
    name="main.features.customer_features",
    df=customer_features_new,
    mode="merge"
)

print("✅ Feature Table actualizada")

# Leer Feature Table
features_df = fe.read_table(name="main.features.customer_features")
print(f"Total features: {features_df.count()}")
features_df.display()

# COMMAND ----------

# DBTITLE 1,6. Uso de Features en Training
# 6. Uso de Features en Training

from databricks.feature_engineering import FeatureLookup

# Dataset con labels
labels_df = spark.table("main.default.training_labels").select(
    "customer_id",
    "churn",  # Target variable
    "label_date"  # Timestamp para point-in-time lookup
)

# Definir qué features usar
feature_lookups = [
    FeatureLookup(
        table_name="main.features.customer_features",
        lookup_key="customer_id",
        timestamp_lookup_key="label_date",  # Point-in-time correctness
        feature_names=[
            "days_since_signup",
            "total_orders",
            "avg_order_value",
            "preferred_category"
        ]
    )
]

# Crear training set
training_set = fe.create_training_set(
    df=labels_df,
    feature_lookups=feature_lookups,
    label="churn",
    exclude_columns=["label_date"]
)

# Cargar como DataFrame
training_df = training_set.load_df()
print(f"Training set shape: {training_df.count()} rows")
training_df.display()

# COMMAND ----------

# DBTITLE 1,7. Integración con AutoML
# 7. Integración AutoML + Feature Store

import databricks.automl

# Opción 1: AutoML con Training Set del Feature Store
training_df_pandas = training_set.load_df().toPandas()

summary = databricks.automl.classify(
    dataset=training_df_pandas,
    target_col="churn",
    timeout_minutes=10,
    primary_metric="f1"
)

print(f"Mejor modelo: {summary.best_trial.model_description}")
print(f"F1 Score: {summary.best_trial.metrics['val_f1_score']:.4f}")

# El modelo queda registrado en MLflow con referencia al Feature Store
# Esto permite:
# 1. Reproducibilidad: Features versionadas
# 2. Serving: Lookup automático de features en inferencia
# 3. Lineage: Trazabilidad desde features hasta predicciones

# COMMAND ----------

# DBTITLE 1,8. Feature Serving en Producción
# 8. Feature Serving en Producción

import mlflow

# Cargar modelo registrado (incluye Feature Store metadata)
model_uri = "models:/churn_prediction/production"
model = mlflow.pyfunc.load_model(model_uri)

# Predicción Batch
# Solo necesitas las primary keys, las features se buscan automáticamente
batch_df = spark.createDataFrame([
    (12345,),
    (67890,)
], ["customer_id"])

predictions = fe.score_batch(
    model_uri=model_uri,
    df=batch_df
)

predictions.display()

# Las features se obtienen automáticamente del Feature Store
# Garantiza consistencia entre training y serving

# COMMAND ----------

# DBTITLE 1,9. Mejores Prácticas
# MAGIC %md
# MAGIC # 9. Mejores Prácticas
# MAGIC
# MAGIC ## Diseño de Feature Tables
# MAGIC
# MAGIC ### ✅ Buenas Prácticas
# MAGIC
# MAGIC | Práctica | Razón |
# MAGIC |----------|-------|
# MAGIC | **Una entidad por tabla** | Customer features, product features separados |
# MAGIC | **Primary keys claros** | IDs únicos y estables |
# MAGIC | **Timestamp keys** | Para point-in-time correctness |
# MAGIC | **Documentación** | Descripción detallada de cada feature |
# MAGIC | **Versionado** | Usar Delta time travel para auditoría |
# MAGIC | **Tipos de datos consistentes** | Evitar casting innecesario |
# MAGIC
# MAGIC ### ❌ Anti-patrones
# MAGIC
# MAGIC | Anti-patrón | Problema |
# MAGIC |-------------|----------|
# MAGIC | Features de múltiples entidades | Complica joins y actualizaciones |
# MAGIC | Sin timestamp keys | No permite lookups temporales correctos |
# MAGIC | Features calculadas en serving | Inconsistencia training-serving |
# MAGIC | Sin documentación | Features no reutilizables |
# MAGIC | Actualización manual | Propenso a errores y desactualización |
# MAGIC
# MAGIC ## Workflow Recomendado
# MAGIC
# MAGIC ```python
# MAGIC # 1. Ingeniería de Features (una vez)
# MAGIC features = compute_features(raw_data)
# MAGIC fe.create_table(name="main.features.my_features", df=features, ...)
# MAGIC
# MAGIC # 2. Training (múltiples modelos)
# MAGIC training_set = fe.create_training_set(labels, feature_lookups)
# MAGIC summary = databricks.automl.classify(training_set.load_df(), ...)
# MAGIC
# MAGIC # 3. Serving (automático)
# MAGIC predictions = fe.score_batch(model_uri, new_data)
# MAGIC ```
# MAGIC
# MAGIC ## AutoML - Mejores Prácticas
# MAGIC
# MAGIC | Práctica | Descripción |
# MAGIC |----------|-------------|
# MAGIC | **Datos limpios** | AutoML no hace limpieza avanzada |
# MAGIC | **Features significativas** | Feature engineering previo mejora resultados |
# MAGIC | **Timeout adecuado** | Más tiempo = mejor exploración (15-60 min) |
# MAGIC | **Métrica correcta** | Alineada con objetivo de negocio |
# MAGIC | **Revisar notebooks** | Aprender de las técnicas generadas |
# MAGIC | **Benchmark** | Usar como baseline, iterar manualmente |
# MAGIC
# MAGIC ## Cuándo Usar Cada Herramienta
# MAGIC
# MAGIC ### Usar AutoML cuando:
# MAGIC - Necesitas un modelo rápidamente
# MAGIC - Quieres establecer un baseline
# MAGIC - No tienes experiencia profunda en ML
# MAGIC - El problema es estándar (clasificación/regresión)
# MAGIC
# MAGIC ### Usar Feature Store cuando:
# MAGIC - Múltiples modelos usan las mismas features
# MAGIC - Necesitas consistencia training-serving
# MAGIC - Trabajas en equipo
# MAGIC - Modelos en producción
# MAGIC
# MAGIC ### Combinar ambos cuando:
# MAGIC - Quieres acelerar desarrollo **Y** mantener calidad
# MAGIC - Múltiples científicos de datos colaboran
# MAGIC - Ciclo de vida completo: prototipo → producción

# COMMAND ----------

# DBTITLE 1,10. Comparación: Workflow Tradicional vs. Feature Store
# MAGIC %md
# MAGIC # 10. Comparación de Workflows
# MAGIC
# MAGIC ## Workflow Tradicional (sin Feature Store)
# MAGIC
# MAGIC ```python
# MAGIC # Modelo 1: Churn
# MAGIC features_churn = compute_customer_features(raw_data)  # ← Código duplicado
# MAGIC train_model_churn(features_churn)
# MAGIC
# MAGIC # Modelo 2: Lifetime Value
# MAGIC features_ltv = compute_customer_features(raw_data)    # ← Mismo cálculo otra vez
# MAGIC train_model_ltv(features_ltv)
# MAGIC
# MAGIC # Producción: Recomputar features en serving
# MAGIC def predict_churn(customer_id):
# MAGIC     features = compute_customer_features(customer_id)  # ← Riesgo de inconsistencia
# MAGIC     return model.predict(features)
# MAGIC ```
# MAGIC
# MAGIC **Problemas**:
# MAGIC - Features calculadas múltiples veces
# MAGIC - Riesgo de inconsistencia entre training y serving
# MAGIC - Difícil compartir entre equipos
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Workflow con Feature Store
# MAGIC
# MAGIC ```python
# MAGIC # 1. Computar features UNA VEZ
# MAGIC features = compute_customer_features(raw_data)
# MAGIC fe.create_table("main.features.customer_features", df=features)
# MAGIC
# MAGIC # 2. Modelo 1: Churn
# MAGIC training_set_churn = fe.create_training_set(
# MAGIC     labels_churn, 
# MAGIC     feature_lookups=[...]
# MAGIC )
# MAGIC train_model_churn(training_set_churn)
# MAGIC
# MAGIC # 3. Modelo 2: Lifetime Value (reutiliza mismas features)
# MAGIC training_set_ltv = fe.create_training_set(
# MAGIC     labels_ltv,
# MAGIC     feature_lookups=[...]  # ← Misma tabla, cero cálculo adicional
# MAGIC )
# MAGIC train_model_ltv(training_set_ltv)
# MAGIC
# MAGIC # 4. Producción: Features automáticas
# MAGIC predictions = fe.score_batch(model_uri, customer_ids)  # ← Lookup automático
# MAGIC ```
# MAGIC
# MAGIC **Ventajas**:
# MAGIC - ✅ Features calculadas una sola vez
# MAGIC - ✅ Consistencia garantizada
# MAGIC - ✅ Reutilización entre modelos
# MAGIC - ✅ Menos código, menos bugs
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Impacto en Métricas
# MAGIC
# MAGIC | Métrica | Sin Feature Store | Con Feature Store |
# MAGIC |---------|-------------------|-------------------|
# MAGIC | Tiempo desarrollo nuevo modelo | 2-4 semanas | 1-2 semanas |
# MAGIC | Training-serving skew | 10-30% de modelos | <5% |
# MAGIC | Features reutilizadas | <20% | >70% |
# MAGIC | Tiempo de deployment | Días | Horas |
# MAGIC | Esfuerzo de mantenimiento | Alto | Bajo |

# COMMAND ----------

# DBTITLE 1,Resumen y Conclusiones
# MAGIC %md
# MAGIC # Resumen y Conclusiones
# MAGIC
# MAGIC ## Conceptos Clave Aprendidos
# MAGIC
# MAGIC ### AutoML
# MAGIC - ✅ Automatiza tareas repetitivas del proceso ML
# MAGIC - ✅ Acelera desarrollo y proporciona baseline sólido
# MAGIC - ✅ Databricks AutoML genera notebooks reproducibles
# MAGIC - ⚠️ No reemplaza conocimiento del dominio
# MAGIC - ⚠️ Requiere datos bien preparados
# MAGIC
# MAGIC ### Feature Store
# MAGIC - ✅ Repositorio centralizado de features
# MAGIC - ✅ Garantiza consistencia training-serving
# MAGIC - ✅ Facilita reutilización y colaboración
# MAGIC - ✅ Point-in-time correctness para evitar leakage
# MAGIC - ✅ Integrado con Unity Catalog y MLflow
# MAGIC
# MAGIC ### Integración AutoML + Feature Store
# MAGIC - Workflow completo: features → training → serving
# MAGIC - Reproducibilidad y trazabilidad end-to-end
# MAGIC - Aceleración dramática del ciclo de desarrollo
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Flujo de Trabajo Recomendado
# MAGIC
# MAGIC ```
# MAGIC 1. Feature Engineering
# MAGIC    └→ Crear Feature Tables
# MAGIC
# MAGIC 2. Training
# MAGIC    ├→ Create Training Set (Feature Lookups)
# MAGIC    └→ AutoML para exploración rápida
# MAGIC
# MAGIC 3. Refinamiento
# MAGIC    ├→ Revisar notebooks de AutoML
# MAGIC    └→ Iterar manualmente si necesario
# MAGIC
# MAGIC 4. Deployment
# MAGIC    └→ Feature Store garantiza consistencia
# MAGIC
# MAGIC 5. Monitoring
# MAGIC    └→ MLflow tracking + Feature Store lineage
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Próximos Pasos
# MAGIC
# MAGIC En el **notebook de práctica** aplicarás:
# MAGIC 1. Crear Feature Tables para datasets reales
# MAGIC 2. Ejecutar AutoML con Feature Store
# MAGIC 3. Comparar resultados con/sin Feature Store
# MAGIC 4. Implementar feature serving para producción
# MAGIC 5. Analizar métricas y mejores prácticas
# MAGIC
# MAGIC **Continúa al notebook:** `Práctica - AutoML y Feature Store` 🚀

# COMMAND ----------

