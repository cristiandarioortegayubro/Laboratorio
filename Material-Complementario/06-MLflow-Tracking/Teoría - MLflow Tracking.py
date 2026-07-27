# Databricks notebook source
# DBTITLE 1,Módulo 06 - MLflow Tracking
# MAGIC %md
# MAGIC # Módulo 06: MLflow Tracking
# MAGIC
# MAGIC ## Contenido del Módulo
# MAGIC
# MAGIC 1. **Introducción a MLflow**
# MAGIC    - ¿Qué es MLflow?
# MAGIC    - Componentes de MLflow
# MAGIC    - MLflow en Databricks
# MAGIC
# MAGIC 2. **MLflow Tracking**
# MAGIC    - Experimentos y Runs
# MAGIC    - Logging: Parámetros, Métricas, Artifacts
# MAGIC    - Modelos y Tags
# MAGIC
# MAGIC 3. **MLflow UI**
# MAGIC    - Navegación y búsqueda
# MAGIC    - Comparación de runs
# MAGIC    - Visualizaciones
# MAGIC
# MAGIC 4. **Integración con Databricks**
# MAGIC    - AutoML + MLflow
# MAGIC    - Feature Store + MLflow
# MAGIC    - Notebooks + MLflow
# MAGIC
# MAGIC 5. **Mejores Prácticas**
# MAGIC    - Organización de experimentos
# MAGIC    - Logging efectivo
# MAGIC    - Reproducibilidad
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC **Objetivo**: Dominar el tracking de experimentos ML para garantizar reproducibilidad, comparabilidad y trazabilidad en el ciclo de vida de modelos.

# COMMAND ----------

# DBTITLE 1,1. Introducción a MLflow
# MAGIC %md
# MAGIC # 1. Introducción a MLflow
# MAGIC
# MAGIC ## ¿Qué es MLflow?
# MAGIC
# MAGIC **MLflow** es una plataforma open-source para gestionar el ciclo de vida completo de Machine Learning:
# MAGIC - Experimentación y desarrollo
# MAGIC - Reproducción de resultados
# MAGIC - Deployment y productización
# MAGIC - Monitoreo y gobernanza
# MAGIC
# MAGIC ## Componentes de MLflow
# MAGIC
# MAGIC MLflow consta de **4 componentes principales**:
# MAGIC
# MAGIC | Componente | Descripción | Uso |
# MAGIC |------------|-------------|-----|
# MAGIC | **MLflow Tracking** | Registro de experimentos, parámetros, métricas, artifacts | 🔵 **Foco de este módulo** |
# MAGIC | **MLflow Projects** | Formato para empaquetar código ML reproducible | Proyectos reutilizables |
# MAGIC | **MLflow Models** | Formato estándar para empaquetar modelos | Deployment multi-framework |
# MAGIC | **MLflow Registry** | Repositorio centralizado de modelos | Versionado y lifecycle |
# MAGIC
# MAGIC ## MLflow Tracking: El Corazón de MLflow
# MAGIC
# MAGIC **MLflow Tracking** es el componente más usado. Permite:
# MAGIC
# MAGIC ```
# MAGIC ┌─────────────────────────────────────────┐
# MAGIC │          MLFLOW TRACKING                 │
# MAGIC ├─────────────────────────────────────────┤
# MAGIC │  📋 Parámetros (learning_rate, epochs)  │
# MAGIC │  📊 Métricas (accuracy, loss, f1)      │
# MAGIC │  💾 Artifacts (modelos, plots, data)    │
# MAGIC │  🏷️ Tags (metadata, descripción)          │
# MAGIC │  📄 Código fuente (automático)           │
# MAGIC │  🕰️ Timestamps (inicio, fin, duración)   │
# MAGIC └─────────────────────────────────────────┘
# MAGIC ```
# MAGIC
# MAGIC ## MLflow en Databricks
# MAGIC
# MAGIC En Databricks, MLflow está:
# MAGIC - **✅ Preinstalado** y preconfigurado
# MAGIC - **✅ Integrado** con notebooks, AutoML, Feature Store
# MAGIC - **✅ Gestionado** (no requiere setup de infraestructura)
# MAGIC - **✅ Conectado** con Unity Catalog para gobernanza
# MAGIC - **✅ Accesible** vía UI web interactiva
# MAGIC
# MAGIC ```python
# MAGIC # MLflow en Databricks está listo para usar
# MAGIC import mlflow
# MAGIC
# MAGIC # Automáticamente conectado al workspace
# MAGIC print(f"MLflow version: {mlflow.__version__}")
# MAGIC print(f"Tracking URI: {mlflow.get_tracking_uri()}")
# MAGIC ```
# MAGIC
# MAGIC ## ¿Por Qué Necesitamos MLflow Tracking?
# MAGIC
# MAGIC ### Sin MLflow
# MAGIC
# MAGIC ```python
# MAGIC # Experimento 1
# MAGIC model = train_model(lr=0.01, epochs=10)
# MAGIC accuracy = 0.85  # ¿Cómo obtuvo este resultado?
# MAGIC
# MAGIC # Experimento 2 (al día siguiente)
# MAGIC model = train_model(lr=0.001, epochs=20)  
# MAGIC accuracy = 0.87  # ¿Qué parámetros usé?
# MAGIC
# MAGIC # ❌ Problemas:
# MAGIC # - No puedo reproducir resultados
# MAGIC # - No sé qué parámetros dieron mejor accuracy
# MAGIC # - No tengo el modelo guardado
# MAGIC # - No puedo comparar experimentos
# MAGIC ```
# MAGIC
# MAGIC ### Con MLflow
# MAGIC
# MAGIC ```python
# MAGIC import mlflow
# MAGIC
# MAGIC with mlflow.start_run():
# MAGIC     mlflow.log_param("learning_rate", 0.01)
# MAGIC     mlflow.log_param("epochs", 10)
# MAGIC     
# MAGIC     model = train_model(lr=0.01, epochs=10)
# MAGIC     accuracy = evaluate(model)
# MAGIC     
# MAGIC     mlflow.log_metric("accuracy", accuracy)
# MAGIC     mlflow.sklearn.log_model(model, "model")
# MAGIC
# MAGIC # ✅ Ventajas:
# MAGIC # - Todos los parámetros registrados
# MAGIC # - Métricas historizadas
# MAGIC # - Modelo versionado
# MAGIC # - Reproducible
# MAGIC # - Comparable en UI
# MAGIC ```

# COMMAND ----------



# COMMAND ----------

# DBTITLE 1,2. Conceptos Clave de MLflow Tracking
# MAGIC %md
# MAGIC # 2. Conceptos Clave de MLflow Tracking
# MAGIC
# MAGIC ## Jerarquía de MLflow
# MAGIC
# MAGIC ```
# MAGIC Workspace
# MAGIC   │
# MAGIC   ├── Experiment 1: "Customer Churn"
# MAGIC   │     ├── Run 1 (Random Forest, lr=0.01)
# MAGIC   │     ├── Run 2 (XGBoost, lr=0.001)
# MAGIC   │     └── Run 3 (Logistic Regression)
# MAGIC   │
# MAGIC   ├── Experiment 2: "Sales Forecasting"
# MAGIC   │     ├── Run 1 (ARIMA)
# MAGIC   │     └── Run 2 (Prophet)
# MAGIC   │
# MAGIC   └── Experiment 3: "Image Classification"
# MAGIC         ├── Run 1 (CNN)
# MAGIC         └── Run 2 (Transfer Learning)
# MAGIC ```
# MAGIC
# MAGIC ## 1. Experiment (Experimento)
# MAGIC
# MAGIC Un **Experiment** agrupa múltiples runs relacionados:
# MAGIC - Un proyecto o problema de negocio
# MAGIC - Diferentes enfoques para resolver el mismo problema
# MAGIC - Iteraciones sobre el mismo modelo
# MAGIC
# MAGIC **Ejemplo**: "Predicción de Churn Bancario"
# MAGIC
# MAGIC ## 2. Run (Ejecución)
# MAGIC
# MAGIC Un **Run** es una ejecución individual de código ML:
# MAGIC - Un entrenamiento de modelo
# MAGIC - Una configuración de hiperparámetros
# MAGIC - Un experimento específico
# MAGIC
# MAGIC **Cada run registra**:
# MAGIC
# MAGIC | Elemento | Descripción | Ejemplo |
# MAGIC |----------|-------------|--------|
# MAGIC | **Parámetros** | Inputs configurables (inmutables) | `learning_rate=0.01` |
# MAGIC | **Métricas** | Outputs medibles (mutables, históricos) | `accuracy=0.85` |
# MAGIC | **Artifacts** | Archivos de salida | Modelo, plots, CSV |
# MAGIC | **Tags** | Metadata clave-valor | `environment=production` |
# MAGIC | **Código** | Versión del código (Git commit) | `commit_hash=abc123` |
# MAGIC | **Timestamps** | Inicio, fin, duración | `start_time`, `end_time` |
# MAGIC
# MAGIC ## 3. Parámetros vs. Métricas
# MAGIC
# MAGIC ### Parámetros (Immutable Inputs)
# MAGIC
# MAGIC - **Qué son**: Configuraciones del modelo/entrenamiento
# MAGIC - **Características**:
# MAGIC   - Definidos **antes** del entrenamiento
# MAGIC   - **Inmutables** durante el run
# MAGIC   - Solo **valores escalares** (int, float, string)
# MAGIC
# MAGIC ```python
# MAGIC mlflow.log_param("learning_rate", 0.01)
# MAGIC mlflow.log_param("max_depth", 10)
# MAGIC mlflow.log_param("algorithm", "random_forest")
# MAGIC ```
# MAGIC
# MAGIC ### Métricas (Mutable Outputs)
# MAGIC
# MAGIC - **Qué son**: Mediciones del rendimiento
# MAGIC - **Características**:
# MAGIC   - Calculadas **durante/después** del entrenamiento
# MAGIC   - **Mutables** (pueden actualizarse en cada epoch)
# MAGIC   - Solo **valores numéricos** (int, float)
# MAGIC   - Soportan **historial** (para gráficas de entrenamiento)
# MAGIC
# MAGIC ```python
# MAGIC mlflow.log_metric("accuracy", 0.85)
# MAGIC mlflow.log_metric("f1_score", 0.82)
# MAGIC
# MAGIC # Logging iterativo (por epoch)
# MAGIC for epoch in range(10):
# MAGIC     loss = train_epoch()
# MAGIC     mlflow.log_metric("loss", loss, step=epoch)
# MAGIC ```
# MAGIC
# MAGIC ## 4. Artifacts
# MAGIC
# MAGIC **Artifacts** son archivos generados durante el run:
# MAGIC
# MAGIC | Tipo | Ejemplos |
# MAGIC |------|----------|
# MAGIC | **Modelos** | `model.pkl`, `model.h5` |
# MAGIC | **Plots** | `confusion_matrix.png`, `roc_curve.png` |
# MAGIC | **Data** | `predictions.csv`, `feature_importance.json` |
# MAGIC | **Configs** | `config.yaml`, `requirements.txt` |
# MAGIC | **Notebooks** | `training_notebook.html` |
# MAGIC
# MAGIC ```python
# MAGIC # Log modelo
# MAGIC mlflow.sklearn.log_model(model, "model")
# MAGIC
# MAGIC # Log archivo
# MAGIC mlflow.log_artifact("confusion_matrix.png")
# MAGIC
# MAGIC # Log directorio completo
# MAGIC mlflow.log_artifacts("output_folder/")
# MAGIC ```
# MAGIC
# MAGIC ## 5. Tags
# MAGIC
# MAGIC **Tags** son metadata adicional en formato clave-valor:
# MAGIC
# MAGIC ```python
# MAGIC mlflow.set_tag("environment", "production")
# MAGIC mlflow.set_tag("team", "data-science")
# MAGIC mlflow.set_tag("model_type", "ensemble")
# MAGIC mlflow.set_tag("notes", "Mejor modelo hasta ahora")
# MAGIC ```
# MAGIC
# MAGIC **Uso común**:
# MAGIC - Filtrado y búsqueda en UI
# MAGIC - Organización de experimentos
# MAGIC - Anotaciones y documentación

# COMMAND ----------

# DBTITLE 1,3. Ejemplo Completo de Run
# 3. Ejemplo Completo de Run

import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# Crear datos de ejemplo
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Iniciar un run de MLflow
with mlflow.start_run(run_name="ejemplo_completo_rf") as run:
    
    # 1. Log parámetros
    params = {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 5,
        "random_state": 42
    }
    mlflow.log_params(params)
    
    # 2. Entrenar modelo
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    
    # 3. Evaluar y log métricas
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba)
    }
    mlflow.log_metrics(metrics)
    
    # 4. Log modelo
    mlflow.sklearn.log_model(model, "model")
    
    # 5. Log tags
    mlflow.set_tag("model_type", "random_forest")
    mlflow.set_tag("dataset", "synthetic_classification")
    mlflow.set_tag("framework", "sklearn")
    
    # Imprimir información del run
    print(f"✅ Run completado")
    print(f"\nRun ID: {run.info.run_id}")
    print(f"Experiment ID: {run.info.experiment_id}")
    print(f"\nMétricas:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")

# COMMAND ----------

# DBTITLE 1,4. MLflow UI - Navegación
# MAGIC %md
# MAGIC # 4. MLflow UI - Navegación y Visualización
# MAGIC
# MAGIC ## Acceder a la UI de MLflow
# MAGIC
# MAGIC En Databricks, hay **3 formas** de acceder a MLflow UI:
# MAGIC
# MAGIC ### 1. Desde el Notebook
# MAGIC
# MAGIC Después de ejecutar un run:
# MAGIC ```python
# MAGIC with mlflow.start_run() as run:
# MAGIC     # ... tu código ...
# MAGIC     print(f"Ver run: {run.info.artifact_uri}")
# MAGIC ```
# MAGIC
# MAGIC Haz clic en el icono de MLflow 📋 en la barra lateral del notebook.
# MAGIC
# MAGIC ### 2. Desde el Menú Principal
# MAGIC
# MAGIC * **Machine Learning** → **Experiments**
# MAGIC * Ver todos los experimentos del workspace
# MAGIC
# MAGIC ### 3. URL Directa
# MAGIC
# MAGIC ```
# MAGIC https://<workspace-url>/#mlflow/experiments/<experiment_id>
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Vistas Principales de la UI
# MAGIC
# MAGIC ### Vista de Experimentos
# MAGIC
# MAGIC Muestra todos los runs de un experimento:
# MAGIC
# MAGIC | Columna | Descripción |
# MAGIC |---------|-------------|
# MAGIC | **Run Name** | Nombre del run |
# MAGIC | **Created** | Timestamp de creación |
# MAGIC | **Duration** | Tiempo de ejecución |
# MAGIC | **Source** | Notebook/script origen |
# MAGIC | **User** | Usuario que ejecutó el run |
# MAGIC | **Metrics** | Métricas clave (configurables) |
# MAGIC | **Params** | Parámetros (configurables) |
# MAGIC
# MAGIC ### Vista de Run Individual
# MAGIC
# MAGIC Detalles de un run específico:
# MAGIC
# MAGIC ```
# MAGIC ┌──────────────────────────────────┐
# MAGIC │        RUN DETAILS              │
# MAGIC ├──────────────────────────────────┤
# MAGIC │ 📋 Parameters              │
# MAGIC │   - learning_rate: 0.01      │
# MAGIC │   - max_depth: 10            │
# MAGIC ├──────────────────────────────────┤
# MAGIC │ 📊 Metrics                  │
# MAGIC │   - accuracy: 0.85           │
# MAGIC │   - f1_score: 0.82           │
# MAGIC ├──────────────────────────────────┤
# MAGIC │ 💾 Artifacts                │
# MAGIC │   - model/                   │
# MAGIC │   - confusion_matrix.png     │
# MAGIC ├──────────────────────────────────┤
# MAGIC │ 🏷️ Tags                      │
# MAGIC │   - model_type: rf           │
# MAGIC │   - environment: dev         │
# MAGIC └──────────────────────────────────┘
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Funcionalidades Clave
# MAGIC
# MAGIC ### 1. Comparación de Runs
# MAGIC
# MAGIC * Selecciona múltiples runs (checkbox)
# MAGIC * Click en **"Compare"**
# MAGIC * Ve diferencias en:
# MAGIC   - Parámetros
# MAGIC   - Métricas
# MAGIC   - Visualizaciones paralelas
# MAGIC
# MAGIC ### 2. Búsqueda y Filtrado
# MAGIC
# MAGIC ```sql
# MAGIC -- Buscar por métrica
# MAGIC metrics.accuracy > 0.8
# MAGIC
# MAGIC -- Buscar por parámetro
# MAGIC params.learning_rate = "0.01"
# MAGIC
# MAGIC -- Buscar por tag
# MAGIC tags.model_type = "random_forest"
# MAGIC
# MAGIC -- Combinaciones
# MAGIC metrics.accuracy > 0.8 AND params.max_depth < 15
# MAGIC ```
# MAGIC
# MAGIC ### 3. Gráficas de Métricas
# MAGIC
# MAGIC Para métricas con historial (`step`):
# MAGIC * Gráfica de loss vs. epoch
# MAGIC * Comparación de curvas de aprendizaje
# MAGIC * Identificación de overfitting
# MAGIC
# MAGIC ### 4. Descarga de Artifacts
# MAGIC
# MAGIC * Click en cualquier artifact
# MAGIC * Visualización inline (imágenes, JSON, CSV)
# MAGIC * Descarga local
# MAGIC
# MAGIC ### 5. Registro de Modelos
# MAGIC
# MAGIC Desde un run:
# MAGIC * **"Register Model"**
# MAGIC * Crear modelo nuevo o agregar versión
# MAGIC * Integración con Model Registry

# COMMAND ----------

# DBTITLE 1,5. Comparación de Múltiples Runs
# 5. Comparación de Múltiples Runs

import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Datos
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Experimento para comparar diferentes configuraciones
mlflow.set_experiment("/Users/cortega@uda.edu.ar/comparacion_hiperparametros")

print("Ejecutando múltiples runs con diferentes hiperparámetros...\n")

# Configuraciones a probar
configs = [
    {"n_estimators": 50, "max_depth": 5},
    {"n_estimators": 100, "max_depth": 10},
    {"n_estimators": 200, "max_depth": 15},
    {"n_estimators": 100, "max_depth": 20},
]

for i, config in enumerate(configs, 1):
    with mlflow.start_run(run_name=f"config_{i}"):
        # Log parámetros
        mlflow.log_params(config)
        
        # Entrenar
        model = RandomForestClassifier(**config, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluar
        accuracy = accuracy_score(y_test, model.predict(X_test))
        mlflow.log_metric("accuracy", accuracy)
        
        # Tag
        mlflow.set_tag("config_id", f"config_{i}")
        
        print(f"Config {i}: n_estimators={config['n_estimators']}, "
              f"max_depth={config['max_depth']} → Accuracy={accuracy:.4f}")

print("\n✅ Runs completados")
print("\n👉 Ve a MLflow UI para comparar:")
print("   1. Selecciona los 4 runs")
print("   2. Click en 'Compare'")
print("   3. Analiza Parallel Coordinates Plot")

# COMMAND ----------

# DBTITLE 1,6. Mejores Prácticas
# MAGIC %md
# MAGIC # 6. Mejores Prácticas de MLflow Tracking
# MAGIC
# MAGIC ## Organización de Experimentos
# MAGIC
# MAGIC ### ✅ Buenas Prácticas
# MAGIC
# MAGIC | Práctica | Descripción |
# MAGIC |----------|-------------|
# MAGIC | **Nombres descriptivos** | `customer_churn_v2` en vez de `experiment_1` |
# MAGIC | **Jerarquía por proyecto** | `/Shared/banking/credit_risk/model_v2` |
# MAGIC | **Un experimento por modelo** | No mezclar clasificación con regresión |
# MAGIC | **Documentar con tags** | Añadir contexto en cada run |
# MAGIC | **Versionado semántico** | `v1.0`, `v1.1`, `v2.0` en tags |
# MAGIC
# MAGIC ### ❌ Anti-patrones
# MAGIC
# MAGIC * Experimentos con nombres genéricos (`test`, `experiment1`)
# MAGIC * Mixing unrelated runs en un solo experimento
# MAGIC * No documentar parámetros críticos
# MAGIC * No usar tags para filtrado
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Logging Efectivo
# MAGIC
# MAGIC ### Qué Loggear
# MAGIC
# MAGIC | Tipo | ¿Cuándo loggear? | Ejemplos |
# MAGIC |------|----------------|----------|
# MAGIC | **Parámetros** | Siempre | Hiperparámetros, configuraciones |
# MAGIC | **Métricas** | Siempre | Accuracy, loss, F1, AUC |
# MAGIC | **Modelo** | Modelos finales | Mejores modelos, checkpoints |
# MAGIC | **Artifacts** | Visualizaciones importantes | Confusion matrix, ROC curves |
# MAGIC | **Tags** | Para organización | Environment, version, status |
# MAGIC
# MAGIC ### Parámetros Esenciales a Loggear
# MAGIC
# MAGIC ```python
# MAGIC # Hiperparámetros del modelo
# MAGIC mlflow.log_param("learning_rate", lr)
# MAGIC mlflow.log_param("batch_size", batch_size)
# MAGIC mlflow.log_param("epochs", epochs)
# MAGIC
# MAGIC # Parámetros de datos
# MAGIC mlflow.log_param("train_size", len(X_train))
# MAGIC mlflow.log_param("test_size", len(X_test))
# MAGIC mlflow.log_param("n_features", X_train.shape[1])
# MAGIC
# MAGIC # Parámetros de preprocessing
# MAGIC mlflow.log_param("scaler", "StandardScaler")
# MAGIC mlflow.log_param("handle_missing", "mean_imputation")
# MAGIC
# MAGIC # Configuración del entorno
# MAGIC mlflow.log_param("framework", "sklearn")
# MAGIC mlflow.log_param("python_version", sys.version)
# MAGIC ```
# MAGIC
# MAGIC ### Métricas Esenciales
# MAGIC
# MAGIC ```python
# MAGIC # Métricas de rendimiento
# MAGIC mlflow.log_metric("train_accuracy", train_acc)
# MAGIC mlflow.log_metric("val_accuracy", val_acc)
# MAGIC mlflow.log_metric("test_accuracy", test_acc)
# MAGIC
# MAGIC # Métricas de negocio (si aplica)
# MAGIC mlflow.log_metric("expected_revenue_increase", revenue_gain)
# MAGIC mlflow.log_metric("false_positive_cost", fp_cost)
# MAGIC
# MAGIC # Metadata de entrenamiento
# MAGIC mlflow.log_metric("training_time_seconds", duration)
# MAGIC mlflow.log_metric("model_size_mb", model_size)
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Reproducibilidad
# MAGIC
# MAGIC ### Elementos Clave
# MAGIC
# MAGIC 1. **Parámetros completos**: Todos los hiperparámetros relevantes
# MAGIC 2. **Seeds**: Random seeds para reproducibilidad
# MAGIC 3. **Environment**: Versiones de librerías
# MAGIC 4. **Data**: Versión/snapshot de datos usados
# MAGIC 5. **Código**: Git commit hash
# MAGIC
# MAGIC ### Ejemplo Completo Reproducible
# MAGIC
# MAGIC ```python
# MAGIC import mlflow
# MAGIC import sys
# MAGIC import sklearn
# MAGIC import numpy as np
# MAGIC
# MAGIC with mlflow.start_run():
# MAGIC     # 1. Log seeds
# MAGIC     seed = 42
# MAGIC     mlflow.log_param("random_seed", seed)
# MAGIC     np.random.seed(seed)
# MAGIC     
# MAGIC     # 2. Log versiones
# MAGIC     mlflow.log_param("sklearn_version", sklearn.__version__)
# MAGIC     mlflow.log_param("python_version", sys.version)
# MAGIC     
# MAGIC     # 3. Log data version
# MAGIC     mlflow.log_param("data_version", "2024-01-15")
# MAGIC     mlflow.log_param("data_source", "main.features.customer_features")
# MAGIC     
# MAGIC     # 4. Log git commit (si está disponible)
# MAGIC     try:
# MAGIC         import git
# MAGIC         repo = git.Repo(search_parent_directories=True)
# MAGIC         mlflow.set_tag("git_commit", repo.head.object.hexsha)
# MAGIC     except:
# MAGIC         pass
# MAGIC     
# MAGIC     # 5. Entrenar modelo
# MAGIC     model = train_model(seed=seed)
# MAGIC     
# MAGIC     # 6. Log modelo con signature
# MAGIC     mlflow.sklearn.log_model(
# MAGIC         model, 
# MAGIC         "model",
# MAGIC         signature=mlflow.models.infer_signature(X_train, y_train)
# MAGIC     )
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Integración con Otros Componentes
# MAGIC
# MAGIC ### AutoML + MLflow
# MAGIC
# MAGIC Databricks AutoML automáticamente:
# MAGIC * Crea experimento MLflow
# MAGIC * Registra todos los runs
# MAGIC * Loggea parámetros, métricas, modelos
# MAGIC * Genera notebooks con código MLflow
# MAGIC
# MAGIC ### Feature Store + MLflow
# MAGIC
# MAGIC Modelos entrenados con Feature Store:
# MAGIC * Metadata de features en MLflow
# MAGIC * Linaje de features
# MAGIC * Serving automático con feature lookup
# MAGIC
# MAGIC ### Notebooks + MLflow
# MAGIC
# MAGIC En Databricks notebooks:
# MAGIC * MLflow tracking automático
# MAGIC * Experimento por notebook (por defecto)
# MAGIC * UI integrada en sidebar
# MAGIC * Link directo a runs en output cells

# COMMAND ----------

# DBTITLE 1,7. Búsqueda y Consulta de Runs
# 7. Búsqueda y Consulta de Runs Programaticamente

import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient()

# 1. Buscar runs por métrica
print("±±=" * 30)
print("BUSCAR RUNS CON ACCURACY > 0.80")
print("=" * 60)

runs = mlflow.search_runs(
    filter_string="metrics.accuracy > 0.80",
    order_by=["metrics.accuracy DESC"],
    max_results=5
)

if not runs.empty:
    print(runs[['run_id', 'params.n_estimators', 'metrics.accuracy']].head())
else:
    print("No se encontraron runs con accuracy > 0.80")

# 2. Buscar por parámetro
print("\n" + "=" * 60)
print("BUSCAR RUNS CON max_depth = 10")
print("=" * 60)

runs_by_param = mlflow.search_runs(
    filter_string="params.max_depth = '10'"
)

if not runs_by_param.empty:
    print(f"Encontrados: {len(runs_by_param)} runs")
else:
    print("No se encontraron runs con max_depth=10")

# 3. Buscar best run
print("\n" + "=" * 60)
print("MEJOR RUN POR ACCURACY")
print("=" * 60)

best_runs = mlflow.search_runs(
    order_by=["metrics.accuracy DESC"],
    max_results=1
)

if not best_runs.empty:
    best_run = best_runs.iloc[0]
    print(f"Run ID: {best_run['run_id']}")
    print(f"Accuracy: {best_run['metrics.accuracy']:.4f}")
    if 'params.n_estimators' in best_run:
        print(f"N Estimators: {best_run['params.n_estimators']}")
    if 'params.max_depth' in best_run:
        print(f"Max Depth: {best_run['params.max_depth']}")

print("\n✅ Búsquedas completadas")

# COMMAND ----------

# DBTITLE 1,Resumen y Conclusiones
# MAGIC %md
# MAGIC # Resumen y Conclusiones
# MAGIC
# MAGIC ## Conceptos Clave Aprendidos
# MAGIC
# MAGIC ### MLflow Tracking
# MAGIC
# MAGIC ✅ **Experiment**: Agrupa runs relacionados  
# MAGIC ✅ **Run**: Ejecución individual con parámetros, métricas, artifacts  
# MAGIC ✅ **Parámetros**: Configuraciones inmutables (inputs)  
# MAGIC ✅ **Métricas**: Mediciones mutables (outputs)  
# MAGIC ✅ **Artifacts**: Archivos generados (modelos, plots, data)  
# MAGIC ✅ **Tags**: Metadata para organización y filtrado  
# MAGIC
# MAGIC ### MLflow UI
# MAGIC
# MAGIC ✅ Navegación intuitiva de experimentos y runs  
# MAGIC ✅ Comparación visual de múltiples runs  
# MAGIC ✅ Búsqueda y filtrado avanzado  
# MAGIC ✅ Visualización de métricas históricas  
# MAGIC ✅ Descarga y visualización de artifacts  
# MAGIC
# MAGIC ### Mejores Prácticas
# MAGIC
# MAGIC ✅ Nombres descriptivos para experimentos  
# MAGIC ✅ Loggear parámetros completos  
# MAGIC ✅ Métricas de negocio además de técnicas  
# MAGIC ✅ Tags para organización  
# MAGIC ✅ Reproducibilidad (seeds, versions, data)  
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Workflow Recomendado
# MAGIC
# MAGIC ```
# MAGIC 1. Crear Experimento
# MAGIC    mlflow.set_experiment("/path/to/experiment")
# MAGIC
# MAGIC 2. Iniciar Run
# MAGIC    with mlflow.start_run(run_name="descriptive_name"):
# MAGIC
# MAGIC 3. Log Parámetros
# MAGIC    mlflow.log_params({...})
# MAGIC
# MAGIC 4. Entrenar Modelo
# MAGIC    model = train(...)
# MAGIC
# MAGIC 5. Log Métricas
# MAGIC    mlflow.log_metrics({...})
# MAGIC
# MAGIC 6. Log Modelo
# MAGIC    mlflow.sklearn.log_model(model, "model")
# MAGIC
# MAGIC 7. Log Artifacts
# MAGIC    mlflow.log_artifact("plot.png")
# MAGIC
# MAGIC 8. Tags
# MAGIC    mlflow.set_tag("status", "production")
# MAGIC
# MAGIC 9. Comparar en UI
# MAGIC    - Seleccionar runs
# MAGIC    - Comparar métricas
# MAGIC    - Identificar mejor modelo
# MAGIC
# MAGIC 10. Registrar Mejor Modelo
# MAGIC     - Model Registry
# MAGIC     - Versionado
# MAGIC     - Lifecycle management
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC ## Beneficios de MLflow Tracking
# MAGIC
# MAGIC | Beneficio | Impacto |
# MAGIC |-----------|--------|
# MAGIC | **Reproducibilidad** | Cualquier experimento puede repetirse exactamente |
# MAGIC | **Comparabilidad** | Fácil comparar múltiples enfoques |
# MAGIC | **Trazabilidad** | Historial completo de experimentación |
# MAGIC | **Colaboración** | Compartir resultados entre equipos |
# MAGIC | **Documentación** | Registro automático de decisiones |
# MAGIC | **Gobernanza** | Auditoría y compliance |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Próximos Pasos
# MAGIC
# MAGIC En el **notebook de práctica** aplicarás:
# MAGIC
# MAGIC 1. Crear experimentos organizados
# MAGIC 2. Loggear parámetros, métricas, artifacts
# MAGIC 3. Comparar múltiples runs
# MAGIC 4. Buscar y filtrar runs programaticamente
# MAGIC 5. Implementar workflow completo reproducible
# MAGIC 6. Integrar con Feature Store y AutoML
# MAGIC
# MAGIC **Continúa al notebook:** `Práctica - MLflow Tracking` 🚀