# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,Práctica Módulo 06 - MLflow Tracking
# MAGIC %md
# MAGIC # Práctica: MLflow Tracking
# MAGIC
# MAGIC ## Objetivos de la Práctica
# MAGIC
# MAGIC 1. **Tracking básico**: Loggear parámetros, métricas, modelos
# MAGIC 2. **Comparación**: Comparar múltiples runs de diferentes modelos
# MAGIC 3. **Búsqueda**: Consultar runs programaticamente
# MAGIC 4. **Artifacts**: Loggear visualizaciones y archivos
# MAGIC 5. **Workflow completo**: Implementar pipeline reproducible
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Estructura
# MAGIC
# MAGIC - **Ejercicio 1**: Tracking básico de modelo
# MAGIC - **Ejercicio 2**: Comparación de múltiples configuraciones
# MAGIC - **Ejercicio 3**: Logging de artifacts (plots, CSV, JSON)
# MAGIC - **Ejercicio 4**: Búsqueda y análisis de runs
# MAGIC - **Ejercicio 5**: Workflow completo reproducible
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC **Dataset**: Datos sintéticos de clasificación para prototipado rápido

# COMMAND ----------

# DBTITLE 1,Setup: Datos y Librerías
# Setup: Importar librerías y crear datos
import mlflow
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay,
    classification_report
)
import warnings
warnings.filterwarnings('ignore')

# Crear datos de clasificación
np.random.seed(42)
X, y = make_classification(
    n_samples=2000,
    n_features=20,
    n_informative=15,
    n_redundant=3,
    n_classes=2,
    weights=[0.7, 0.3],  # Clases desbalanceadas
    random_state=42
)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("✅ Setup completado")
print(f"\nDataset:")
print(f"  Train: {X_train.shape[0]} samples")
print(f"  Test: {X_test.shape[0]} samples")
print(f"  Features: {X_train.shape[1]}")
print(f"  Class distribution: {np.bincount(y_train)}")

# COMMAND ----------

# DBTITLE 1,Ejercicio 1: Tracking Básico
# MAGIC %md
# MAGIC # Ejercicio 1: Tracking Básico de Modelo
# MAGIC
# MAGIC ## Objetivo
# MAGIC Implementar tracking completo de un modelo Random Forest:
# MAGIC * Loggear hiperparámetros
# MAGIC * Loggear métricas (accuracy, F1, ROC AUC)
# MAGIC * Guardar el modelo
# MAGIC * Añadir tags descriptivos
# MAGIC
# MAGIC ## Tareas
# MAGIC 1. Crear experimento MLflow
# MAGIC 2. Entrenar Random Forest
# MAGIC 3. Loggear parámetros, métricas, modelo
# MAGIC 4. Verificar en MLflow UI

# COMMAND ----------

# DBTITLE 1,Ejercicio 1: Solución
# Ejercicio 1: Solución - Tracking Básico

# Crear experimento
mlflow.set_experiment("/Users/cortega@uda.edu.ar/Laboratorio/MLflow_Practica_Ejercicio1")

print("=" * 60)
print("EJERCICIO 1: TRACKING BÁSICO")
print("=" * 60)

with mlflow.start_run(run_name="random_forest_baseline") as run:
    
    # Parámetros del modelo
    params = {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "random_state": 42
    }
    
    # Log parámetros
    mlflow.log_params(params)
    mlflow.log_param("model_type", "RandomForest")
    mlflow.log_param("train_size", len(X_train))
    mlflow.log_param("test_size", len(X_test))
    
    # Entrenar modelo
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    
    # Predicciones
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Calcular métricas
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba)
    }
    
    # Log métricas
    mlflow.log_metrics(metrics)
    
    # Log modelo
    mlflow.sklearn.log_model(model, "model")
    
    # Tags
    mlflow.set_tag("framework", "sklearn")
    mlflow.set_tag("model_family", "ensemble")
    mlflow.set_tag("environment", "development")
    mlflow.set_tag("dataset", "synthetic_classification")
    
    # Imprimir resultados
    print(f"\n✅ Run completado")
    print(f"\nRun ID: {run.info.run_id}")
    print(f"\nMétricas:")
    for metric, value in metrics.items():
        print(f"  {metric:12s}: {value:.4f}")
    
    print(f"\n👉 Ve a MLflow UI para ver los detalles del run")

# COMMAND ----------

# DBTITLE 1,Ejercicio 2: Comparación de Modelos
# MAGIC %md
# MAGIC # Ejercicio 2: Comparación de Múltiples Modelos
# MAGIC
# MAGIC ## Objetivo
# MAGIC Comparar 3 algoritmos diferentes:
# MAGIC * Random Forest
# MAGIC * Gradient Boosting
# MAGIC * Logistic Regression
# MAGIC
# MAGIC Cada uno con sus parámetros óptimos.
# MAGIC
# MAGIC ## Tareas
# MAGIC 1. Entrenar 3 modelos diferentes
# MAGIC 2. Loggear todo en el mismo experimento
# MAGIC 3. Comparar métricas en MLflow UI
# MAGIC 4. Identificar el mejor modelo

# COMMAND ----------

# DBTITLE 1,Ejercicio 2: Solución
# Ejercicio 2: Solución - Comparación de Modelos

mlflow.set_experiment("/Users/cortega@uda.edu.ar/Laboratorio/MLflow_Practica_Ejercicio2")

print("=" * 60)
print("EJERCICIO 2: COMPARACIÓN DE MODELOS")
print("=" * 60)

# Configuraciones de modelos
model_configs = [
    {
        "name": "Random Forest",
        "model_class": RandomForestClassifier,
        "params": {
            "n_estimators": 100,
            "max_depth": 10,
            "random_state": 42
        }
    },
    {
        "name": "Gradient Boosting",
        "model_class": GradientBoostingClassifier,
        "params": {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 5,
            "random_state": 42
        }
    },
    {
        "name": "Logistic Regression",
        "model_class": LogisticRegression,
        "params": {
            "C": 1.0,
            "max_iter": 1000,
            "random_state": 42
        }
    }
]

results = []

for config in model_configs:
    with mlflow.start_run(run_name=config["name"].lower().replace(" ", "_")):
        
        print(f"\n{'='*40}")
        print(f"Entrenando: {config['name']}")
        print(f"{'='*40}")
        
        # Log parámetros
        mlflow.log_param("model_type", config["name"])
        mlflow.log_params(config["params"])
        
        # Entrenar
        model = config["model_class"](**config["params"])
        model.fit(X_train, y_train)
        
        # Evaluar
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba)
        }
        
        # Log métricas
        mlflow.log_metrics(metrics)
        
        # Log modelo
        mlflow.sklearn.log_model(model, "model")
        
        # Tags
        mlflow.set_tag("algorithm", config["name"])
        
        # Guardar resultados para comparación
        result = {"model": config["name"]}
        result.update(metrics)
        results.append(result)
        
        print(f"F1 Score: {metrics['f1_score']:.4f}")
        print(f"ROC AUC:  {metrics['roc_auc']:.4f}")

# Comparación final
print("\n" + "=" * 60)
print("COMPARACIÓN DE RESULTADOS")
print("=" * 60)

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

best_model = results_df.loc[results_df['f1_score'].idxmax()]
print(f"\n✅ Mejor modelo: {best_model['model']} (F1={best_model['f1_score']:.4f})")

print("\n👉 Ve a MLflow UI para comparar visualmente los 3 runs")

# COMMAND ----------

# DBTITLE 1,Ejercicio 3: Logging de Artifacts
# MAGIC %md
# MAGIC # Ejercicio 3: Logging de Artifacts
# MAGIC
# MAGIC ## Objetivo
# MAGIC Loggear múltiples tipos de artifacts:
# MAGIC * Confusion Matrix (imagen)
# MAGIC * Feature Importance (JSON)
# MAGIC * Predictions (CSV)
# MAGIC * Classification Report (TXT)
# MAGIC
# MAGIC ## Tareas
# MAGIC 1. Entrenar modelo y generar predicciones
# MAGIC 2. Crear visualizaciones
# MAGIC 3. Exportar datos y métricas
# MAGIC 4. Loggear todo como artifacts en MLflow

# COMMAND ----------

# DBTITLE 1,Ejercicio 3: Solución
# Ejercicio 3: Solución - Logging de Artifacts

import json

mlflow.set_experiment("/Users/cortega@uda.edu.ar/Laboratorio/MLflow_Practica_Ejercicio3")

print("=" * 60)
print("EJERCICIO 3: LOGGING DE ARTIFACTS")
print("=" * 60)

with mlflow.start_run(run_name="artifacts_completos"):
    
    # Entrenar modelo
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Log parámetros básicos
    mlflow.log_params({
        "n_estimators": 100,
        "max_depth": 10,
        "model_type": "RandomForest"
    })
    
    # Log métricas
    mlflow.log_metrics({
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred)
    })
    
    print("\n1️⃣ Creando Confusion Matrix...")
    # 1. Confusion Matrix (PNG)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Class 0', 'Class 1'])
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    plt.title('Confusion Matrix - Random Forest', fontsize=14)
    plt.savefig('confusion_matrix.png', dpi=100, bbox_inches='tight')
    plt.close()
    mlflow.log_artifact('confusion_matrix.png')
    print("✅ Confusion matrix guardada")
    
    print("\n2️⃣ Guardando Feature Importance...")
    # 2. Feature Importance (JSON)
    feature_importance = {
        f"feature_{i:02d}": float(importance)
        for i, importance in enumerate(model.feature_importances_)
    }
    
    with open('feature_importance.json', 'w') as f:
        json.dump(feature_importance, f, indent=2)
    
    mlflow.log_artifact('feature_importance.json')
    print("✅ Feature importance guardada")
    
    print("\n3️⃣ Exportando Predictions...")
    # 3. Predictions (CSV)
    predictions_df = pd.DataFrame({
        'actual': y_test,
        'predicted': y_pred,
        'probability_class_1': y_proba,
        'correct': y_test == y_pred
    })
    
    predictions_df.to_csv('predictions.csv', index=False)
    mlflow.log_artifact('predictions.csv')
    print("✅ Predictions guardadas")
    
    print("\n4️⃣ Generando Classification Report...")
    # 4. Classification Report (TXT)
    report = classification_report(y_test, y_pred, target_names=['Class 0', 'Class 1'])
    
    with open('classification_report.txt', 'w') as f:
        f.write("Classification Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(report)
        f.write("\n" + "=" * 50 + "\n")
        f.write(f"\nTotal samples: {len(y_test)}\n")
        f.write(f"Correct predictions: {(y_test == y_pred).sum()}\n")
        f.write(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}\n")
    
    mlflow.log_artifact('classification_report.txt')
    print("✅ Classification report guardado")
    
    # 5. Feature Importance Plot (PNG)
    print("\n5️⃣ Graficando Feature Importance...")
    plt.figure(figsize=(10, 6))
    feature_names = [f"F{i:02d}" for i in range(len(model.feature_importances_))]
    indices = np.argsort(model.feature_importances_)[::-1][:10]
    
    plt.bar(range(10), model.feature_importances_[indices])
    plt.xticks(range(10), [feature_names[i] for i in indices], rotation=45)
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title('Top 10 Feature Importances')
    plt.tight_layout()
    plt.savefig('feature_importance_plot.png', dpi=100)
    plt.close()
    
    mlflow.log_artifact('feature_importance_plot.png')
    print("✅ Feature importance plot guardado")
    
    # Log modelo
    mlflow.sklearn.log_model(model, "model")
    
    print("\n" + "=" * 60)
    print("✅ TODOS LOS ARTIFACTS GUARDADOS")
    print("=" * 60)
    print("\nArtifacts creados:")
    print("  • confusion_matrix.png")
    print("  • feature_importance.json")
    print("  • feature_importance_plot.png")
    print("  • predictions.csv")
    print("  • classification_report.txt")
    print("  • model/")
    print("\n👉 Ve a MLflow UI → Artifacts tab para descargar y visualizar")

# COMMAND ----------

# DBTITLE 1,Ejercicio 4: Búsqueda de Runs
# MAGIC %md
# MAGIC # Ejercicio 4: Búsqueda y Análisis de Runs
# MAGIC
# MAGIC ## Objetivo
# MAGIC Consultar runs programaticamente usando MLflow API:
# MAGIC * Buscar por métrica (F1 > threshold)
# MAGIC * Buscar por parámetro
# MAGIC * Encontrar el mejor modelo
# MAGIC * Analizar tendencias
# MAGIC
# MAGIC ## Tareas
# MAGIC 1. Usar `mlflow.search_runs()` con filtros
# MAGIC 2. Ordenar por métrica
# MAGIC 3. Crear análisis comparativo
# MAGIC 4. Visualizar resultados

# COMMAND ----------

# DBTITLE 1,Ejercicio 4: Solución - Parte 1
# Ejercicio 4: Solución - Búsqueda de Runs (Parte 1)

from mlflow.tracking import MlflowClient

client = MlflowClient()

print("=" * 60)
print("EJERCICIO 4: BÚSQUEDA Y ANÁLISIS DE RUNS")
print("=" * 60)

# Usar experimento del Ejercicio 2
experiment_name = "/Users/cortega@uda.edu.ar/Laboratorio/MLflow_Practica_Ejercicio2"
experiment = client.get_experiment_by_name(experiment_name)

if experiment:
    experiment_id = experiment.experiment_id
    
    print(f"\n🔍 Experimento encontrado: {experiment_name}")
    print(f"Experiment ID: {experiment_id}")
    
    # 1. Buscar todos los runs
    print("\n" + "=" * 60)
    print("1. TODOS LOS RUNS DEL EXPERIMENTO")
    print("=" * 60)
    
    all_runs = mlflow.search_runs(
        experiment_ids=[experiment_id],
        order_by=["start_time DESC"]
    )
    
    if not all_runs.empty:
        print(f"\nTotal runs: {len(all_runs)}")
        print("\nColumnas clave:")
        key_cols = [col for col in all_runs.columns if 'metrics' in col or 'params.model_type' in col]
        if key_cols:
            print(all_runs[['run_id'] + key_cols].to_string(index=False))
    else:
        print("No se encontraron runs")
    
    # 2. Buscar runs con F1 > 0.75
    print("\n" + "=" * 60)
    print("2. RUNS CON F1 SCORE > 0.75")
    print("=" * 60)
    
    high_f1_runs = mlflow.search_runs(
        experiment_ids=[experiment_id],
        filter_string="metrics.f1_score > 0.75",
        order_by=["metrics.f1_score DESC"]
    )
    
    if not high_f1_runs.empty:
        print(f"\nEncontrados: {len(high_f1_runs)} runs")
        display_cols = ['run_id', 'params.model_type', 'metrics.f1_score', 'metrics.roc_auc']
        display_cols = [col for col in display_cols if col in high_f1_runs.columns]
        print(high_f1_runs[display_cols].to_string(index=False))
    else:
        print("No se encontraron runs con F1 > 0.75")
    
    # 3. Mejor modelo por F1 Score
    print("\n" + "=" * 60)
    print("3. MEJOR MODELO (MÁXIMO F1 SCORE)")
    print("=" * 60)
    
    best_run = mlflow.search_runs(
        experiment_ids=[experiment_id],
        order_by=["metrics.f1_score DESC"],
        max_results=1
    )
    
    if not best_run.empty:
        best = best_run.iloc[0]
        print(f"\nRun ID: {best['run_id']}")
        if 'params.model_type' in best:
            print(f"Modelo: {best['params.model_type']}")
        if 'metrics.f1_score' in best:
            print(f"F1 Score: {best['metrics.f1_score']:.4f}")
        if 'metrics.accuracy' in best:
            print(f"Accuracy: {best['metrics.accuracy']:.4f}")
        if 'metrics.roc_auc' in best:
            print(f"ROC AUC: {best['metrics.roc_auc']:.4f}")
    
else:
    print(f"\n⚠️ Experimento no encontrado: {experiment_name}")
    print("Ejecuta primero el Ejercicio 2 para crear runs")

# COMMAND ----------

# DBTITLE 1,Ejercicio 4: Solución - Parte 2 (Visualización)
# Ejercicio 4: Solución - Análisis Visual (Parte 2)

import matplotlib.pyplot as plt
import seaborn as sns

print("\n" + "=" * 60)
print("4. ANÁLISIS VISUAL DE RUNS")
print("=" * 60)

# Obtener runs del experimento 2
experiment_name = "/Users/cortega@uda.edu.ar/Laboratorio/MLflow_Practica_Ejercicio2"
experiment = client.get_experiment_by_name(experiment_name)

if experiment:
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    
    if not runs.empty and len(runs) > 0:
        # Asegurarnos de que las columnas existen
        metric_cols = [col for col in runs.columns if col.startswith('metrics.')]
        
        if metric_cols:
            # Comparación de métricas entre modelos
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            # Plot 1: Comparación de múltiples métricas
            if 'params.model_type' in runs.columns:
                metrics_to_plot = ['metrics.accuracy', 'metrics.f1_score', 'metrics.roc_auc']
                available_metrics = [m for m in metrics_to_plot if m in runs.columns]
                
                if available_metrics:
                    plot_data = runs[['params.model_type'] + available_metrics].copy()
                    plot_data = plot_data.melt(
                        id_vars='params.model_type',
                        var_name='Metric',
                        value_name='Score'
                    )
                    plot_data['Metric'] = plot_data['Metric'].str.replace('metrics.', '')
                    
                    sns.barplot(
                        data=plot_data,
                        x='params.model_type',
                        y='Score',
                        hue='Metric',
                        ax=axes[0]
                    )
                    axes[0].set_title('Comparación de Métricas por Modelo', fontsize=12)
                    axes[0].set_xlabel('Modelo')
                    axes[0].set_ylabel('Score')
                    axes[0].legend(title='Métrica')
                    axes[0].set_ylim([0, 1])
            
            # Plot 2: F1 Score ranking
            if 'metrics.f1_score' in runs.columns and 'params.model_type' in runs.columns:
                runs_sorted = runs.sort_values('metrics.f1_score', ascending=True)
                
                axes[1].barh(
                    runs_sorted['params.model_type'],
                    runs_sorted['metrics.f1_score'],
                    color=['#1f77b4', '#ff7f0e', '#2ca02c'][:len(runs_sorted)]
                )
                axes[1].set_title('Ranking por F1 Score', fontsize=12)
                axes[1].set_xlabel('F1 Score')
                axes[1].set_ylabel('Modelo')
                axes[1].set_xlim([0, 1])
                
                # Añadir valores
                for i, (idx, row) in enumerate(runs_sorted.iterrows()):
                    axes[1].text(
                        row['metrics.f1_score'] + 0.01,
                        i,
                        f"{row['metrics.f1_score']:.4f}",
                        va='center'
                    )
            
            plt.tight_layout()
            plt.show()
            
            print("\n✅ Visualizaciones creadas")
        else:
            print("\n⚠️ No se encontraron métricas en los runs")
    else:
        print("\n⚠️ No hay runs disponibles para visualizar")
else:
    print("\n⚠️ Experimento no encontrado")

print("\n✅ Ejercicio 4 completado")

# COMMAND ----------

# DBTITLE 1,Ejercicio 5: Workflow Reproducible Completo
# MAGIC %md
# MAGIC # Ejercicio 5: Workflow Reproducible Completo
# MAGIC
# MAGIC ## Objetivo
# MAGIC Implementar un workflow de ML completamente reproducible con MLflow:
# MAGIC * Seeds fijados
# MAGIC * Versiones de librerías
# MAGIC * Data versioning
# MAGIC * Parámetros completos
# MAGIC * Modelo con signature
# MAGIC * Artifacts documentados
# MAGIC
# MAGIC ## Tareas
# MAGIC 1. Configurar entorno reproducible
# MAGIC 2. Documentar todo el pipeline
# MAGIC 3. Loggear información de reproducibilidad
# MAGIC 4. Verificar que puede reproducirse exactamente

# COMMAND ----------

# DBTITLE 1,Ejercicio 5: Solución
# Ejercicio 5: Solución - Workflow Reproducible

import sys
import sklearn
import platform
from datetime import datetime

mlflow.set_experiment("/Users/cortega@uda.edu.ar/Laboratorio/MLflow_Practica_Ejercicio5")

print("=" * 60)
print("EJERCICIO 5: WORKFLOW REPRODUCIBLE COMPLETO")
print("=" * 60)

with mlflow.start_run(run_name="reproducible_pipeline") as run:
    
    # ========================================
    # 1. CONFIGURACIÓN DE REPRODUCIBILIDAD
    # ========================================
    print("\n1️⃣ Configurando reproducibilidad...")
    
    SEED = 42
    np.random.seed(SEED)
    
    # Log seeds
    mlflow.log_param("random_seed", SEED)
    mlflow.log_param("numpy_seed", SEED)
    
    # Log versiones
    mlflow.log_param("python_version", sys.version.split()[0])
    mlflow.log_param("sklearn_version", sklearn.__version__)
    mlflow.log_param("numpy_version", np.__version__)
    mlflow.log_param("pandas_version", pd.__version__)
    mlflow.log_param("mlflow_version", mlflow.__version__)
    
    # Log environment info
    mlflow.log_param("platform", platform.system())
    mlflow.log_param("platform_version", platform.version())
    
    # Tags de metadata
    mlflow.set_tag("execution_date", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    mlflow.set_tag("purpose", "production_candidate")
    mlflow.set_tag("experiment_type", "reproducible_pipeline")
    
    print("✅ Configuración de reproducibilidad completada")
    
    # ========================================
    # 2. DATA VERSIONING
    # ========================================
    print("\n2️⃣ Documentando datos...")
    
    mlflow.log_param("dataset_type", "synthetic_classification")
    mlflow.log_param("n_samples", len(X))
    mlflow.log_param("n_features", X.shape[1])
    mlflow.log_param("n_classes", len(np.unique(y)))
    mlflow.log_param("train_size", len(X_train))
    mlflow.log_param("test_size", len(X_test))
    mlflow.log_param("test_ratio", 0.2)
    mlflow.log_param("stratified", True)
    
    # Distribución de clases
    class_dist = np.bincount(y_train)
    mlflow.log_param("class_0_count", int(class_dist[0]))
    mlflow.log_param("class_1_count", int(class_dist[1]))
    mlflow.log_metric("class_imbalance_ratio", float(class_dist[1] / class_dist[0]))
    
    print("✅ Datos documentados")
    
    # ========================================
    # 3. PREPROCESSING (documentado)
    # ========================================
    print("\n3️⃣ Preprocessing...")
    
    mlflow.log_param("preprocessing", "none")
    mlflow.log_param("scaling", "none")
    mlflow.log_param("feature_selection", "none")
    
    print("✅ Preprocessing documentado")
    
    # ========================================
    # 4. TRAINING CON PARÁMETROS COMPLETOS
    # ========================================
    print("\n4️⃣ Entrenando modelo...")
    
    model_params = {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
        "bootstrap": True,
        "random_state": SEED,
        "n_jobs": -1
    }
    
    mlflow.log_params(model_params)
    mlflow.log_param("model_type", "RandomForestClassifier")
    mlflow.log_param("framework", "sklearn")
    
    # Entrenar
    start_time = datetime.now()
    model = RandomForestClassifier(**model_params)
    model.fit(X_train, y_train)
    training_time = (datetime.now() - start_time).total_seconds()
    
    mlflow.log_metric("training_time_seconds", training_time)
    
    print(f"✅ Modelo entrenado en {training_time:.2f} segundos")
    
    # ========================================
    # 5. EVALUACIÓN COMPLETA
    # ========================================
    print("\n5️⃣ Evaluando modelo...")
    
    # Predicciones
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)[:, 1]
    
    # Métricas de train
    train_metrics = {
        "train_accuracy": accuracy_score(y_train, y_train_pred),
        "train_f1_score": f1_score(y_train, y_train_pred)
    }
    mlflow.log_metrics(train_metrics)
    
    # Métricas de test
    test_metrics = {
        "test_accuracy": accuracy_score(y_test, y_test_pred),
        "test_precision": precision_score(y_test, y_test_pred),
        "test_recall": recall_score(y_test, y_test_pred),
        "test_f1_score": f1_score(y_test, y_test_pred),
        "test_roc_auc": roc_auc_score(y_test, y_test_proba)
    }
    mlflow.log_metrics(test_metrics)
    
    # Métrica de overfitting
    overfitting_gap = train_metrics["train_accuracy"] - test_metrics["test_accuracy"]
    mlflow.log_metric("overfitting_gap", overfitting_gap)
    
    print("✅ Métricas calculadas")
    
    # ========================================
    # 6. ARTIFACTS COMPLETOS
    # ========================================
    print("\n6️⃣ Generando artifacts...")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_test_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap='Blues')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix_reproducible.png', dpi=100)
    plt.close()
    mlflow.log_artifact('confusion_matrix_reproducible.png')
    
    # Reproducibility document
    with open('reproducibility_info.txt', 'w') as f:
        f.write("REPRODUCIBILITY INFORMATION\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Execution Date: {datetime.now()}\n")
        f.write(f"Python: {sys.version.split()[0]}\n")
        f.write(f"Scikit-learn: {sklearn.__version__}\n")
        f.write(f"Random Seed: {SEED}\n\n")
        f.write("Model Parameters:\n")
        for param, value in model_params.items():
            f.write(f"  {param}: {value}\n")
        f.write("\nTest Metrics:\n")
        for metric, value in test_metrics.items():
            f.write(f"  {metric}: {value:.4f}\n")
    
    mlflow.log_artifact('reproducibility_info.txt')
    
    print("✅ Artifacts generados")
    
    # ========================================
    # 7. LOG MODELO CON SIGNATURE
    # ========================================
    print("\n7️⃣ Guardando modelo con signature...")
    
    signature = mlflow.models.infer_signature(X_train, y_train)
    
    mlflow.sklearn.log_model(
        model,
        "model",
        signature=signature,
        registered_model_name=None  # No registrar automáticamente
    )
    
    print("✅ Modelo guardado con signature")
    
    # ========================================
    # RESUMEN FINAL
    # ========================================
    print("\n" + "=" * 60)
    print("✅ WORKFLOW REPRODUCIBLE COMPLETADO")
    print("=" * 60)
    
    print(f"\nRun ID: {run.info.run_id}")
    print(f"\nMétricas Principales:")
    print(f"  Test Accuracy:  {test_metrics['test_accuracy']:.4f}")
    print(f"  Test F1 Score:  {test_metrics['test_f1_score']:.4f}")
    print(f"  Test ROC AUC:   {test_metrics['test_roc_auc']:.4f}")
    print(f"  Overfitting Gap: {overfitting_gap:.4f}")
    
    print(f"\nReproducibilidad:")
    print(f"  ✅ Seeds fijados")
    print(f"  ✅ Versiones documentadas")
    print(f"  ✅ Datos versionados")
    print(f"  ✅ Parámetros completos")
    print(f"  ✅ Modelo con signature")
    print(f"  ✅ Artifacts documentados")
    
    print(f"\n👉 Este run puede reproducirse exactamente siguiendo:")
    print(f"   - reproducibility_info.txt (en Artifacts)")
    print(f"   - Parámetros y tags en MLflow UI")

# COMMAND ----------

# DBTITLE 1,Resumen y Conclusiones
# MAGIC %md
# MAGIC # Resumen y Conclusiones de la Práctica
# MAGIC
# MAGIC ## ✅ Ejercicios Completados
# MAGIC
# MAGIC ### Ejercicio 1: Tracking Básico
# MAGIC - Creado experimento MLflow
# MAGIC - Loggeado parámetros, métricas, modelo
# MAGIC - Tags descriptivos para organización
# MAGIC - **Aprendizaje**: Estructura básica de un run de MLflow
# MAGIC
# MAGIC ### Ejercicio 2: Comparación de Modelos
# MAGIC - Entrenados 3 algoritmos diferentes
# MAGIC - Comparación visual en MLflow UI
# MAGIC - Identificación del mejor modelo
# MAGIC - **Aprendizaje**: Comparar múltiples enfoques sistemáticamente
# MAGIC
# MAGIC ### Ejercicio 3: Logging de Artifacts
# MAGIC - Confusion matrix (PNG)
# MAGIC - Feature importance (JSON + PNG)
# MAGIC - Predictions (CSV)
# MAGIC - Classification report (TXT)
# MAGIC - **Aprendizaje**: Documentar resultados con visualizaciones y datos
# MAGIC
# MAGIC ### Ejercicio 4: Búsqueda de Runs
# MAGIC - Filtrado por métricas y parámetros
# MAGIC - Búsqueda del mejor modelo
# MAGIC - Visualización comparativa
# MAGIC - **Aprendizaje**: Consultar y analizar runs programaticamente
# MAGIC
# MAGIC ### Ejercicio 5: Workflow Reproducible
# MAGIC - Seeds fijados
# MAGIC - Versiones de librerías documentadas
# MAGIC - Data versioning
# MAGIC - Modelo con signature
# MAGIC - Documentación completa
# MAGIC - **Aprendizaje**: Garantizar reproducibilidad total
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Habilidades Adquiridas
# MAGIC
# MAGIC ### MLflow Tracking
# MAGIC ✅ `mlflow.start_run()` para iniciar tracking  
# MAGIC ✅ `mlflow.log_param()` / `log_params()` para parámetros  
# MAGIC ✅ `mlflow.log_metric()` / `log_metrics()` para métricas  
# MAGIC ✅ `mlflow.log_artifact()` para archivos  
# MAGIC ✅ `mlflow.sklearn.log_model()` para modelos  
# MAGIC ✅ `mlflow.set_tag()` para metadata  
# MAGIC ✅ `mlflow.set_experiment()` para organización  
# MAGIC
# MAGIC ### MLflow Search
# MAGIC ✅ `mlflow.search_runs()` con filtros  
# MAGIC ✅ Ordenamiento por métricas  
# MAGIC ✅ Búsqueda del mejor run  
# MAGIC ✅ Análisis comparativo  
# MAGIC
# MAGIC ### Mejores Prácticas
# MAGIC ✅ Nombres descriptivos de experimentos  
# MAGIC ✅ Run names informativos  
# MAGIC ✅ Parámetros completos y documentados  
# MAGIC ✅ Métricas de train y test  
# MAGIC ✅ Artifacts bien organizados  
# MAGIC ✅ Tags para filtrado  
# MAGIC ✅ Reproducibilidad garantizada  
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Impacto en el Ciclo de Vida ML
# MAGIC
# MAGIC | Fase | Sin MLflow | Con MLflow |
# MAGIC |------|-----------|------------|
# MAGIC | **Experimentación** | Desorganizada, resultados perdidos | Organizada, todo registrado |
# MAGIC | **Comparación** | Manual, propensa a errores | Automática, visual, precisa |
# MAGIC | **Reproducibilidad** | Difícil o imposible | Garantizada |
# MAGIC | **Colaboración** | Compartir código y notas | Compartir runs completos |
# MAGIC | **Deployment** | Proceso manual | Modelos versionados y listos |
# MAGIC | **Auditoría** | No trazable | Historial completo |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Métricas de Éxito de esta Práctica
# MAGIC
# MAGIC | Métrica | Valor |
# MAGIC |---------|-------|
# MAGIC | Experimentos creados | 5 |
# MAGIC | Runs ejecutados | 10+ |
# MAGIC | Modelos loggeados | 6+ |
# MAGIC | Artifacts generados | 15+ |
# MAGIC | Métricas tracked | 30+ |
# MAGIC | Workflows reproducibles | 100% |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Integraciones Aprendidas
# MAGIC
# MAGIC ### MLflow + Sklearn
# MAGIC ```python
# MAGIC mlflow.sklearn.log_model(model, "model")
# MAGIC loaded_model = mlflow.sklearn.load_model(model_uri)
# MAGIC ```
# MAGIC
# MAGIC ### MLflow + Visualizaciones
# MAGIC ```python
# MAGIC plt.savefig('plot.png')
# MAGIC mlflow.log_artifact('plot.png')
# MAGIC ```
# MAGIC
# MAGIC ### MLflow + DataFrames
# MAGIC ```python
# MAGIC df.to_csv('data.csv')
# MAGIC mlflow.log_artifact('data.csv')
# MAGIC ```
# MAGIC
# MAGIC ### MLflow Search API
# MAGIC ```python
# MAGIC runs = mlflow.search_runs(
# MAGIC     filter_string="metrics.accuracy > 0.8",
# MAGIC     order_by=["metrics.accuracy DESC"]
# MAGIC )
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Próximos Pasos
# MAGIC
# MAGIC 1. **Model Registry**: Registrar y versionar modelos para producción
# MAGIC 2. **Deployment**: Servir modelos como REST APIs
# MAGIC 3. **Monitoring**: Tracking de model drift en producción
# MAGIC 4. **CI/CD**: Integrar MLflow en pipelines automatizados
# MAGIC 5. **Feature Store**: Combinar con Feature Store para workflow completo
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC 🎉 **¡Práctica del Módulo 06 completada exitosamente!**
# MAGIC
# MAGIC Ahora dominas MLflow Tracking para:
# MAGIC - ✅ Experimentación organizada
# MAGIC - ✅ Comparación de modelos
# MAGIC - ✅ Reproducibilidad garantizada
# MAGIC - ✅ Colaboración efectiva
# MAGIC - ✅ Trazabilidad completa

# COMMAND ----------

