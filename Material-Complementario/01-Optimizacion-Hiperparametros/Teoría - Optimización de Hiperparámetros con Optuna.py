# Databricks notebook source
# DBTITLE 1,Resultados de optimización
print("\n" + "="*80)
print("RESULTADOS DE LA OPTIMIZACIÓN")
print("="*80)

# Mejor trial
best_trial = study.best_trial

print(f"\n🏆 Mejores Hiperparámetros Encontrados:")
for key, value in best_trial.params.items():
    print(f"   {key}: {value}")

print(f"\n📊 Mejor MAE (validación cruzada): ${best_trial.value:,.2f}")
print(f"\n📉 Mejora vs. Baseline: ${mae_baseline - best_trial.value:,.2f} ({((mae_baseline - best_trial.value)/mae_baseline)*100:.2f}%)")

# Estadísticas del estudio
print(f"\n📊 Estadísticas del Estudio:")
print(f"   Total de trials: {len(study.trials)}")
print(f"   Trials completos: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
print(f"   Mejor trial: #{best_trial.number}")

# COMMAND ----------

# DBTITLE 1,Entrenar modelo final optimizado
print("\n" + "="*80)
print("MODELO FINAL CON HIPERPARÁMETROS OPTIMIZADOS")
print("="*80)

# Entrenar modelo con mejores hiperparámetros
model_optimized = RandomForestRegressor(**best_trial.params)
model_optimized.fit(X_train, y_train)

# Evaluar en test set
y_pred_optimized = model_optimized.predict(X_test)

mae_optimized = mean_absolute_error(y_test, y_pred_optimized)
rmse_optimized = np.sqrt(mean_squared_error(y_test, y_pred_optimized))
r2_optimized = r2_score(y_test, y_pred_optimized)

print(f"\n📊 Métricas en Test Set:")
print(f"   MAE:  ${mae_optimized:,.2f}")
print(f"   RMSE: ${rmse_optimized:,.2f}")
print(f"   R²:   {r2_optimized:.4f}")

# Comparación
print(f"\n🏆 Comparación Baseline vs. Optimizado:")
print(f"\n   {'Métrica':<10} {'Baseline':<15} {'Optimizado':<15} {'Mejora'}")
print(f"   {'-'*10} {'-'*15} {'-'*15} {'-'*15}")
print(f"   {'MAE':<10} ${mae_baseline:<14,.2f} ${mae_optimized:<14,.2f} {((mae_baseline-mae_optimized)/mae_baseline)*100:>6.2f}%")
print(f"   {'RMSE':<10} ${rmse_baseline:<14,.2f} ${rmse_optimized:<14,.2f} {((rmse_baseline-rmse_optimized)/rmse_baseline)*100:>6.2f}%")
print(f"   {'R²':<10} {mae_baseline:<15.4f} {r2_optimized:<15.4f} {((r2_optimized-r2_baseline)/r2_baseline)*100:>6.2f}%")

# COMMAND ----------

# DBTITLE 1,Visualización: Historia de optimización
print("\n" + "="*80)
print("VISUALIZACIONES DE OPTIMIZACIÓN")
print("="*80)

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Gráfico 1: Historia de optimización
fig = optuna.visualization.plot_optimization_history(study)
fig.update_layout(
    title="Historia de Optimización: Mejor MAE por Trial",
    xaxis_title="Número de Trial",
    yaxis_title="MAE ($)",
    height=500
)
fig.show()

print("\n📊 Gráfico 1: Historia de Optimización")
print("   Muestra cómo mejora el mejor MAE encontrado a lo largo de los trials")
print("   La línea azul es el mejor valor acumulado")

# COMMAND ----------

# DBTITLE 1,Visualización: Importancia de hiperparámetros
# Gráfico 2: Importancia de hiperparámetros
fig = optuna.visualization.plot_param_importances(study)
fig.update_layout(
    title="Importancia de Hiperparámetros",
    xaxis_title="Importancia",
    height=500
)
fig.show()

print("\n📊 Gráfico 2: Importancia de Hiperparámetros")
print("   Muestra qué hiperparámetros tienen mayor impacto en el resultado")
print("   Los más importantes son los que más deberías ajustar con cuidado")

# COMMAND ----------

# DBTITLE 1,Visualización: Relaciones entre hiperparámetros
# Gráfico 3: Slice plot (relación individual)
fig = optuna.visualization.plot_slice(study)
fig.update_layout(
    title="Relación entre Hiperparámetros y MAE",
    height=600
)
fig.show()

print("\n📊 Gráfico 3: Slice Plot")
print("   Muestra cómo cada hiperparámetro afecta individualmente al MAE")
print("   Puntos azules: trials individuales")
print("   Ayuda a entender rangos óptimos para cada parámetro")

# COMMAND ----------

# DBTITLE 1,Técnicas avanzadas - Teoría
# MAGIC %md
# MAGIC ## 5️⃣ Técnicas Avanzadas con Optuna
# MAGIC
# MAGIC ### ✏️ Pruning Automático
# MAGIC
# MAGIC **Concepto**: Detener trials poco prometedores **antes de terminar** el entrenamiento completo.
# MAGIC
# MAGIC **Ejemplo**: Si después de 10 epochs tu modelo tiene peor loss que el 75% de trials anteriores, deténlo.
# MAGIC
# MAGIC ✅ **Ventaja**: Ahorra tiempo al no entrenar modelos malos hasta el final.
# MAGIC
# MAGIC ```python
# MAGIC import optuna
# MAGIC from optuna.pruners import MedianPruner
# MAGIC
# MAGIC # Crear estudio con pruning
# MAGIC study = optuna.create_study(
# MAGIC     pruner=MedianPruner(
# MAGIC         n_startup_trials=5,  # No pruning en primeros 5 trials
# MAGIC         n_warmup_steps=3     # Esperar 3 steps antes de pruning
# MAGIC     )
# MAGIC )
# MAGIC ```
# MAGIC
# MAGIC **Tipos de Pruners**:
# MAGIC - **MedianPruner**: Poda si peor que mediana de trials
# MAGIC - **PercentilePruner**: Poda si peor que percentil X
# MAGIC - **SuccessiveHalvingPruner**: Estilo torneo (elimina mitad peor iterativamente)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 🚀 Paralelización
# MAGIC
# MAGIC **Concepto**: Ejecutar múltiples trials **simultáneamente** en varios cores.
# MAGIC
# MAGIC ```python
# MAGIC # Opción 1: Con joblib (varios procesos)
# MAGIC study.optimize(objective, n_trials=100, n_jobs=4)  # 4 procesos paralelos
# MAGIC
# MAGIC # Opción 2: Múltiples scripts apuntando a misma DB
# MAGIC # Script 1:
# MAGIC study = optuna.load_study(
# MAGIC     study_name='mi_estudio',
# MAGIC     storage='sqlite:///optuna.db'
# MAGIC )
# MAGIC study.optimize(objective, n_trials=50)
# MAGIC
# MAGIC # Script 2 (en paralelo):
# MAGIC study = optuna.load_study(
# MAGIC     study_name='mi_estudio',
# MAGIC     storage='sqlite:///optuna.db'
# MAGIC )
# MAGIC study.optimize(objective, n_trials=50)
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 💾 Persistencia en Base de Datos
# MAGIC
# MAGIC **Concepto**: Guardar progreso para reanudar o analizar después.
# MAGIC
# MAGIC ```python
# MAGIC # Crear estudio con storage
# MAGIC study = optuna.create_study(
# MAGIC     study_name='rf_optimization',
# MAGIC     storage='sqlite:///optuna_study.db',  # Base de datos local
# MAGIC     load_if_exists=True  # Continuar si ya existe
# MAGIC )
# MAGIC
# MAGIC # Entrenar
# MAGIC study.optimize(objective, n_trials=50)
# MAGIC
# MAGIC # Cargar más tarde
# MAGIC study_loaded = optuna.load_study(
# MAGIC     study_name='rf_optimization',
# MAGIC     storage='sqlite:///optuna_study.db'
# MAGIC )
# MAGIC ```
# MAGIC
# MAGIC ✅ **Ventaja**: Nunca pierdes progreso, incluso si se interrumpe.
# MAGIC
# MAGIC ---

# COMMAND ----------

# DBTITLE 1,Ejemplo: Pruning en práctica
print("\n" + "="*80)
print("EJEMPLO: PRUNING AUTOMÁTICO")
print("="*80)

from optuna.pruners import MedianPruner

# Función objetivo con pruning
def objective_with_pruning(trial):
    """
    Función objetivo que reporta valores intermedios para pruning.
    """
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 20),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'random_state': 42
    }
    
    # Entrenar incrementalmente y reportar valores intermedios
    from sklearn.model_selection import cross_validate
    
    model = RandomForestRegressor(**params)
    
    # CV con 3 folds, reportando resultados parciales
    scores = cross_val_score(
        model, X_train, y_train, 
        cv=3, 
        scoring='neg_mean_absolute_error'
    )
    
    # Reportar valores intermedios para pruning
    for step, score in enumerate(scores):
        trial.report(-score, step)
        
        # Verificar si debe hacer pruning
        if trial.should_prune():
            raise optuna.TrialPruned()
    
    return -scores.mean()

# Crear estudio con pruning
study_pruning = optuna.create_study(
    direction='minimize',
    pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=1)
)

print("\n🚀 Optimizando con pruning automático...")
print("   Trials que se vean poco prometedores serán detenidos temprano\n")

study_pruning.optimize(objective_with_pruning, n_trials=30, show_progress_bar=True)

print(f"\n📊 Resultados con Pruning:")
print(f"   Trials completos: {len([t for t in study_pruning.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
print(f"   Trials podados: {len([t for t in study_pruning.trials if t.state == optuna.trial.TrialState.PRUNED])}")
print(f"   Mejor MAE: ${study_pruning.best_value:,.2f}")
print(f"\n   ⚙️ Ahorro: {len([t for t in study_pruning.trials if t.state == optuna.trial.TrialState.PRUNED])} trials detenidos temprano")

# COMMAND ----------

# DBTITLE 1,Ejercicios prácticos
# MAGIC %md
# MAGIC ## 6️⃣ Ejercicios Prácticos
# MAGIC
# MAGIC ### 📝 Ejercicio 1: Optimizar Gradient Boosting
# MAGIC
# MAGIC **Objetivo**: Usar Optuna para optimizar un modelo GradientBoostingRegressor.
# MAGIC
# MAGIC **Tareas**:
# MAGIC 1. Definir función objetivo para GradientBoostingRegressor
# MAGIC 2. Hiperparámetros a optimizar:
# MAGIC    - `n_estimators`: 50-500
# MAGIC    - `learning_rate`: 0.01-0.3 (log scale)
# MAGIC    - `max_depth`: 3-10
# MAGIC    - `subsample`: 0.5-1.0
# MAGIC    - `min_samples_split`: 2-20
# MAGIC 3. Ejecutar 50 trials
# MAGIC 4. Comparar con Random Forest optimizado
# MAGIC
# MAGIC **Pista**: Usa `trial.suggest_float('learning_rate', 0.01, 0.3, log=True)` para escala logarítmica.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 📝 Ejercicio 2: Multi-objetivo
# MAGIC
# MAGIC **Objetivo**: Optimizar **dos métricas simultáneamente**: MAE y tiempo de entrenamiento.
# MAGIC
# MAGIC **Tareas**:
# MAGIC 1. Modificar función objetivo para retornar tupla `(mae, tiempo)`
# MAGIC 2. Crear estudio multi-objetivo: `optuna.create_study(directions=['minimize', 'minimize'])`
# MAGIC 3. Analizar Pareto front (trade-off entre precisión y velocidad)
# MAGIC 4. Seleccionar modelo según prioridad del negocio
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 📝 Ejercicio 3: Persistencia y Continuación
# MAGIC
# MAGIC **Objetivo**: Guardar estudio en base de datos y continuar después.
# MAGIC
# MAGIC **Tareas**:
# MAGIC 1. Crear estudio con storage SQLite
# MAGIC 2. Ejecutar 25 trials
# MAGIC 3. "Interrumpir" (simular)
# MAGIC 4. Cargar estudio y ejecutar 25 trials adicionales
# MAGIC 5. Verificar que el total sea 50 trials
# MAGIC
# MAGIC **Pista**: Usa `storage='sqlite:///mi_estudio.db'`
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 📝 Ejercicio 4: Features Espaciales
# MAGIC
# MAGIC **Objetivo**: Incorporar features geoespaciales H3 y optimizar.
# MAGIC
# MAGIC **Tareas**:
# MAGIC 1. Agregar features espaciales al dataset:
# MAGIC    - `dist_sucursal_min`
# MAGIC    - `densidad_zona`
# MAGIC    - `facturacion_promedio_zona`
# MAGIC 2. Re-optimizar Random Forest con features ampliadas
# MAGIC 3. Comparar mejora vs. modelo sin features espaciales
# MAGIC 4. Analizar importancia de features espaciales
# MAGIC
# MAGIC ---

# COMMAND ----------

# DBTITLE 1,Conclusiones y mejores prácticas
# MAGIC %md
# MAGIC ## ✅ Conclusiones y Mejores Prácticas
# MAGIC
# MAGIC ### 🎯 Resumen del Módulo
# MAGIC
# MAGIC **Lo que aprendimos**:
# MAGIC
# MAGIC 1. ✅ Diferencia entre **parámetros** e **hiperparámetros**
# MAGIC 2. ✅ Métodos tradicionales: **Grid Search** y **Random Search**
# MAGIC 3. ✅ **Optuna**: Búsqueda bayesiana eficiente
# MAGIC 4. ✅ **Pruning**: Detener trials poco prometedores
# MAGIC 5. ✅ **Visualizaciones**: Entender proceso de optimización
# MAGIC 6. ✅ **Persistencia**: Guardar progreso en base de datos
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 💡 Mejores Prácticas
# MAGIC
# MAGIC #### 1. **Definir Budget de Tiempo**
# MAGIC
# MAGIC ```python
# MAGIC # Opción A: Número de trials
# MAGIC study.optimize(objective, n_trials=100)
# MAGIC
# MAGIC # Opción B: Tiempo límite
# MAGIC import datetime
# MAGIC timeout = 60 * 30  # 30 minutos
# MAGIC study.optimize(objective, timeout=timeout)
# MAGIC ```
# MAGIC
# MAGIC #### 2. **Usar Validación Cruzada**
# MAGIC
# MAGIC ✅ **BIEN**: Validación cruzada (más robusto)
# MAGIC ```python
# MAGIC scores = cross_val_score(model, X_train, y_train, cv=5)
# MAGIC return -scores.mean()
# MAGIC ```
# MAGIC
# MAGIC ❌ **MAL**: Train/validation split simple (puede overfittear)
# MAGIC ```python
# MAGIC model.fit(X_train, y_train)
# MAGIC score = model.score(X_val, y_val)
# MAGIC return -score
# MAGIC ```
# MAGIC
# MAGIC #### 3. **Rangos Sensatos de Hiperparámetros**
# MAGIC
# MAGIC ✅ **BIEN**: Rangos informados
# MAGIC ```python
# MAGIC 'n_estimators': trial.suggest_int('n_estimators', 50, 300)  # Rango razonable
# MAGIC 'max_depth': trial.suggest_int('max_depth', 3, 20)         # No demasiado profundo
# MAGIC ```
# MAGIC
# MAGIC ❌ **MAL**: Rangos demasiado amplios
# MAGIC ```python
# MAGIC 'n_estimators': trial.suggest_int('n_estimators', 1, 10000)  # Muy amplio
# MAGIC 'max_depth': trial.suggest_int('max_depth', 1, 1000)        # Excesivo
# MAGIC ```
# MAGIC
# MAGIC #### 4. **Escala Logarítmica para Learning Rate**
# MAGIC
# MAGIC ```python
# MAGIC # Para hiperparámetros que actúan en escala log
# MAGIC 'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True)
# MAGIC ```
# MAGIC
# MAGIC #### 5. **Guardar Mejor Modelo**
# MAGIC
# MAGIC ```python
# MAGIC # Al final de la optimización
# MAGIC import joblib
# MAGIC
# MAGIC model_final = RandomForestRegressor(**study.best_params)
# MAGIC model_final.fit(X_train, y_train)
# MAGIC
# MAGIC joblib.dump(model_final, 'best_model.pkl')
# MAGIC print("✅ Mejor modelo guardado")
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 🚀 Cuándo Usar Optuna
# MAGIC
# MAGIC ✅ **Usar Optuna cuando**:
# MAGIC - Tienes muchos hiperparámetros (>5)
# MAGIC - Grid Search es demasiado lento
# MAGIC - Necesitas optimización rápida
# MAGIC - Quieres visualizaciones automáticas
# MAGIC - Necesitas persistencia
# MAGIC
# MAGIC ❌ **NO usar Optuna cuando**:
# MAGIC - Solo 1-2 hiperparámetros (manual es más rápido)
# MAGIC - Dataset muy pequeño (<1000 registros)
# MAGIC - Modelo entrena en <1 segundo
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 📚 Recursos Adicionales
# MAGIC
# MAGIC - **Documentación Oficial**: https://optuna.readthedocs.io/
# MAGIC - **Tutoriales**: https://optuna.org/#code_examples
# MAGIC - **Paper Original**: "Optuna: A Next-generation Hyperparameter Optimization Framework"
# MAGIC - **Comparaciones**: https://github.com/optuna/optuna-examples
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🎓 ¡Felicitaciones!
# MAGIC
# MAGIC **Has completado el módulo de Optimización de Hiperparámetros con Optuna.**
# MAGIC
# MAGIC Ahora puedes:
# MAGIC - ✅ Optimizar modelos de forma eficiente
# MAGIC - ✅ Usar técnicas avanzadas (pruning, paralelización)
# MAGIC - ✅ Interpretar visualizaciones de optimización
# MAGIC - ✅ Aplicar mejores prácticas en proyectos reales
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC **Universidad del Aconcagua**  
# MAGIC **Laboratorio (Herramientas)**  
# MAGIC **Mendoza, Argentina**

# COMMAND ----------

# DBTITLE 1,Título del notebook
# MAGIC %md
# MAGIC # 🎯 Optimización de Hiperparámetros con Optuna
# MAGIC ## Material Complementario - Laboratorio (Herramientas)
# MAGIC ### Universidad del Aconcagua - Mendoza, Argentina
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 🎯 Objetivos de Aprendizaje
# MAGIC
# MAGIC 1. Comprender qué son los **hiperparámetros** y por qué optimizarlos
# MAGIC 2. Conocer métodos tradicionales (**Grid Search**, **Random Search**)
# MAGIC 3. Aprender a usar **Optuna** para optimización eficiente
# MAGIC 4. Aplicar técnicas avanzadas: **pruning**, **paralelización**, **visualización**
# MAGIC 5. Comparar resultados entre métodos
# MAGIC
# MAGIC ### 📁 Contenido
# MAGIC
# MAGIC 1. Introducción a Hiperparámetros
# MAGIC 2. Métodos Tradicionales de Optimización
# MAGIC 3. Introducción a Optuna
# MAGIC 4. Ejemplo Práctico: Predicción de Ventas de Panadería
# MAGIC 5. Técnicas Avanzadas
# MAGIC 6. Ejercicios Prácticos
# MAGIC
# MAGIC ### ⏱️ Duración Estimada: 2 horas
# MAGIC
# MAGIC ---

# COMMAND ----------

# DBTITLE 1,1. ¿Qué son los hiperparámetros?
# MAGIC %md
# MAGIC ## 1️⃣ ¿Qué son los Hiperparámetros?
# MAGIC
# MAGIC ### Diferencia: Parámetros vs. Hiperparámetros
# MAGIC
# MAGIC #### 📊 **Parámetros**
# MAGIC - Son **aprendidos por el modelo** durante el entrenamiento
# MAGIC - Ejemplos: Pesos en redes neuronales, coeficientes en regresión lineal
# MAGIC - Se ajustan automáticamente con los datos
# MAGIC
# MAGIC ```python
# MAGIC # Ejemplo: En regresión lineal y = mx + b
# MAGIC m, b  # <- PARÁMETROS (aprendidos)
# MAGIC ```
# MAGIC
# MAGIC #### ⚙️ **Hiperparámetros**
# MAGIC - Son **configuraciones externas** que controlamos ANTES del entrenamiento
# MAGIC - Ejemplos: Número de árboles en Random Forest, learning rate, profundidad máxima
# MAGIC - **NO** se aprenden automáticamente, los definimos nosotros
# MAGIC
# MAGIC ```python
# MAGIC # Ejemplo: Random Forest
# MAGIC RandomForestClassifier(
# MAGIC     n_estimators=100,      # <- HIPERPARÁMETRO
# MAGIC     max_depth=10,          # <- HIPERPARÁMETRO
# MAGIC     min_samples_split=5    # <- HIPERPARÁMETRO
# MAGIC )
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 🎯 Por qué Importan los Hiperparámetros
# MAGIC
# MAGIC Los hiperparámetros tienen **impacto directo** en:
# MAGIC
# MAGIC 1. **Precisión del modelo**: Configuración incorrecta → bajo rendimiento
# MAGIC 2. **Overfitting/Underfitting**: 
# MAGIC    - `max_depth` muy alto → overfitting
# MAGIC    - `max_depth` muy bajo → underfitting
# MAGIC 3. **Tiempo de entrenamiento**: Más árboles = más tiempo
# MAGIC 4. **Memoria utilizada**: Modelos más complejos = más RAM
# MAGIC
# MAGIC **Ejemplo real**:
# MAGIC
# MAGIC | max_depth | Accuracy Train | Accuracy Test | Conclusión |
# MAGIC |-----------|----------------|---------------|------------|
# MAGIC | 3 | 75% | 74% | 🟡 Underfitting |
# MAGIC | 10 | 92% | 89% | ✅ Equilibrado |
# MAGIC | 50 | 100% | 78% | 🔴 Overfitting |
# MAGIC
# MAGIC ✅ **Conclusión**: Encontrar los hiperparámetros óptimos es crucial.
# MAGIC
# MAGIC ---

# COMMAND ----------

# DBTITLE 1,2. Métodos Tradicionales
# MAGIC %md
# MAGIC ## 2️⃣ Métodos Tradicionales de Optimización
# MAGIC
# MAGIC ### 🔲 Grid Search (Búsqueda Exhaustiva)
# MAGIC
# MAGIC **Concepto**: Prueba **todas las combinaciones posibles** de hiperparámetros.
# MAGIC
# MAGIC ```python
# MAGIC from sklearn.model_selection import GridSearchCV
# MAGIC
# MAGIC param_grid = {
# MAGIC     'n_estimators': [50, 100, 200],
# MAGIC     'max_depth': [5, 10, 15, 20],
# MAGIC     'min_samples_split': [2, 5, 10]
# MAGIC }
# MAGIC
# MAGIC # Total de combinaciones: 3 x 4 x 3 = 36 modelos a entrenar
# MAGIC ```
# MAGIC
# MAGIC ✅ **Ventajas**:
# MAGIC - Garantiza encontrar la mejor combinación en el grid
# MAGIC - Fácil de entender e implementar
# MAGIC
# MAGIC ❌ **Desventajas**:
# MAGIC - 🐢 **Extremadamente lento** con muchos hiperparámetros
# MAGIC - Explosión combinatoria: 10 parámetros con 5 valores = 9,765,625 combinaciones!
# MAGIC - Desperdicia recursos en zonas poco prometedoras
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 🎲 Random Search (Búsqueda Aleatoria)
# MAGIC
# MAGIC **Concepto**: Prueba combinaciones **aleatorias** de hiperparámetros.
# MAGIC
# MAGIC ```python
# MAGIC from sklearn.model_selection import RandomizedSearchCV
# MAGIC
# MAGIC param_dist = {
# MAGIC     'n_estimators': [50, 100, 150, 200, 250],
# MAGIC     'max_depth': [5, 10, 15, 20, 25, 30],
# MAGIC     'min_samples_split': [2, 5, 10, 15]
# MAGIC }
# MAGIC
# MAGIC # Prueba solo 20 combinaciones aleatorias (de miles posibles)
# MAGIC random_search = RandomizedSearchCV(
# MAGIC     estimator=model,
# MAGIC     param_distributions=param_dist,
# MAGIC     n_iter=20  # Número de combinaciones a probar
# MAGIC )
# MAGIC ```
# MAGIC
# MAGIC ✅ **Ventajas**:
# MAGIC - Más rápido que Grid Search
# MAGIC - Explora más espacio con menos iteraciones
# MAGIC
# MAGIC ❌ **Desventajas**:
# MAGIC - No aprende de iteraciones anteriores
# MAGIC - Puede perder combinaciones óptimas
# MAGIC - Aun puede ser lento
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 📈 Comparación Visual
# MAGIC
# MAGIC **Grid Search vs. Random Search**:
# MAGIC
# MAGIC ```
# MAGIC Grid Search (exhaustivo):        Random Search:
# MAGIC
# MAGIC   [X] [X] [X] [X] [X]              [ ] [X] [ ] [ ] [X]
# MAGIC   [X] [X] [X] [X] [X]              [X] [ ] [ ] [X] [ ]
# MAGIC   [X] [X] [X] [X] [X]              [ ] [ ] [X] [ ] [ ]
# MAGIC   [X] [X] [X] [X] [X]              [ ] [X] [ ] [X] [ ]
# MAGIC   [X] [X] [X] [X] [X]              [X] [ ] [ ] [ ] [X]
# MAGIC ```
# MAGIC
# MAGIC ✅ Random Search cubre más área con menos iteraciones
# MAGIC
# MAGIC ---

# COMMAND ----------

# DBTITLE 1,3. Introducción a Optuna
# MAGIC %md
# MAGIC ## 3️⃣ Introducción a Optuna
# MAGIC
# MAGIC ### 🧠 ¿Qué es Optuna?
# MAGIC
# MAGIC **Optuna** es un framework de optimización de hiperparámetros que usa **búsqueda bayesiana** y **pruning automático**.
# MAGIC
# MAGIC #### Características Clave
# MAGIC
# MAGIC 1. ⚙️ **Optimización Automática**: Decide qué combinaciones probar próximamente
# MAGIC 2. ✏️ **Pruning Inteligente**: Detiene entrenamientos poco prometedores
# MAGIC 3. 🚀 **Paralelización**: Ejecuta múltiples trials simultáneamente
# MAGIC 4. 📊 **Visualización**: Gráficos de optimización y relaciones entre hiperparámetros
# MAGIC 5. 💾 **Persistencia**: Guarda progreso en base de datos
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 🔍 Cómo Funciona Optuna
# MAGIC
# MAGIC #### Búsqueda Bayesiana Simplificada
# MAGIC
# MAGIC Optuna usa **Tree-structured Parzen Estimator (TPE)** por defecto:
# MAGIC
# MAGIC 1. 🎯 **Inicio**: Prueba combinaciones aleatorias
# MAGIC 2. 🧠 **Aprende**: Modela qué hiperparámetros funcionan mejor
# MAGIC 3. 🎯 **Explota**: Sugiere valores en zonas prometedoras
# MAGIC 4. 🔄 **Explora**: Ocasionalmente prueba zonas nuevas
# MAGIC 5. 🔁 **Repite**: Converge hacia óptimo
# MAGIC
# MAGIC **Ventaja**: Aprende de cada trial y sugiere mejores valores.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 🎯 Ventajas de Optuna vs. Métodos Tradicionales
# MAGIC
# MAGIC | Característica | Grid Search | Random Search | Optuna |
# MAGIC |------------------|-------------|---------------|--------|
# MAGIC | Velocidad | 🐢 Muy lento | 🐇 Rápido | 🚀 Muy rápido |
# MAGIC | Aprende de trials | ❌ No | ❌ No | ✅ Sí |
# MAGIC | Pruning automático | ❌ No | ❌ No | ✅ Sí |
# MAGIC | Paralelización | ✅ Sí | ✅ Sí | ✅ Sí |
# MAGIC | Fácil uso | ✅ Sí | ✅ Sí | ✅ Sí |
# MAGIC | Visualizaciones | ❌ No | ❌ No | ✅ Sí |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 📦 Instalación
# MAGIC
# MAGIC ```python
# MAGIC %pip install optuna --quiet
# MAGIC ```
# MAGIC
# MAGIC ---

# COMMAND ----------

# DBTITLE 1,4. Setup y carga de datos
# Instalar Optuna
%pip install optuna==3.5.0 --quiet

# Importar librerías
import pandas as pd
import numpy as np
import optuna
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

print("✅ Librerías importadas")
print(f"   Optuna versión: {optuna.__version__}")

# COMMAND ----------

# DBTITLE 1,Cargar datos de la panadería
# Cargar datasets
ruta_datos = '/Workspace/Users/cortega@uda.edu.ar/Laboratorio/Datasets/'

df_ventas = pd.read_csv(ruta_datos + 'ventas.csv')
df_clientes = pd.read_csv(ruta_datos + 'clientes.csv')
df_productos = pd.read_csv(ruta_datos + 'productos.csv')
df_detalles = pd.read_csv(ruta_datos + 'detalles_ventas.csv')

print("✅ Datasets cargados")
print(f"   Ventas: {len(df_ventas):,}")
print(f"   Clientes: {len(df_clientes):,}")
print(f"   Productos: {len(df_productos):,}")

# COMMAND ----------

# DBTITLE 1,Preparar datos para ML
print("="*80)
print("PREPARACIÓN DE DATOS PARA MODELO")
print("="*80)

# Convertir fecha
df_ventas['fecha'] = pd.to_datetime(df_ventas['fecha'])

# Crear features temporales
df_ventas['dia_semana'] = df_ventas['fecha'].dt.dayofweek
df_ventas['dia_mes'] = df_ventas['fecha'].dt.day
df_ventas['mes'] = df_ventas['fecha'].dt.month
df_ventas['es_fin_de_semana'] = df_ventas['dia_semana'].isin([5, 6]).astype(int)

# Unir con clientes para features
df_ml = df_ventas[df_ventas['cliente_id'].notna()].merge(
    df_clientes[['cliente_id', 'segmento']], 
    on='cliente_id',
    how='left'
)

# Codificar categorías
df_ml['segmento_encoded'] = df_ml['segmento'].astype('category').cat.codes

# Features finales
features = [
    'sucursal_id', 
    'dia_semana', 
    'dia_mes', 
    'mes', 
    'es_fin_de_semana',
    'segmento_encoded'
]

target = 'total'

X = df_ml[features]
y = df_ml[target]

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\n📊 Datos preparados:")
print(f"   Train: {len(X_train):,} registros")
print(f"   Test:  {len(X_test):,} registros")
print(f"   Features: {len(features)}")
print(f"   Target: {target} (predicción de monto de venta)")

# COMMAND ----------

# DBTITLE 1,Baseline sin optimizar
print("\n" + "="*80)
print("BASELINE: MODELO SIN OPTIMIZAR")
print("="*80)

# Entrenar modelo con hiperparámetros por defecto
model_baseline = RandomForestRegressor(random_state=42)
model_baseline.fit(X_train, y_train)

# Evaluar
y_pred_baseline = model_baseline.predict(X_test)

mae_baseline = mean_absolute_error(y_test, y_pred_baseline)
rmse_baseline = np.sqrt(mean_squared_error(y_test, y_pred_baseline))
r2_baseline = r2_score(y_test, y_pred_baseline)

print(f"\n📊 Métricas Baseline (hiperparámetros por defecto):")
print(f"   MAE:  ${mae_baseline:,.2f}")
print(f"   RMSE: ${rmse_baseline:,.2f}")
print(f"   R²:   {r2_baseline:.4f}")

print(f"\n⚙️ Hiperparámetros usados:")
print(f"   n_estimators: {model_baseline.n_estimators}")
print(f"   max_depth: {model_baseline.max_depth}")
print(f"   min_samples_split: {model_baseline.min_samples_split}")
print(f"   min_samples_leaf: {model_baseline.min_samples_leaf}")

# COMMAND ----------

# DBTITLE 1,Función objetivo para Optuna
# MAGIC %md
# MAGIC ### 🎯 Definir Función Objetivo para Optuna
# MAGIC
# MAGIC La función objetivo:
# MAGIC 1. Recibe un `trial` (sugiere hiperparámetros)
# MAGIC 2. Entrena el modelo
# MAGIC 3. Evalúa con validación cruzada
# MAGIC 4. Retorna la métrica a **minimizar** (MAE en este caso)
# MAGIC
# MAGIC **Importante**: Optuna **minimiza** por defecto. Para maximizar (ej. R²), retornar `-r2`.

# COMMAND ----------

# DBTITLE 1,Definir función objetivo
def objective(trial):
    """
    Función objetivo para Optuna.
    Sugiere hiperparámetros, entrena modelo y retorna MAE.
    """
    
    # Sugerir hiperparámetros
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 20),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        'random_state': 42
    }
    
    # Crear modelo con hiperparámetros sugeridos
    model = RandomForestRegressor(**params)
    
    # Validación cruzada (3-fold para velocidad)
    scores = cross_val_score(
        model, 
        X_train, 
        y_train, 
        cv=3, 
        scoring='neg_mean_absolute_error',
        n_jobs=-1
    )
    
    # Retornar MAE promedio (valor absoluto)
    mae = -scores.mean()
    
    return mae

print("✅ Función objetivo definida")

# COMMAND ----------

# DBTITLE 1,Ejecutar optimización con Optuna
print("\n" + "="*80)
print("OPTIMIZACIÓN CON OPTUNA")
print("="*80)

# Crear estudio
study = optuna.create_study(
    direction='minimize',  # Minimizar MAE
    study_name='rf_optimization',
    sampler=optuna.samplers.TPESampler(seed=42)  # Tree-structured Parzen Estimator
)

print(f"\n🚀 Iniciando optimización...")
print(f"   Algoritmo: TPE (Tree-structured Parzen Estimator)")
print(f"   Métrica: MAE (minimizar)")
print(f"   Trials: 50")
print(f"\n   Progreso:")

# Ejecutar optimización
study.optimize(
    objective, 
    n_trials=50,  # Número de combinaciones a probar
    show_progress_bar=True
)

print(f"\n✅ Optimización completada")

# COMMAND ----------

