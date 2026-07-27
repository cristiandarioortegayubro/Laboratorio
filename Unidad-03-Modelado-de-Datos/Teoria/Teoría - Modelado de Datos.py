# Databricks notebook source
# DBTITLE 1,Contenido Completo U3
# MAGIC %md
# MAGIC # 📊 Unidad 3: Modelado de Datos y Machine Learning
# MAGIC ## Laboratorio (Herramientas) - Universidad del Aconcagua
# MAGIC ### Contenido Teórico
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🎯 Objetivos de Aprendizaje
# MAGIC
# MAGIC 1. ✅ Comprender Delta Lake y sus ventajas
# MAGIC 2. ✅ Dominar agregaciones y window functions
# MAGIC 3. ✅ Aplicar feature engineering efectivo
# MAGIC 4. ✅ Entender fundamentos de machine learning
# MAGIC 5. ✅ Entrenar y evaluar modelos básicos
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 1️⃣ Delta Lake: Almacenamiento Optimizado
# MAGIC
# MAGIC ### ¿Qué es Delta Lake?
# MAGIC
# MAGIC **Delta Lake** es una capa de almacenamiento open-source que aporta confiabilidad a data lakes.
# MAGIC
# MAGIC ### Características Clave
# MAGIC
# MAGIC ⚡ **ACID Transactions**
# MAGIC * Atomicidad: Todo o nada
# MAGIC * Consistencia: Datos siempre válidos
# MAGIC * Aislamiento: Transacciones concurrentes seguras
# MAGIC * Durabilidad: Cambios permanentes
# MAGIC
# MAGIC 🕙 **Time Travel**
# MAGIC ```sql
# MAGIC -- Ver datos de hace 7 días
# MAGIC SELECT * FROM tabla VERSION AS OF 7 DAYS AGO
# MAGIC
# MAGIC -- Ver versión específica
# MAGIC SELECT * FROM tabla@v5
# MAGIC ```
# MAGIC
# MAGIC 🔄 **Schema Evolution**
# MAGIC ```python
# MAGIC # Agregar columna sin reescribir toda la tabla
# MAGIC df.write.mode('append').option('mergeSchema', 'true').saveAsTable('tabla')
# MAGIC ```
# MAGIC
# MAGIC 🚀 **Performance**
# MAGIC * Formato columnar (Parquet optimizado)
# MAGIC * Z-ordering para co-locality
# MAGIC * Data skipping automático
# MAGIC
# MAGIC ### Crear Tablas Delta
# MAGIC
# MAGIC ```python
# MAGIC # Desde DataFrame
# MAGIC df.write.format('delta').mode('overwrite').saveAsTable('catalogo.esquema.tabla')
# MAGIC
# MAGIC # Con particionamiento
# MAGIC df.write.format('delta') \
# MAGIC     .partitionBy('fecha', 'region') \
# MAGIC     .saveAsTable('ventas')
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 2️⃣ Agregaciones Avanzadas
# MAGIC
# MAGIC ### GROUP BY Básico
# MAGIC
# MAGIC ```python
# MAGIC # Pandas
# MAGIC df.groupby('categoria').agg({
# MAGIC     'precio': ['mean', 'min', 'max'],
# MAGIC     'ventas': 'sum'
# MAGIC })
# MAGIC
# MAGIC # PySpark
# MAGIC df.groupBy('categoria').agg(
# MAGIC     F.mean('precio').alias('precio_promedio'),
# MAGIC     F.sum('ventas').alias('total_ventas')
# MAGIC )
# MAGIC ```
# MAGIC
# MAGIC ### Window Functions
# MAGIC
# MAGIC **Funciones de ventana** permiten cálculos sobre particiones de datos sin colapsar filas.
# MAGIC
# MAGIC ```python
# MAGIC from pyspark.sql import Window
# MAGIC
# MAGIC # Definir ventana
# MAGIC window_spec = Window.partitionBy('categoria').orderBy('fecha')
# MAGIC
# MAGIC # Ranking
# MAGIC df.withColumn('ranking', F.row_number().over(window_spec))
# MAGIC
# MAGIC # Acumulado
# MAGIC df.withColumn('ventas_acum', F.sum('ventas').over(window_spec))
# MAGIC
# MAGIC # Comparación con periodo anterior
# MAGIC df.withColumn('ventas_mes_anterior', F.lag('ventas', 1).over(window_spec))
# MAGIC ```
# MAGIC
# MAGIC ### Funciones Window Útiles
# MAGIC
# MAGIC | Función | Descripción |
# MAGIC |---|---|
# MAGIC | `ROW_NUMBER()` | Número de fila único |
# MAGIC | `RANK()` | Ranking con gaps |
# MAGIC | `DENSE_RANK()` | Ranking sin gaps |
# MAGIC | `LAG()` | Valor de fila anterior |
# MAGIC | `LEAD()` | Valor de fila siguiente |
# MAGIC | `SUM() OVER()` | Suma acumulada |
# MAGIC | `AVG() OVER()` | Promedio móvil |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 3️⃣ Feature Engineering
# MAGIC
# MAGIC ### ¿Qué es Feature Engineering?
# MAGIC
# MAGIC El proceso de **transformar datos crudos en features** que mejoran el rendimiento de modelos ML.
# MAGIC
# MAGIC ### Técnicas Comunes
# MAGIC
# MAGIC #### 📅 **Features Temporales**
# MAGIC ```python
# MAGIC df['anio'] = df['fecha'].dt.year
# MAGIC df['mes'] = df['fecha'].dt.month
# MAGIC df['dia_semana'] = df['fecha'].dt.dayofweek
# MAGIC df['es_fin_semana'] = (df['dia_semana'] >= 5).astype(int)
# MAGIC df['trimestre'] = df['fecha'].dt.quarter
# MAGIC ```
# MAGIC
# MAGIC #### 🔢 **Encoding Categórico**
# MAGIC
# MAGIC **One-Hot Encoding:**
# MAGIC ```python
# MAGIC import pandas as pd
# MAGIC df_encoded = pd.get_dummies(df, columns=['categoria'])
# MAGIC ```
# MAGIC
# MAGIC **Label Encoding:**
# MAGIC ```python
# MAGIC from sklearn.preprocessing import LabelEncoder
# MAGIC le = LabelEncoder()
# MAGIC df['categoria_encoded'] = le.fit_transform(df['categoria'])
# MAGIC ```
# MAGIC
# MAGIC #### 📊 **Escalado**
# MAGIC
# MAGIC **StandardScaler** (media=0, std=1):
# MAGIC ```python
# MAGIC from sklearn.preprocessing import StandardScaler
# MAGIC scaler = StandardScaler()
# MAGIC df[['precio', 'cantidad']] = scaler.fit_transform(df[['precio', 'cantidad']])
# MAGIC ```
# MAGIC
# MAGIC **MinMaxScaler** (0-1):
# MAGIC ```python
# MAGIC from sklearn.preprocessing import MinMaxScaler
# MAGIC scaler = MinMaxScaler()
# MAGIC df[['precio']] = scaler.fit_transform(df[['precio']])
# MAGIC ```
# MAGIC
# MAGIC #### 💰 **Features de Interacción**
# MAGIC ```python
# MAGIC df['precio_por_cantidad'] = df['precio'] * df['cantidad']
# MAGIC df['margen_porcentaje'] = (df['precio'] - df['costo']) / df['precio'] * 100
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 4️⃣ Machine Learning: Fundamentos
# MAGIC
# MAGIC ### Tipos de Aprendizaje
# MAGIC
# MAGIC #### **Supervised Learning** (🎯 Con etiquetas)
# MAGIC
# MAGIC **Clasificación** (categorías discretas):
# MAGIC * Spam vs. No spam
# MAGIC * Cliente VIP vs. Regular
# MAGIC * Churn vs. Retención
# MAGIC
# MAGIC **Regresión** (valores continuos):
# MAGIC * Predecir precio
# MAGIC * Pronosticar ventas
# MAGIC * Estimar demanda
# MAGIC
# MAGIC #### **Unsupervised Learning** (🔍 Sin etiquetas)
# MAGIC
# MAGIC **Clustering**:
# MAGIC * Segmentación de clientes
# MAGIC * Detección de anomalías
# MAGIC * Agrupamiento de productos
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Workflow de ML
# MAGIC
# MAGIC ```
# MAGIC 1. Preparar Datos → 2. Split Train/Test → 3. Entrenar Modelo → 4. Evaluar → 5. Optimizar
# MAGIC ```
# MAGIC
# MAGIC #### 1. Preparar Datos
# MAGIC ```python
# MAGIC # Features (X) y Target (y)
# MAGIC X = df[['precio', 'cantidad', 'descuento']]
# MAGIC y = df['ventas']
# MAGIC ```
# MAGIC
# MAGIC #### 2. Split Train/Test
# MAGIC ```python
# MAGIC from sklearn.model_selection import train_test_split
# MAGIC X_train, X_test, y_train, y_test = train_test_split(
# MAGIC     X, y, test_size=0.2, random_state=42
# MAGIC )
# MAGIC ```
# MAGIC
# MAGIC #### 3. Entrenar Modelo
# MAGIC ```python
# MAGIC from sklearn.ensemble import RandomForestRegressor
# MAGIC
# MAGIC model = RandomForestRegressor(n_estimators=100, random_state=42)
# MAGIC model.fit(X_train, y_train)
# MAGIC ```
# MAGIC
# MAGIC #### 4. Evaluar
# MAGIC ```python
# MAGIC from sklearn.metrics import mean_absolute_error, r2_score
# MAGIC
# MAGIC y_pred = model.predict(X_test)
# MAGIC mae = mean_absolute_error(y_test, y_pred)
# MAGIC r2 = r2_score(y_test, y_pred)
# MAGIC
# MAGIC print(f'MAE: {mae:.2f}')
# MAGIC print(f'R²: {r2:.4f}')
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 5️⃣ Métricas de Evaluación
# MAGIC
# MAGIC ### Regresión
# MAGIC
# MAGIC | Métrica | Fórmula | Interpretación |
# MAGIC |---|---|---|
# MAGIC | **MAE** | Media de errores absolutos | Error promedio en unidades originales |
# MAGIC | **RMSE** | Raíz de errores cuadrados | Penaliza errores grandes |
# MAGIC | **R²** | Varianza explicada | 0-1, cercano a 1 es mejor |
# MAGIC
# MAGIC ### Clasificación
# MAGIC
# MAGIC | Métrica | Descripción |
# MAGIC |---|---|
# MAGIC | **Accuracy** | % de predicciones correctas |
# MAGIC | **Precision** | De los predichos positivos, cuántos son correctos |
# MAGIC | **Recall** | De los reales positivos, cuántos detecté |
# MAGIC | **F1-Score** | Media armónica de Precision y Recall |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🎓 Resumen Unidad 3
# MAGIC
# MAGIC ### Conceptos Clave
# MAGIC
# MAGIC 1. **Delta Lake**: ACID, time travel, performance
# MAGIC 2. **Agregaciones**: GROUP BY + Window functions
# MAGIC 3. **Feature Engineering**: Temporal, encoding, escalado, interacciones
# MAGIC 4. **ML Workflow**: Prep → Split → Train → Eval → Optimize
# MAGIC 5. **Métricas**: MAE, RMSE, R² (regresión) | Accuracy, F1 (clasificación)
# MAGIC
# MAGIC ### Próximos Pasos
# MAGIC
# MAGIC **TP05: Estructuración y Agregación**
# MAGIC * Crear tablas Delta particionadas
# MAGIC * Agregaciones por múltiples dimensiones
# MAGIC * Window functions para rankings
# MAGIC
# MAGIC **TP06: Feature Engineering y Modelado**
# MAGIC * Crear features de negocio
# MAGIC * Entrenar Random Forest
# MAGIC * Evaluar y optimizar modelo
# MAGIC
# MAGIC ✅ **¡Listo para modelar!**

# COMMAND ----------

