# Databricks notebook source
# DBTITLE 1,Encabezado
# MAGIC %md
# MAGIC # 📊 Unidad 1: Análisis de Datos en la Nube
# MAGIC ## Laboratorio (Herramientas) - Universidad del Aconcagua
# MAGIC ### Contenido Teórico
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🎯 Objetivos de Aprendizaje
# MAGIC
# MAGIC Al finalizar esta unidad teórica, serás capaz de:
# MAGIC
# MAGIC 1. ✅ Comprender los fundamentos del análisis de datos en entornos cloud
# MAGIC 2. ✅ Identificar las ventajas de plataformas como Databricks
# MAGIC 3. ✅ Conocer las herramientas fundamentales: Pandas, PySpark, SQL
# MAGIC 4. ✅ Entender el ciclo de vida de un proyecto de análisis de datos
# MAGIC 5. ✅ Aplicar mejores prácticas en exploración y manipulación de datos
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 📚 Contenido
# MAGIC
# MAGIC 1. Introducción al Análisis de Datos
# MAGIC 2. Computación en la Nube para Data Science
# MAGIC 3. Databricks como Plataforma Unificada
# MAGIC 4. Herramientas de Análisis: Pandas y PySpark
# MAGIC 5. SQL en el Análisis de Datos
# MAGIC 6. Ciclo de Vida de un Proyecto de Datos
# MAGIC 7. Mejores Prácticas
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### ⏱️ Duración Estimada: 2 horas

# COMMAND ----------

# DBTITLE 1,1. Introducción
# MAGIC %md
# MAGIC ## 1️⃣ Introducción al Análisis de Datos
# MAGIC
# MAGIC ### ¿Qué es el Análisis de Datos?
# MAGIC
# MAGIC El **análisis de datos** es el proceso de inspeccionar, limpiar, transformar y modelar datos con el objetivo de descubrir información útil, llegar a conclusiones y apoyar la toma de decisiones.
# MAGIC
# MAGIC ### Tipos de Análisis de Datos
# MAGIC
# MAGIC #### 📈 **Análisis Descriptivo** (¿Qué pasó?)
# MAGIC * Resume datos históricos
# MAGIC * Proporciona contexto sobre el pasado
# MAGIC * Ejemplos: reportes de ventas, dashboards de KPIs
# MAGIC * **Herramientas**: agregaciones, estadísticas descriptivas, visualizaciones
# MAGIC
# MAGIC #### 🔍 **Análisis Diagnóstico** (¿Por qué pasó?)
# MAGIC * Identifica causas y relaciones
# MAGIC * Profundiza en patrones y anomalías
# MAGIC * Ejemplos: análisis de caída en ventas, identificación de outliers
# MAGIC * **Herramientas**: correlaciones, drill-down, comparaciones
# MAGIC
# MAGIC #### 🔮 **Análisis Predictivo** (¿Qué pasará?)
# MAGIC * Utiliza datos históricos para predecir futuros
# MAGIC * Aplica modelos estadísticos y machine learning
# MAGIC * Ejemplos: pronóstico de demanda, predicción de churn
# MAGIC * **Herramientas**: regresión, series temporales, ML
# MAGIC
# MAGIC #### 💡 **Análisis Prescriptivo** (¿Qué debemos hacer?)
# MAGIC * Recomienda acciones basadas en datos
# MAGIC * Optimiza decisiones de negocio
# MAGIC * Ejemplos: optimización de inventario, pricing dinámico
# MAGIC * **Herramientas**: optimización, simulación, IA
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### El Valor del Análisis de Datos
# MAGIC
# MAGIC 🎯 **Para el Negocio:**
# MAGIC * Toma de decisiones basada en evidencia
# MAGIC * Identificación de oportunidades y riesgos
# MAGIC * Mejora de eficiencia operativa
# MAGIC * Ventaja competitiva
# MAGIC
# MAGIC 👥 **Para los Profesionales:**
# MAGIC * Habilidad altamente demandada
# MAGIC * Carrera en crecimiento
# MAGIC * Salarios competitivos
# MAGIC * Versatilidad (aplicable a todas las industrias)
# MAGIC
# MAGIC ---

# COMMAND ----------

# DBTITLE 1,2. Cloud Computing
# MAGIC %md
# MAGIC ## 2️⃣ Computación en la Nube para Data Science
# MAGIC
# MAGIC ### ¿Por qué Cloud?
# MAGIC
# MAGIC La **computación en la nube** ha revolucionado el análisis de datos al democratizar el acceso a recursos computacionales masivos.
# MAGIC
# MAGIC ### Ventajas del Cloud para Análisis de Datos
# MAGIC
# MAGIC #### ⚡ **Escalabilidad**
# MAGIC * **Elastic Compute**: Ajusta recursos según demanda
# MAGIC * **Sin límites físicos**: Procesa petabytes de datos
# MAGIC * **Scale up/down**: Paga solo por lo que usas
# MAGIC
# MAGIC ```
# MAGIC Trabajo local: 8GB RAM, 4 cores
# MAGIC            vs.
# MAGIC Cluster cloud: 1TB RAM, 256 cores
# MAGIC ```
# MAGIC
# MAGIC #### 💰 **Costo-Eficiencia**
# MAGIC * **CapEx → OpEx**: No inversión inicial en hardware
# MAGIC * **Pay-as-you-go**: Pago por uso
# MAGIC * **No mantenimiento**: Sin costos de infraestructura
# MAGIC
# MAGIC #### 🤝 **Colaboración**
# MAGIC * **Trabajo remoto**: Acceso desde cualquier lugar
# MAGIC * **Compartir recursos**: Equipos distribuidos
# MAGIC * **Control de versiones**: Git integrado
# MAGIC
# MAGIC #### 🔒 **Seguridad y Confiabilidad**
# MAGIC * **Backups automáticos**: Sin pérdida de datos
# MAGIC * **Alta disponibilidad**: 99.9% uptime
# MAGIC * **Certificaciones**: Cumplimiento regulatorio
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Modelos de Servicio Cloud
# MAGIC
# MAGIC #### **IaaS** (Infrastructure as a Service)
# MAGIC * Control total sobre VMs y redes
# MAGIC * Ejemplo: AWS EC2, Azure VMs
# MAGIC
# MAGIC #### **PaaS** (Platform as a Service)
# MAGIC * Plataforma managed para desarrollo
# MAGIC * Ejemplo: **Databricks**, Google App Engine
# MAGIC
# MAGIC #### **SaaS** (Software as a Service)
# MAGIC * Aplicaciones listas para usar
# MAGIC * Ejemplo: Salesforce, Google Workspace
# MAGIC
# MAGIC **📊 Para Data Science, PaaS es ideal**: abstrae infraestructura, permite enfocarse en análisis.
# MAGIC
# MAGIC ---

# COMMAND ----------

# DBTITLE 1,3. Databricks
# MAGIC %md
# MAGIC ## 3️⃣ Databricks como Plataforma Unificada
# MAGIC
# MAGIC ### ¿Qué es Databricks?
# MAGIC
# MAGIC **Databricks** es una plataforma de análisis de datos unificada basada en Apache Spark, diseñada para:
# MAGIC
# MAGIC * 📊 **Data Engineering**: Pipelines de datos escalables
# MAGIC * 🤖 **Machine Learning**: Entrenamiento y despliegue de modelos
# MAGIC * 📈 **Analytics**: Análisis y visualización de datos
# MAGIC * 🤝 **Colaboración**: Trabajo en equipo sobre datos
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Arquitectura de Databricks
# MAGIC
# MAGIC ```
# MAGIC ┌────────────────────────────────────┐
# MAGIC │  WORKSPACE (Interfaz de Usuario)        │
# MAGIC │  Notebooks | Dashboards | Jobs          │
# MAGIC ├────────────────────────────────────┤
# MAGIC │  UNITY CATALOG (Gobernanza)             │
# MAGIC │  Catálogos | Esquemas | Tablas | Permisos│
# MAGIC ├────────────────────────────────────┤
# MAGIC │  DELTA LAKE (Almacenamiento)            │
# MAGIC │  Tablas Delta | Transacciones ACID      │
# MAGIC ├────────────────────────────────────┤
# MAGIC │  APACHE SPARK (Motor de Cómputo)        │
# MAGIC │  Procesamiento Distribuido | Scala/PySpark│
# MAGIC └────────────────────────────────────┘
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Componentes Clave
# MAGIC
# MAGIC #### 📓 **Notebooks**
# MAGIC * Entorno interactivo para análisis
# MAGIC * Soporta Python, SQL, Scala, R
# MAGIC * Colaboración en tiempo real
# MAGIC * Control de versiones integrado
# MAGIC
# MAGIC #### 📦 **Delta Lake**
# MAGIC * Formato de almacenamiento optimizado
# MAGIC * **ACID transactions**: Consistencia garantizada
# MAGIC * **Time Travel**: Viaja en el tiempo de tus datos
# MAGIC * **Schema Evolution**: Esquema flexible
# MAGIC
# MAGIC #### 📋 **Unity Catalog**
# MAGIC * Gobierno de datos unificado
# MAGIC * Control de acceso granular
# MAGIC * Lineage de datos
# MAGIC * Auditoría completa
# MAGIC
# MAGIC #### ⚡ **Clusters**
# MAGIC * **Serverless**: Auto-scaling, sin configuración
# MAGIC * **All-purpose**: Desarrollo interactivo
# MAGIC * **Job clusters**: Ejecución automatizada
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Ventajas de Databricks para Análisis
# MAGIC
# MAGIC ✅ **Unificación**: Todo en un solo lugar (EDA, ML, BI)  
# MAGIC ✅ **Performance**: Spark optimizado  
# MAGIC ✅ **Colaboración**: Notebooks compartidos  
# MAGIC ✅ **Escalabilidad**: De MB a PB sin cambios de código  
# MAGIC ✅ **Integración**: Conectores a todas las fuentes de datos  
# MAGIC ✅ **Seguridad**: Gobernanza enterprise-grade  
# MAGIC
# MAGIC ---

# COMMAND ----------

# DBTITLE 1,4. Pandas vs PySpark
# MAGIC %md
# MAGIC ## 4️⃣ Herramientas de Análisis: Pandas vs PySpark
# MAGIC
# MAGIC ### 🐼 Pandas: Análisis en Memoria
# MAGIC
# MAGIC **Pandas** es la biblioteca de Python más popular para manipulación y análisis de datos.
# MAGIC
# MAGIC #### Características:
# MAGIC * 📊 **DataFrame**: Estructura tabular en memoria
# MAGIC * 🚀 **Rápido**: Para datasets pequeños-medianos (< 10GB)
# MAGIC * 💻 **Single-node**: Ejecuta en una sola máquina
# MAGIC * 🔧 **Flexible**: API rica y expresiva
# MAGIC
# MAGIC #### Cuándo usar Pandas:
# MAGIC ✅ Datasets < 10GB  
# MAGIC ✅ Prototipado rápido  
# MAGIC ✅ Análisis exploratorio  
# MAGIC ✅ Visualizaciones con matplotlib/seaborn  
# MAGIC
# MAGIC #### Ejemplo:
# MAGIC ```python
# MAGIC import pandas as pd
# MAGIC
# MAGIC # Cargar datos
# MAGIC df = pd.read_csv('ventas.csv')
# MAGIC
# MAGIC # Análisis
# MAGIC top_productos = df.groupby('producto')['total'].sum().sort_values(ascending=False).head(10)
# MAGIC
# MAGIC # Filtrado
# MAGIC ventas_2025 = df[df['fecha'].dt.year == 2025]
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### ⚡ PySpark: Análisis Distribuido
# MAGIC
# MAGIC **PySpark** es la API de Python para Apache Spark, diseñada para big data.
# MAGIC
# MAGIC #### Características:
# MAGIC * 🌍 **Distribuido**: Procesa datos en múltiples nodos
# MAGIC * 📊 **Escalable**: Maneja petabytes de datos
# MAGIC * ⏱️ **Lazy Evaluation**: Optimiza ejecución
# MAGIC * 🔥 **In-Memory**: Caché distribuido
# MAGIC
# MAGIC #### Cuándo usar PySpark:
# MAGIC ✅ Datasets > 10GB  
# MAGIC ✅ Procesamiento batch a gran escala  
# MAGIC ✅ ETL pipelines  
# MAGIC ✅ Data engineering  
# MAGIC
# MAGIC #### Ejemplo:
# MAGIC ```python
# MAGIC from pyspark.sql import functions as F
# MAGIC
# MAGIC # Cargar datos
# MAGIC df_spark = spark.read.csv('/data/ventas.csv', header=True, inferSchema=True)
# MAGIC
# MAGIC # Análisis
# MAGIC top_productos = df_spark.groupBy('producto') \
# MAGIC     .agg(F.sum('total').alias('total_ventas')) \
# MAGIC     .orderBy(F.desc('total_ventas')) \
# MAGIC     .limit(10)
# MAGIC
# MAGIC # Filtrado
# MAGIC ventas_2025 = df_spark.filter(F.year('fecha') == 2025)
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 🎯 Comparación
# MAGIC
# MAGIC | Característica | Pandas 🐼 | PySpark ⚡ |
# MAGIC |---|---|---|
# MAGIC | **Tamaño de datos** | < 10GB | > 10GB |
# MAGIC | **Arquitectura** | Single-node | Distribuido |
# MAGIC | **Performance** | Rápido en pequeño | Escalable en grande |
# MAGIC | **API** | Muy flexible | Más verbosa |
# MAGIC | **Curva de aprendizaje** | Baja | Media |
# MAGIC | **Costo** | Gratis | Requiere cluster |
# MAGIC | **Uso típico** | EDA, prototipado | ETL, producción |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 🤝 Enfoque Híbrido
# MAGIC
# MAGIC En Databricks, puedes combinar ambos:
# MAGIC
# MAGIC 1. **Pandas**: Exploración inicial, visualizaciones
# MAGIC 2. **PySpark**: Procesamiento pesado, transformaciones
# MAGIC 3. **Pandas UDF**: Lo mejor de ambos mundos
# MAGIC
# MAGIC ```python
# MAGIC # PySpark para procesamiento
# MAGIC df_spark = spark.read.table('ventas')
# MAGIC df_agregado = df_spark.groupBy('mes').agg(F.sum('total'))
# MAGIC
# MAGIC # Convertir a Pandas para visualización
# MAGIC df_pandas = df_agregado.toPandas()
# MAGIC df_pandas.plot(kind='bar')
# MAGIC ```
# MAGIC
# MAGIC **Regla práctica**: Usa Pandas cuando puedas, PySpark cuando debas.
# MAGIC
# MAGIC ---

# COMMAND ----------

# DBTITLE 1,5. SQL
# MAGIC %md
# MAGIC ## 5️⃣ SQL en el Análisis de Datos
# MAGIC
# MAGIC ### ¿Por qué SQL?
# MAGIC
# MAGIC **SQL** (Structured Query Language) es el lenguaje universal para trabajar con datos estructurados.
# MAGIC
# MAGIC ### Ventajas de SQL para Análisis
# MAGIC
# MAGIC ✅ **Universal**: Estándar en toda la industria  
# MAGIC ✅ **Declarativo**: Dices QUÉ quieres, no CÓMO  
# MAGIC ✅ **Optimizado**: Motores de consulta altamente eficientes  
# MAGIC ✅ **Accesible**: Fácil de aprender y leer  
# MAGIC ✅ **Integrado**: Funciona con Pandas, Spark, BI tools  
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### SQL en Databricks
# MAGIC
# MAGIC Databricks soporta **Spark SQL**, una implementación de SQL optimizada para big data.
# MAGIC
# MAGIC #### Características:
# MAGIC * **ANSI SQL compliant**: Estándar SQL
# MAGIC * **Funciones avanzadas**: Window functions, CTEs, pivots
# MAGIC * **Integración**: Mezcla SQL con Python/Scala
# MAGIC * **Performance**: Catalyst optimizer
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Operaciones Clave en SQL
# MAGIC
# MAGIC #### 📊 **Agregaciones**
# MAGIC ```sql
# MAGIC SELECT 
# MAGIC     categoria,
# MAGIC     COUNT(*) as num_productos,
# MAGIC     SUM(precio) as total,
# MAGIC     AVG(precio) as precio_promedio,
# MAGIC     MAX(precio) as precio_max
# MAGIC FROM productos
# MAGIC GROUP BY categoria
# MAGIC ORDER BY total DESC;
# MAGIC ```
# MAGIC
# MAGIC #### 🔗 **Joins**
# MAGIC ```sql
# MAGIC SELECT 
# MAGIC     v.fecha,
# MAGIC     c.nombre as cliente,
# MAGIC     p.nombre as producto,
# MAGIC     d.cantidad,
# MAGIC     d.subtotal
# MAGIC FROM ventas v
# MAGIC INNER JOIN clientes c ON v.cliente_id = c.cliente_id
# MAGIC INNER JOIN detalles_ventas d ON v.venta_id = d.venta_id
# MAGIC INNER JOIN productos p ON d.producto_id = p.producto_id
# MAGIC WHERE v.fecha >= '2025-01-01';
# MAGIC ```
# MAGIC
# MAGIC #### 📊 **Window Functions**
# MAGIC ```sql
# MAGIC SELECT 
# MAGIC     fecha,
# MAGIC     producto,
# MAGIC     ventas,
# MAGIC     SUM(ventas) OVER (PARTITION BY producto ORDER BY fecha) as ventas_acumuladas,
# MAGIC     ROW_NUMBER() OVER (PARTITION BY DATE_TRUNC('month', fecha) ORDER BY ventas DESC) as ranking_mes
# MAGIC FROM ventas_diarias;
# MAGIC ```
# MAGIC
# MAGIC #### 🔍 **CTEs (Common Table Expressions)**
# MAGIC ```sql
# MAGIC WITH ventas_mensuales AS (
# MAGIC     SELECT 
# MAGIC         DATE_TRUNC('month', fecha) as mes,
# MAGIC         SUM(total) as ventas
# MAGIC     FROM ventas
# MAGIC     GROUP BY DATE_TRUNC('month', fecha)
# MAGIC ),
# MAGIC crecimiento AS (
# MAGIC     SELECT
# MAGIC         mes,
# MAGIC         ventas,
# MAGIC         LAG(ventas) OVER (ORDER BY mes) as ventas_mes_anterior,
# MAGIC         (ventas - LAG(ventas) OVER (ORDER BY mes)) / LAG(ventas) OVER (ORDER BY mes) * 100 as crecimiento_porcentaje
# MAGIC     FROM ventas_mensuales
# MAGIC )
# MAGIC SELECT * FROM crecimiento;
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### SQL + Python: Lo Mejor de Ambos Mundos
# MAGIC
# MAGIC En Databricks notebooks, puedes mezclar SQL y Python:
# MAGIC
# MAGIC ```python
# MAGIC # Python cell
# MAGIC import pandas as pd
# MAGIC
# MAGIC # Ejecutar SQL desde Python
# MAGIC df = spark.sql("""
# MAGIC     SELECT categoria, SUM(total) as ventas
# MAGIC     FROM ventas
# MAGIC     GROUP BY categoria
# MAGIC """)
# MAGIC
# MAGIC # Convertir a Pandas
# MAGIC df_pandas = df.toPandas()
# MAGIC
# MAGIC # Visualizar
# MAGIC df_pandas.plot(kind='bar')
# MAGIC ```
# MAGIC
# MAGIC ```sql
# MAGIC -- SQL cell
# MAGIC %sql
# MAGIC SELECT * FROM ventas
# MAGIC WHERE fecha > current_date() - INTERVAL 30 DAYS
# MAGIC ORDER BY total DESC
# MAGIC LIMIT 100
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Mejores Prácticas en SQL
# MAGIC
# MAGIC ✅ **Usa nombres descriptivos** para columnas y tablas  
# MAGIC ✅ **Indenta tu código** para legibilidad  
# MAGIC ✅ **Filtra temprano** (WHERE antes de JOIN)  
# MAGIC ✅ **Usa CTEs** para queries complejas  
# MAGIC ✅ **Evita SELECT \*** en producción  
# MAGIC ✅ **Aprovecha particiones** en tablas grandes  
# MAGIC
# MAGIC ---

# COMMAND ----------

# DBTITLE 1,6. Ciclo de Vida
# MAGIC %md
# MAGIC ## 6️⃣ Ciclo de Vida de un Proyecto de Análisis de Datos
# MAGIC
# MAGIC ### El Proceso Completo
# MAGIC
# MAGIC ```
# MAGIC 1. ENTENDER → 2. OBTENER → 3. LIMPIAR → 4. EXPLORAR → 5. ANALIZAR → 6. COMUNICAR
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 1️⃣ Entender el Problema
# MAGIC
# MAGIC **Objetivo**: Definir claramente qué preguntas queremos responder.
# MAGIC
# MAGIC 🎯 **Preguntas clave:**
# MAGIC * ¿Cuál es el objetivo de negocio?
# MAGIC * ¿Qué decisiones se tomarán con este análisis?
# MAGIC * ¿Quién es la audiencia?
# MAGIC * ¿Qué métricas son importantes?
# MAGIC
# MAGIC 📝 **Entregable**: Documento de alcance del proyecto
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 2️⃣ Obtener los Datos
# MAGIC
# MAGIC **Objetivo**: Identificar y acceder a las fuentes de datos relevantes.
# MAGIC
# MAGIC 📊 **Fuentes comunes:**
# MAGIC * Bases de datos (SQL, NoSQL)
# MAGIC * APIs (REST, GraphQL)
# MAGIC * Archivos (CSV, Excel, Parquet)
# MAGIC * Data Lakes / Data Warehouses
# MAGIC * Streaming (Kafka, Event Hubs)
# MAGIC
# MAGIC 🔧 **Herramientas en Databricks:**
# MAGIC ```python
# MAGIC # Leer desde Unity Catalog
# MAGIC df = spark.read.table('catalogo.esquema.tabla')
# MAGIC
# MAGIC # Leer CSV
# MAGIC df = pd.read_csv('/dbfs/mnt/datos/ventas.csv')
# MAGIC
# MAGIC # Leer desde API
# MAGIC import requests
# MAGIC response = requests.get('https://api.ejemplo.com/datos')
# MAGIC df = pd.DataFrame(response.json())
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 3️⃣ Limpiar los Datos
# MAGIC
# MAGIC **Objetivo**: Preparar datos de calidad para análisis.
# MAGIC
# MAGIC 🧹 **Tareas comunes:**
# MAGIC
# MAGIC #### **Valores Nulos**
# MAGIC ```python
# MAGIC # Detectar
# MAGIC df.isnull().sum()
# MAGIC
# MAGIC # Eliminar
# MAGIC df_clean = df.dropna()
# MAGIC
# MAGIC # Imputar
# MAGIC df['precio'] = df['precio'].fillna(df['precio'].median())
# MAGIC ```
# MAGIC
# MAGIC #### **Duplicados**
# MAGIC ```python
# MAGIC # Detectar
# MAGIC df.duplicated().sum()
# MAGIC
# MAGIC # Eliminar
# MAGIC df_clean = df.drop_duplicates()
# MAGIC ```
# MAGIC
# MAGIC #### **Tipos de Datos**
# MAGIC ```python
# MAGIC # Convertir
# MAGIC df['fecha'] = pd.to_datetime(df['fecha'])
# MAGIC df['precio'] = df['precio'].astype(float)
# MAGIC ```
# MAGIC
# MAGIC #### **Outliers**
# MAGIC ```python
# MAGIC # Detectar con IQR
# MAGIC Q1 = df['precio'].quantile(0.25)
# MAGIC Q3 = df['precio'].quantile(0.75)
# MAGIC IQR = Q3 - Q1
# MAGIC outliers = (df['precio'] < Q1 - 1.5*IQR) | (df['precio'] > Q3 + 1.5*IQR)
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 4️⃣ Explorar los Datos (EDA)
# MAGIC
# MAGIC **Objetivo**: Entender la estructura, patrones y características de los datos.
# MAGIC
# MAGIC 🔍 **Estadísticas Descriptivas**
# MAGIC ```python
# MAGIC # Resumen numérico
# MAGIC df.describe()
# MAGIC
# MAGIC # Resumen categórico
# MAGIC df['categoria'].value_counts()
# MAGIC
# MAGIC # Correlaciones
# MAGIC df.corr()
# MAGIC ```
# MAGIC
# MAGIC 📊 **Visualizaciones Exploratorias**
# MAGIC * Histogramas: Distribución de variables
# MAGIC * Scatter plots: Relaciones entre variables
# MAGIC * Box plots: Detección de outliers
# MAGIC * Time series: Tendencias temporales
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 5️⃣ Analizar
# MAGIC
# MAGIC **Objetivo**: Responder preguntas específicas del negocio.
# MAGIC
# MAGIC 📊 **Técnicas:**
# MAGIC * **Agregaciones**: Resumir datos por grupos
# MAGIC * **Segmentación**: Dividir en cohortes
# MAGIC * **Comparaciones**: Antes/después, A/B testing
# MAGIC * **Tendencias**: Series temporales, forecasting
# MAGIC * **Modelado**: ML para predicciones
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 6️⃣ Comunicar Resultados
# MAGIC
# MAGIC **Objetivo**: Presentar findings de forma clara y accionable.
# MAGIC
# MAGIC 📈 **Herramientas:**
# MAGIC * **Dashboards**: Visualizaciones interactivas
# MAGIC * **Reportes**: Notebooks con narrativa
# MAGIC * **Presentaciones**: Slides con insights clave
# MAGIC
# MAGIC 🎯 **Mejores prácticas:**
# MAGIC * Cuenta una historia con los datos
# MAGIC * Visualiza, no solo tablas
# MAGIC * Enfoca en insights, no en técnica
# MAGIC * Proporciona recomendaciones accionables
# MAGIC * Cuantifica impacto (ROI, ahorros, mejoras)
# MAGIC
# MAGIC ---

# COMMAND ----------

# DBTITLE 1,7. Mejores Prácticas
# MAGIC %md
# MAGIC ## 7️⃣ Mejores Prácticas en Análisis de Datos
# MAGIC
# MAGIC ### 📝 Documentación
# MAGIC
# MAGIC ✅ **Comenta tu código**
# MAGIC ```python
# MAGIC # Calcular total de ventas por categoría
# MAGIC # Filtramos solo ventas de 2025 para análisis actual
# MAGIC ventas_2025 = df[df['fecha'].dt.year == 2025]
# MAGIC total_por_categoria = ventas_2025.groupby('categoria')['total'].sum()
# MAGIC ```
# MAGIC
# MAGIC ✅ **Usa markdown cells** en notebooks para explicar contexto
# MAGIC
# MAGIC ✅ **Nombra variables descriptivamente**
# MAGIC ```python
# MAGIC # ❌ MAL
# MAGIC x = df.groupby('a')['b'].sum()
# MAGIC
# MAGIC # ✅ BIEN
# MAGIC total_ventas_por_categoria = df.groupby('categoria')['total'].sum()
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 📑 Organización de Notebooks
# MAGIC
# MAGIC #### Estructura recomendada:
# MAGIC
# MAGIC ```
# MAGIC 1. Título y Descripción
# MAGIC 2. Imports y Configuración
# MAGIC 3. Carga de Datos
# MAGIC 4. Exploración Inicial
# MAGIC 5. Limpieza y Transformación
# MAGIC 6. Análisis Principal
# MAGIC 7. Visualizaciones
# MAGIC 8. Conclusiones
# MAGIC 9. Siguientes Pasos
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### ⚡ Performance
# MAGIC
# MAGIC ✅ **Filtra temprano** para reducir datos
# MAGIC ```python
# MAGIC # ✅ BIEN: Filtrar primero
# MAGIC df_filtrado = df[df['fecha'] >= '2025-01-01']
# MAGIC resultado = df_filtrado.groupby('categoria').sum()
# MAGIC
# MAGIC # ❌ MAL: Procesar todo y luego filtrar
# MAGIC resultado_completo = df.groupby('categoria').sum()
# MAGIC resultado = resultado_completo[resultado_completo.index > '2025-01-01']
# MAGIC ```
# MAGIC
# MAGIC ✅ **Usa columnar formats** (Parquet, Delta)
# MAGIC ```python
# MAGIC # Más rápido
# MAGIC df = spark.read.parquet('/data/ventas.parquet')
# MAGIC
# MAGIC # vs CSV
# MAGIC df = spark.read.csv('/data/ventas.csv')  # Más lento
# MAGIC ```
# MAGIC
# MAGIC ✅ **Cachea datos** que uses múltiples veces
# MAGIC ```python
# MAGIC df_spark = spark.read.table('ventas')
# MAGIC df_spark.cache()  # Mantener en memoria
# MAGIC
# MAGIC # Múltiples operaciones sin re-leer
# MAGIC resultado1 = df_spark.filter(...)
# MAGIC resultado2 = df_spark.groupBy(...)
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 🔒 Calidad de Datos
# MAGIC
# MAGIC ✅ **Valida siempre**
# MAGIC ```python
# MAGIC # Verificar nulls
# MAGIC assert df['cliente_id'].isnull().sum() == 0, "Hay clientes nulos!"
# MAGIC
# MAGIC # Verificar rangos
# MAGIC assert (df['precio'] > 0).all(), "Hay precios negativos!"
# MAGIC
# MAGIC # Verificar duplicados
# MAGIC assert df.duplicated().sum() == 0, "Hay registros duplicados!"
# MAGIC ```
# MAGIC
# MAGIC ✅ **Documenta asunciones**
# MAGIC ```python
# MAGIC # ASUNCIÓN: Todas las ventas tienen fecha válida
# MAGIC # Si hay nulls, los imputamos con fecha de sistema
# MAGIC df['fecha'] = df['fecha'].fillna(pd.Timestamp.now())
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 🔄 Reproducibilidad
# MAGIC
# MAGIC ✅ **Control de versiones** (Git)
# MAGIC ```bash
# MAGIC git add notebook.ipynb
# MAGIC git commit -m "Agregar análisis de ventas Q1"
# MAGIC git push
# MAGIC ```
# MAGIC
# MAGIC ✅ **Seeds para aleatoriedad**
# MAGIC ```python
# MAGIC import random
# MAGIC import numpy as np
# MAGIC
# MAGIC random.seed(42)
# MAGIC np.random.seed(42)
# MAGIC ```
# MAGIC
# MAGIC ✅ **Documenta el entorno**
# MAGIC ```python
# MAGIC # Al inicio del notebook
# MAGIC import sys
# MAGIC import pandas as pd
# MAGIC import numpy as np
# MAGIC
# MAGIC print(f"Python version: {sys.version}")
# MAGIC print(f"Pandas version: {pd.__version__}")
# MAGIC print(f"Numpy version: {np.__version__}")
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### ⚖️ Ética en Datos
# MAGIC
# MAGIC ⚠️ **Consideraciones importantes:**
# MAGIC
# MAGIC * **Privacidad**: Anonimiza datos personales
# MAGIC * **Bias**: Verifica sesgos en datos y modelos
# MAGIC * **Transparencia**: Documenta decisiones y limitaciones
# MAGIC * **Seguridad**: Protege datos sensibles
# MAGIC * **Consentimiento**: Respeta permisos de uso de datos
# MAGIC
# MAGIC ---

# COMMAND ----------

# DBTITLE 1,Resumen
# MAGIC %md
# MAGIC ## 🎓 Resumen de la Unidad 1
# MAGIC
# MAGIC ### 🎯 Conceptos Clave
# MAGIC
# MAGIC 1. **Análisis de Datos**: Proceso de transformar datos en insights accionables
# MAGIC 2. **Cloud Computing**: Escalabilidad, costo-eficiencia y colaboración
# MAGIC 3. **Databricks**: Plataforma unificada para datos y ML
# MAGIC 4. **Pandas vs PySpark**: Herramientas complementarias según escala
# MAGIC 5. **SQL**: Lenguaje universal para datos estructurados
# MAGIC 6. **Ciclo de Vida**: Entender → Obtener → Limpiar → Explorar → Analizar → Comunicar
# MAGIC 7. **Mejores Prácticas**: Documentación, performance, calidad, reproducibilidad
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 📚 Recursos Adicionales
# MAGIC
# MAGIC #### Documentación Oficial:
# MAGIC * [Databricks Documentation](https://docs.databricks.com/)
# MAGIC * [Pandas Documentation](https://pandas.pydata.org/docs/)
# MAGIC * [PySpark Documentation](https://spark.apache.org/docs/latest/api/python/)
# MAGIC
# MAGIC #### Libros Recomendados:
# MAGIC * "Python for Data Analysis" - Wes McKinney
# MAGIC * "Learning Spark" - Jules S. Damji et al.
# MAGIC * "SQL for Data Analysis" - Cathy Tanimura
# MAGIC
# MAGIC #### Cursos Online:
# MAGIC * Databricks Academy (cursos gratuitos)
# MAGIC * DataCamp: Data Analysis with Python
# MAGIC * Coursera: Applied Data Science with Python
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### ➡️ Próximos Pasos
# MAGIC
# MAGIC Ahora que tienes los fundamentos teóricos, es momento de aplicarlos:
# MAGIC
# MAGIC 1. **TP01**: Configuración Cloud y Almacenamiento
# MAGIC    - Configurar Databricks workspace
# MAGIC    - Cargar datos desde CSV
# MAGIC    - Primeras exploraciones
# MAGIC
# MAGIC 2. **TP02**: Manipulación Programática
# MAGIC    - Transformaciones con Pandas
# MAGIC    - Joins y agregaciones
# MAGIC    - Integración con Spark SQL
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## ✅ ¡Listo para comenzar con los prácticos!
# MAGIC
# MAGIC **La teoría es importante, pero la práctica te hace experto.**
# MAGIC
# MAGIC 🚀 **Adelante con TP01 y TP02**

# COMMAND ----------

