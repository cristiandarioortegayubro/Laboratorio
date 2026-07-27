# Databricks notebook source
# DBTITLE 1,Contenido Completo U4
# MAGIC %md
# MAGIC # 🚀 Unidad 4: Proyectos Integradores
# MAGIC ## Laboratorio (Herramientas) - Universidad del Aconcagua
# MAGIC ### Contenido Teórico
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🎯 Objetivos de Aprendizaje
# MAGIC
# MAGIC 1. ✅ Integrar todos los conceptos del curso
# MAGIC 2. ✅ Diseñar pipelines end-to-end
# MAGIC 3. ✅ Aplicar metodologías de análisis
# MAGIC 4. ✅ Comunicar resultados efectivamente
# MAGIC 5. ✅ Resolver problemas reales de negocio
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 1️⃣ Pipelines de Datos End-to-End
# MAGIC
# MAGIC ### ¿Qué es un Pipeline de Datos?
# MAGIC
# MAGIC Un **pipeline de datos** es un proceso automatizado que:
# MAGIC 1. **Extrae** datos de fuentes
# MAGIC 2. **Transforma** según reglas de negocio
# MAGIC 3. **Carga** en destino para consumo
# MAGIC
# MAGIC ### Componentes de un Pipeline
# MAGIC
# MAGIC ```
# MAGIC 📊 INGESTA → 🧹 LIMPIEZA → 🔧 TRANSFORMACIÓN → 💾 PERSISTENCIA → 📈 ANÁLISIS
# MAGIC ```
# MAGIC
# MAGIC #### **1. Ingesta**
# MAGIC * Leer de múltiples fuentes (CSV, DB, APIs)
# MAGIC * Validar estructura y tipos
# MAGIC * Verificar integridad
# MAGIC
# MAGIC ```python
# MAGIC # Cargar datasets
# MAGIC df_ventas = pd.read_csv('ventas.csv', parse_dates=['fecha'])
# MAGIC df_productos = spark.read.table('catalogo.productos')
# MAGIC df_api = requests.get('https://api.ejemplo.com/data').json()
# MAGIC ```
# MAGIC
# MAGIC #### **2. Limpieza**
# MAGIC * Eliminar duplicados
# MAGIC * Manejar valores nulos
# MAGIC * Corregir tipos de datos
# MAGIC * Validar rangos
# MAGIC
# MAGIC ```python
# MAGIC # Limpieza
# MAGIC df = df.drop_duplicates()
# MAGIC df['precio'] = df['precio'].fillna(df['precio'].median())
# MAGIC df = df[(df['precio'] > 0) & (df['cantidad'] > 0)]
# MAGIC ```
# MAGIC
# MAGIC #### **3. Transformación**
# MAGIC * Joins entre datasets
# MAGIC * Agregaciones
# MAGIC * Feature engineering
# MAGIC * Cálculos de negocio
# MAGIC
# MAGIC ```python
# MAGIC # Consolidar
# MAGIC df_maestro = df_ventas \
# MAGIC     .merge(df_productos, on='producto_id') \
# MAGIC     .merge(df_clientes, on='cliente_id')
# MAGIC
# MAGIC # Features calculadas
# MAGIC df_maestro['margen'] = df_maestro['precio'] - df_maestro['costo']
# MAGIC df_maestro['rentabilidad_%'] = df_maestro['margen'] / df_maestro['precio'] * 100
# MAGIC ```
# MAGIC
# MAGIC #### **4. Persistencia**
# MAGIC * Guardar en Delta Lake
# MAGIC * Particionamiento estratégico
# MAGIC * Optimización
# MAGIC
# MAGIC ```python
# MAGIC # Guardar como tabla Delta
# MAGIC df_spark.write.format('delta') \
# MAGIC     .mode('overwrite') \
# MAGIC     .partitionBy('anio', 'mes') \
# MAGIC     .saveAsTable('warehouse.ventas_consolidadas')
# MAGIC ```
# MAGIC
# MAGIC #### **5. Análisis**
# MAGIC * Exploración y visualización
# MAGIC * Modelado predictivo
# MAGIC * Generación de insights
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 2️⃣ Metodologías de Análisis Avanzado
# MAGIC
# MAGIC ### RFM: Segmentación de Clientes
# MAGIC
# MAGIC **RFM** analiza comportamiento de clientes en 3 dimensiones:
# MAGIC
# MAGIC * **R (Recency)**: ¿Cuán recientemente compró?
# MAGIC * **F (Frequency)**: ¿Qué tan seguido compra?
# MAGIC * **M (Monetary)**: ¿Cuánto gasta?
# MAGIC
# MAGIC #### Proceso:
# MAGIC
# MAGIC ```python
# MAGIC # Calcular métricas RFM
# MAGIC fecha_ref = df['fecha'].max() + pd.Timedelta(days=1)
# MAGIC
# MAGIC rfm = df.groupby('cliente_id').agg({
# MAGIC     'fecha': lambda x: (fecha_ref - x.max()).days,  # Recency
# MAGIC     'venta_id': 'nunique',  # Frequency
# MAGIC     'total': 'sum'  # Monetary
# MAGIC })
# MAGIC
# MAGIC rfm.columns = ['recency', 'frequency', 'monetary']
# MAGIC
# MAGIC # Scoring 1-5 (quintiles)
# MAGIC rfm['R_score'] = pd.qcut(rfm['recency'], 5, labels=[5,4,3,2,1])
# MAGIC rfm['F_score'] = pd.qcut(rfm['frequency'], 5, labels=[1,2,3,4,5])
# MAGIC rfm['M_score'] = pd.qcut(rfm['monetary'], 5, labels=[1,2,3,4,5])
# MAGIC
# MAGIC # Segmentar
# MAGIC def segmentar(row):
# MAGIC     if row['R_score'] >= 4 and row['F_score'] >= 4:
# MAGIC         return 'Campeones'
# MAGIC     elif row['R_score'] <= 2:
# MAGIC         return 'En Riesgo'
# MAGIC     # ... más reglas
# MAGIC
# MAGIC rfm['segmento'] = rfm.apply(segmentar, axis=1)
# MAGIC ```
# MAGIC
# MAGIC #### Segmentos Típicos:
# MAGIC
# MAGIC | Segmento | R | F | M | Acción |
# MAGIC |---|---|---|---|---|
# MAGIC | 🏆 Campeones | Alto | Alto | Alto | Retener, programa VIP |
# MAGIC | 💖 Leales | Medio | Alto | Alto | Upsell, cross-sell |
# MAGIC | 🌱 Nuevos | Alto | Bajo | Medio | Nutrir, activar |
# MAGIC | ⚠️ En Riesgo | Bajo | Alto | Alto | Reactivar urgente |
# MAGIC | ❌ Perdidos | Bajo | Bajo | Bajo | Win-back o abandonar |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### BCG Matrix: Análisis de Portafolio
# MAGIC
# MAGIC **Matriz BCG** clasifica productos según:
# MAGIC * **Eje X**: Participación de mercado / Volumen de ventas
# MAGIC * **Eje Y**: Crecimiento / Rentabilidad
# MAGIC
# MAGIC #### Cuadrantes:
# MAGIC
# MAGIC ```
# MAGIC       Alto Margen
# MAGIC            |
# MAGIC     ❓      |     ⭐
# MAGIC  Interrogante| Estrella
# MAGIC            |
# MAGIC -----------|-----------> Alto Volumen
# MAGIC            |
# MAGIC     🐶      |     🐄
# MAGIC    Perro    |Vaca Lechera
# MAGIC            |
# MAGIC       Bajo Margen
# MAGIC ```
# MAGIC
# MAGIC * **⭐ Estrella**: Alto volumen + Alto margen → Invertir y promover
# MAGIC * **🐄 Vaca Lechera**: Alto volumen + Bajo margen → Optimizar costos
# MAGIC * **❓ Interrogante**: Bajo volumen + Alto margen → Testear marketing
# MAGIC * **🐶 Perro**: Bajo volumen + Bajo margen → Descontinuar
# MAGIC
# MAGIC ```python
# MAGIC # Clasificar productos
# MAGIC mediana_volumen = df['facturacion'].median()
# MAGIC mediana_margen = df['margen_%'].median()
# MAGIC
# MAGIC def clasificar(row):
# MAGIC     if row['facturacion'] >= mediana_volumen:
# MAGIC         if row['margen_%'] >= mediana_margen:
# MAGIC             return 'Estrella'
# MAGIC         else:
# MAGIC             return 'Vaca Lechera'
# MAGIC     else:
# MAGIC         if row['margen_%'] >= mediana_margen:
# MAGIC             return 'Interrogante'
# MAGIC         else:
# MAGIC             return 'Perro'
# MAGIC
# MAGIC df['clasificacion'] = df.apply(clasificar, axis=1)
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 3️⃣ Mejores Prácticas en Proyectos
# MAGIC
# MAGIC ### Estructura de Proyecto
# MAGIC
# MAGIC ```
# MAGIC proyecto/
# MAGIC ├── 1_ingesta.ipynb           # Carga inicial
# MAGIC ├── 2_limpieza.ipynb          # Limpieza y validación
# MAGIC ├── 3_transformacion.ipynb    # Feature engineering
# MAGIC ├── 4_analisis.ipynb          # EDA y modelado
# MAGIC ├── 5_visualizacion.ipynb     # Dashboards
# MAGIC ├── README.md                 # Documentación
# MAGIC └── requirements.txt          # Dependencias
# MAGIC ```
# MAGIC
# MAGIC ### Documentación Efectiva
# MAGIC
# MAGIC #### En cada notebook:
# MAGIC
# MAGIC 1. **Título y Objetivo**
# MAGIC ```markdown
# MAGIC # Análisis de Rentabilidad - Q1 2025
# MAGIC
# MAGIC ## Objetivo
# MAGIC Identificar oportunidades de mejora en margen por producto y sucursal.
# MAGIC
# MAGIC ## Audiencia
# MAGIC Gerencia Comercial
# MAGIC ```
# MAGIC
# MAGIC 2. **Executive Summary** (al final)
# MAGIC ```markdown
# MAGIC ## 📊 Resumen Ejecutivo
# MAGIC
# MAGIC ### Hallazgos Clave
# MAGIC 1. Margen promedio: 32% (target: 35%)
# MAGIC 2. Top 10 productos generan 60% de ganancia
# MAGIC 3. Sucursal Centro tiene mejor margen (+8pp vs. promedio)
# MAGIC
# MAGIC ### Recomendaciones
# MAGIC 1. Descontinuar 15 productos de bajo margen (<15%)
# MAGIC 2. Aumentar precios 3-5% en productos inelásticos
# MAGIC 3. Replicar mix de Centro en otras sucursales
# MAGIC
# MAGIC ### Impacto Estimado
# MAGIC +$250K en ganancia anual (+12%)
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 4️⃣ Comunicación de Resultados
# MAGIC
# MAGIC ### Dashboard Ejecutivo
# MAGIC
# MAGIC #### Estructura recomendada:
# MAGIC
# MAGIC **Sección 1: KPIs** (área superior)
# MAGIC * Métricas principales con indicadores de cambio
# MAGIC * Comparación vs. periodo anterior / meta
# MAGIC
# MAGIC **Sección 2: Tendencias** (centro-izquierda)
# MAGIC * Evolución temporal de KPIs
# MAGIC * Líneas de tendencia
# MAGIC
# MAGIC **Sección 3: Composición** (centro-derecha)
# MAGIC * Distribución por segmentos
# MAGIC * Top/Bottom performers
# MAGIC
# MAGIC **Sección 4: Detalles** (área inferior)
# MAGIC * Tablas con drill-down
# MAGIC * Filtros interactivos
# MAGIC
# MAGIC ### Presentación de Insights
# MAGIC
# MAGIC #### Estructura de slide:
# MAGIC
# MAGIC 1. **Título accionable** (no descriptivo)
# MAGIC    * ❌ MAL: "Ventas por región"
# MAGIC    * ✅ BIEN: "Zona Sur tiene potencial sin explotar (+40% vs. Este)"
# MAGIC
# MAGIC 2. **Visualización clara**
# MAGIC    * Un gráfico principal
# MAGIC    * Resaltar el insight (color, anotaciones)
# MAGIC
# MAGIC 3. **Takeaway en 1 línea**
# MAGIC    * Cuántificar impacto
# MAGIC    * Acción recomendada
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 5️⃣ De Insight a Acción
# MAGIC
# MAGIC ### Framework: Insight → Implicación → Recomendación
# MAGIC
# MAGIC **Ejemplo:**
# MAGIC
# MAGIC **Insight:**
# MAGIC > "Clientes VIP representan 15% de base pero generan 45% de ganancia, con frecuencia de compra 3x mayor que regulares."
# MAGIC
# MAGIC **Implicación:**
# MAGIC > "Retener un cliente VIP vale 3 veces más que adquirir uno nuevo. Incluso pequeña reducción en churn VIP impacta significativamente."
# MAGIC
# MAGIC **Recomendación:**
# MAGIC > "Implementar programa de fidelización VIP con beneficios exclusivos. Inversión estimada: $50K. ROI esperado: 300% en 12 meses."
# MAGIC
# MAGIC ### Cuantificar Impacto
# MAGIC
# MAGIC Siempre incluir:
# MAGIC * **Situación actual** (baseline)
# MAGIC * **Mejora esperada** (uplift)
# MAGIC * **Impacto monetario** ($ o %)
# MAGIC * **Timeline** (cuándo ver resultados)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🎓 Resumen Unidad 4
# MAGIC
# MAGIC ### Conceptos Clave
# MAGIC
# MAGIC 1. **Pipelines**: Ingesta → Limpieza → Transform → Persist → Análisis
# MAGIC 2. **RFM**: Segmentación de clientes por comportamiento
# MAGIC 3. **BCG Matrix**: Análisis de portafolio de productos
# MAGIC 4. **Documentación**: Executive summary, hallazgos, recomendaciones
# MAGIC 5. **Comunicación**: Dashboards estructurados, insights accionables
# MAGIC
# MAGIC ### Próximos Pasos
# MAGIC
# MAGIC **TP07: Pipeline Integrador**
# MAGIC * Pipeline completo end-to-end
# MAGIC * Integración de todas las técnicas
# MAGIC * Persistencia en Delta Lake
# MAGIC * Modelado predictivo
# MAGIC
# MAGIC **TP08: Proyecto Final**
# MAGIC * Caso de negocio real
# MAGIC * Análisis RFM + BCG Matrix
# MAGIC * Dashboard ejecutivo
# MAGIC * Recomendaciones estratégicas
# MAGIC * Cuantificación de impacto
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🎆 ¡Momento de Integrar Todo!
# MAGIC
# MAGIC **Has adquirido todas las herramientas:**
# MAGIC * ✅ Análisis de datos (Pandas, PySpark, SQL)
# MAGIC * ✅ Visualización (Matplotlib, Seaborn)
# MAGIC * ✅ Modelado (Delta Lake, Feature Engineering, ML)
# MAGIC * ✅ Comunicación (Dashboards, Storytelling)
# MAGIC
# MAGIC **Ahora es momento de aplicarlas en proyectos completos.**
# MAGIC
# MAGIC 🚀 **¡Adelante con TP07 y TP08!**
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 📚 Recursos Finales
# MAGIC
# MAGIC ### Lectura Recomendada
# MAGIC * "Storytelling with Data" - Cole Nussbaumer Knaflic
# MAGIC * "The McKinsey Way" - Ethan Rasiel
# MAGIC * "Data Science for Business" - Foster Provost
# MAGIC
# MAGIC ### Comunidades
# MAGIC * Databricks Community Forums
# MAGIC * r/datascience (Reddit)
# MAGIC * Kaggle Discussions
# MAGIC
# MAGIC ### Práctica Continua
# MAGIC * Kaggle Competitions
# MAGIC * DataCamp Projects
# MAGIC * Personal projects con datos reales
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ✅ **Curso Completo - ¡Felicitaciones!**

# COMMAND ----------

