# Databricks notebook source
# DBTITLE 1,Encabezado U2
# MAGIC %md
# MAGIC # 📊 Unidad 2: Visualización de Datos
# MAGIC ## Laboratorio (Herramientas) - Universidad del Aconcagua
# MAGIC ### Contenido Teórico
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🎯 Objetivos de Aprendizaje
# MAGIC
# MAGIC Al finalizar esta unidad teórica, serás capaz de:
# MAGIC
# MAGIC 1. ✅ Comprender los principios fundamentales de visualización de datos
# MAGIC 2. ✅ Elegir el tipo de gráfico adecuado para cada situación
# MAGIC 3. ✅ Dominar matplotlib y seaborn para visualizaciones estáticas
# MAGIC 4. ✅ Crear dashboards interactivos profesionales
# MAGIC 5. ✅ Aplicar mejores prácticas en diseño visual
# MAGIC 6. ✅ Comunicar insights efectivamente con datos
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 📚 Contenido
# MAGIC
# MAGIC 1. Fundamentos de Visualización de Datos
# MAGIC 2. Tipos de Gráficos y Cuándo Usarlos
# MAGIC 3. Matplotlib: Visualizaciones Fundamentales
# MAGIC 4. Seaborn: Visualizaciones Estadísticas
# MAGIC 5. Dashboards y BI
# MAGIC 6. Mejores Prácticas en Visualización
# MAGIC 7. Storytelling con Datos
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### ⏱️ Duración Estimada: 2 horas

# COMMAND ----------

# DBTITLE 1,1. Fundamentos
# MAGIC %md
# MAGIC ## 1️⃣ Fundamentos de Visualización de Datos
# MAGIC
# MAGIC ### ¿Por Qué Visualizar?
# MAGIC
# MAGIC > *"Un gráfico vale más que mil filas de datos"*
# MAGIC
# MAGIC #### Ventajas de la Visualización:
# MAGIC
# MAGIC 👁️ **Percepción Rápida**
# MAGIC * El cerebro procesa imágenes 60,000x más rápido que texto
# MAGIC * Patrones y outliers son inmediatamente visibles
# MAGIC
# MAGIC 💡 **Descubrimiento de Insights**
# MAGIC * Revela relaciones ocultas
# MAGIC * Identifica tendencias y anomalías
# MAGIC * Facilita exploración interactiva
# MAGIC
# MAGIC 🗣️ **Comunicación Efectiva**
# MAGIC * Hace los datos accesibles a audiencias no técnicas
# MAGIC * Persuade y motiva a la acción
# MAGIC * Cuenta historias memorables
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Principios de Visualización Efectiva
# MAGIC
# MAGIC #### 1. **Claridad** > Complejidad
# MAGIC * Elimina elementos innecesarios (chartjunk)
# MAGIC * Un gráfico, un mensaje principal
# MAGIC * Usa etiquetas descriptivas
# MAGIC
# MAGIC #### 2. **Precisión**
# MAGIC * Respeta escalas y proporciones
# MAGIC * No manipules ejes para exagerar
# MAGIC * Indica unidades claramente
# MAGIC
# MAGIC #### 3. **Eficiencia**
# MAGIC * Máxima información con mínima tinta (Tufte)
# MAGIC * Usa colores con propósito, no decoración
# MAGIC * Aprovecha preattentive attributes
# MAGIC
# MAGIC #### 4. **Estética**
# MAGIC * Paletas de colores armónicas
# MAGIC * Tipografía legible
# MAGIC * Balance y alineación
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### La Gramática de Gráficos
# MAGIC
# MAGIC Todo gráfico está compuesto de:
# MAGIC
# MAGIC 1. **Datos**: La información a visualizar
# MAGIC 2. **Estética (Aesthetics)**: Mapeo de datos a propiedades visuales
# MAGIC    * Posición (x, y)
# MAGIC    * Color
# MAGIC    * Tamaño
# MAGIC    * Forma
# MAGIC 3. **Geométricas**: Representación visual
# MAGIC    * Puntos, líneas, barras, áreas
# MAGIC 4. **Escalas**: Cómo se traducen valores de datos a valores visuales
# MAGIC 5. **Coordenadas**: Sistema de coordenadas
# MAGIC 6. **Facets**: Subplots por categorías
# MAGIC
# MAGIC ---

# COMMAND ----------

# DBTITLE 1,2. Tipos de Gráficos
# MAGIC %md
# MAGIC ## 2️⃣ Tipos de Gráficos y Cuándo Usarlos
# MAGIC
# MAGIC ### 📊 Comparación
# MAGIC
# MAGIC #### **Gráfico de Barras**
# MAGIC * **Cuándo**: Comparar categorías
# MAGIC * **Ejemplo**: Ventas por sucursal, productos más vendidos
# MAGIC * **Orientación**: Vertical (pocas categorías) u horizontal (muchas categorías)
# MAGIC
# MAGIC #### **Gráfico de Barras Apiladas**
# MAGIC * **Cuándo**: Comparar totales Y composición
# MAGIC * **Ejemplo**: Ventas por región divididas por producto
# MAGIC
# MAGIC #### **Gráfico de Barras Agrupadas**
# MAGIC * **Cuándo**: Comparar subgrupos entre categorías
# MAGIC * **Ejemplo**: Ventas por trimestre de cada año
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 📈 Evolución Temporal
# MAGIC
# MAGIC #### **Gráfico de Líneas**
# MAGIC * **Cuándo**: Mostrar tendencias en el tiempo
# MAGIC * **Ejemplo**: Ventas mensuales, evolución de KPIs
# MAGIC * **Buena práctica**: Mantener líneas < 5 para claridad
# MAGIC
# MAGIC #### **Gráfico de Áreas**
# MAGIC * **Cuándo**: Enfatizar magnitud de cambio
# MAGIC * **Ejemplo**: Volumen de ventas acumulado
# MAGIC
# MAGIC #### **Gráfico de Áreas Apiladas**
# MAGIC * **Cuándo**: Mostrar composición cambiante
# MAGIC * **Ejemplo**: Market share de productos en el tiempo
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 🔵 Distribución
# MAGIC
# MAGIC #### **Histograma**
# MAGIC * **Cuándo**: Ver distribución de variable continua
# MAGIC * **Ejemplo**: Distribución de edades de clientes
# MAGIC * **Clave**: Elegir número de bins adecuado
# MAGIC
# MAGIC #### **Box Plot (Caja y Bigotes)**
# MAGIC * **Cuándo**: Comparar distribuciones, identificar outliers
# MAGIC * **Ejemplo**: Distribución de precios por categoría
# MAGIC * **Info**: Muestra mediana, cuartiles, outliers
# MAGIC
# MAGIC #### **Violin Plot**
# MAGIC * **Cuándo**: Similar a box plot pero más detalle
# MAGIC * **Ejemplo**: Distribución de tiempos de respuesta
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 🔴 Relaciones
# MAGIC
# MAGIC #### **Scatter Plot (Dispersión)**
# MAGIC * **Cuándo**: Ver relación entre dos variables continuas
# MAGIC * **Ejemplo**: Precio vs. ventas, edad vs. gasto
# MAGIC * **Agregado**: Puede incluir tamaño/color como 3ª/4ª dimensión
# MAGIC
# MAGIC #### **Heatmap (Mapa de Calor)**
# MAGIC * **Cuándo**: Ver correlaciones, patrones en matriz
# MAGIC * **Ejemplo**: Matriz de correlación, ventas por día×hora
# MAGIC
# MAGIC #### **Pairplot**
# MAGIC * **Cuándo**: Explorar relaciones entre múltiples variables
# MAGIC * **Ejemplo**: Análisis exploratorio de dataset
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 🍰 Composición
# MAGIC
# MAGIC #### **Gráfico de Torta (Pie Chart)**
# MAGIC * **Cuándo**: Mostrar partes de un todo (< 6 categorías)
# MAGIC * **Ejemplo**: Distribución de ventas por categoría
# MAGIC * **Limitación**: Difícil comparar ángulos pequeños
# MAGIC
# MAGIC #### **Donut Chart**
# MAGIC * **Cuándo**: Variante de torta con espacio para KPI central
# MAGIC * **Ejemplo**: % de cumplimiento de meta
# MAGIC
# MAGIC #### **Treemap**
# MAGIC * **Cuándo**: Jerarquías y proporciones anidadas
# MAGIC * **Ejemplo**: Ventas por región > ciudad > sucursal
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 📍 Tabla de Selección
# MAGIC
# MAGIC | Pregunta | Gráfico Recomendado |
# MAGIC |---|---|
# MAGIC | ¿Cómo se comparan las categorías? | Barras |
# MAGIC | ¿Cómo evoluciona en el tiempo? | Líneas |
# MAGIC | ¿Cómo se distribuye? | Histograma, Box plot |
# MAGIC | ¿Qué relación hay entre X e Y? | Scatter |
# MAGIC | ¿Cómo se compone el total? | Torta, Barras apiladas |
# MAGIC | ¿Dónde están los outliers? | Box plot, Scatter |
# MAGIC | ¿Qué variables se correlacionan? | Heatmap, Pairplot |
# MAGIC
# MAGIC ---

# COMMAND ----------

# DBTITLE 1,3-7. Herramientas y Prácticas
# MAGIC %md
# MAGIC ## 3️⃣ Matplotlib: La Base
# MAGIC
# MAGIC ### Biblioteca fundamental de Python para visualizaciones
# MAGIC
# MAGIC **Ventajas:**
# MAGIC * Control total sobre cada elemento
# MAGIC * Base de otras bibliotecas (seaborn, pandas.plot)
# MAGIC * Flexibilidad máxima
# MAGIC
# MAGIC **Estructura básica:**
# MAGIC ```python
# MAGIC import matplotlib.pyplot as plt
# MAGIC
# MAGIC # Crear figura y ejes
# MAGIC fig, ax = plt.subplots(figsize=(10, 6))
# MAGIC
# MAGIC # Graficar
# MAGIC ax.plot(x, y, marker='o', linestyle='-', color='blue', label='Serie 1')
# MAGIC
# MAGIC # Personalizar
# MAGIC ax.set_title('Título del Gráfico', fontsize=14, fontweight='bold')
# MAGIC ax.set_xlabel('Eje X')
# MAGIC ax.set_ylabel('Eje Y')
# MAGIC ax.legend()
# MAGIC ax.grid(True, alpha=0.3)
# MAGIC
# MAGIC plt.show()
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 4️⃣ Seaborn: Visualizaciones Estadísticas
# MAGIC
# MAGIC ### Biblioteca de alto nivel construida sobre matplotlib
# MAGIC
# MAGIC **Ventajas:**
# MAGIC * Estilos estéticos por defecto
# MAGIC * Funciones estadísticas integradas
# MAGIC * Menor código para gráficos complejos
# MAGIC
# MAGIC **Ejemplos:**
# MAGIC ```python
# MAGIC import seaborn as sns
# MAGIC
# MAGIC # Configurar estilo
# MAGIC sns.set_style('whitegrid')
# MAGIC sns.set_palette('husl')
# MAGIC
# MAGIC # Heatmap de correlación
# MAGIC sns.heatmap(df.corr(), annot=True, cmap='coolwarm', center=0)
# MAGIC
# MAGIC # Box plot por categoría
# MAGIC sns.boxplot(data=df, x='categoria', y='precio')
# MAGIC
# MAGIC # Pairplot para explorar relaciones
# MAGIC sns.pairplot(df, hue='categoria')
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 5️⃣ Dashboards y BI
# MAGIC
# MAGIC ### Dashboards Efectivos
# MAGIC
# MAGIC **Principios:**
# MAGIC 1. **Jerarquía visual**: KPIs arriba, detalles abajo
# MAGIC 2. **Contexto**: Comparaciones (vs. periodo anterior, vs. meta)
# MAGIC 3. **Interactividad**: Filtros, drill-down
# MAGIC 4. **Actualización**: Frecuencia según necesidad
# MAGIC
# MAGIC **Componentes típicos:**
# MAGIC * 📈 **KPI cards**: Métricas clave con indicadores
# MAGIC * 📉 **Gráficos de tendencia**: Evolución temporal
# MAGIC * 🍰 **Composición**: Distribución por segmentos
# MAGIC * 📊 **Comparaciones**: Ranking, benchmarks
# MAGIC * 📍 **Tablas**: Detalle granular
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 6️⃣ Mejores Prácticas
# MAGIC
# MAGIC ### Colores
# MAGIC ✅ Usa paletas consistentes⚡ Evita rojo/verde para daltonismo  
# MAGIC ✅ Significado: Rojo=negativo, Verde=positivo, Azul=neutral  
# MAGIC ❌ Evita colores saturados o fluorescentes  
# MAGIC
# MAGIC ### Títulos y Etiquetas
# MAGIC ✅ Título descriptivo que responde "qué" y "para qué"  
# MAGIC ✅ Ejes con unidades (%, $, kg)  
# MAGIC ✅ Etiquetas de datos cuando aportan valor  
# MAGIC ❌ Evita rotación de etiquetas > 45°  
# MAGIC
# MAGIC ### Escalas
# MAGIC ✅ Inicia eje Y en cero para barras  
# MAGIC ✅ Escala logarítmica para rangos amplios  
# MAGIC ❌ No rompas ejes para exagerar diferencias  
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 7️⃣ Storytelling con Datos
# MAGIC
# MAGIC ### Estructura de una Historia con Datos
# MAGIC
# MAGIC 1. **Contexto**: ¿Por qué es importante?
# MAGIC 2. **Pregunta**: ¿Qué queremos saber?
# MAGIC 3. **Datos**: ¿Qué muestran los datos?
# MAGIC 4. **Insight**: ¿Qué significa?
# MAGIC 5. **Acción**: ¿Qué debemos hacer?
# MAGIC
# MAGIC ### Patrones Narrativos
# MAGIC
# MAGIC **Problem-Solution:**
# MAGIC * Mostrar problema (tendencia negativa)
# MAGIC * Explicar causa (análisis)
# MAGIC * Proponer solución (recomendaciones)
# MAGIC
# MAGIC **Before-After:**
# MAGIC * Estado inicial
# MAGIC * Intervención
# MAGIC * Resultado
# MAGIC
# MAGIC **Zoom In-Out:**
# MAGIC * Vista general (dashboard ejecutivo)
# MAGIC * Drill-down en areas clave
# MAGIC * Detalle granular
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🎓 Resumen Unidad 2
# MAGIC
# MAGIC ### Conceptos Clave
# MAGIC 1. **Principios**: Claridad, precisión, eficiencia, estética
# MAGIC 2. **Selección**: Elegir el gráfico correcto según pregunta
# MAGIC 3. **Herramientas**: Matplotlib (control) + Seaborn (rapidez)
# MAGIC 4. **Dashboards**: Jerarquía visual, contexto, interactividad
# MAGIC 5. **Storytelling**: Contexto → Pregunta → Datos → Insight → Acción
# MAGIC
# MAGIC ### Próximos Pasos
# MAGIC
# MAGIC **TP03: Perfilado de Datos**
# MAGIC * Estadísticas descriptivas visuales
# MAGIC * Identificación de outliers
# MAGIC * Análisis de correlaciones
# MAGIC
# MAGIC **TP04: Dashboards Interactivos**
# MAGIC * Dashboard ejecutivo completo
# MAGIC * Múltiples visualizaciones coordinadas
# MAGIC * KPIs y tendencias
# MAGIC
# MAGIC ✅ **¡Listo para visualizar insights!**

# COMMAND ----------

