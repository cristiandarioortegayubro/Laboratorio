# 📊 Laboratorio (Herramientas) - Universidad del Aconcagua

Material educativo completo para la materia **Laboratorio (Herramientas)** de la carrera **Licenciatura en Analítica de Negocios**, de la **Facultad de Ciencias Económicas y Jurídicas**, en la Universidad del Aconcagua, Mendoza, Argentina.

## 🎯 Descripción

Este repositorio contiene material teórico y práctico para aprender análisis de datos, visualización, modelado y proyectos integradores usando **Databricks** como plataforma unificada de datos y machine learning.

## 📊 Estadísticas del Curso

| Métrica | Cantidad |
|---------|----------|
| **Unidades temáticas** | 4 |
| **Trabajos Prácticos** | 8 (TP01-TP08) |
| **Notebooks de Teoría** | 4 (uno por unidad) |
| **Material Complementario** | 16+ módulos avanzados |
| **Archivos de Datos (CSV)** | 5 archivos |
| **Registros totales** | ~170,000 (ventas + detalles + clientes + productos + sucursales) |
| **Duración estimada** | 60-80 horas (curso principal + complementario) |
| **Nivel** | Intermedio a Avanzado |

## ✨ Características Destacadas

🎯 **Configuración Portable**: Todos los notebooks usan rutas dinámicas y detección automática de usuario. Funcionan sin modificación en cualquier workspace de Databricks.

🗺️ **Análisis Geoespacial**: Uso extensivo de H3 (Uber's Hexagonal Hierarchical Spatial Index) para clustering espacial, agregaciones por zona, y visualizaciones de mapas.

📊 **Datasets Realistas**: 170,000+ registros de ventas de panadería con 5 tablas relacionadas, geolocalización real de clientes y sucursales, ideal para casos de uso empresariales.

🧠 **Material Complementario Extenso**: 16+ módulos opcionales avanzados por unidad (Plotly, Ensemble Methods, Redes Neuronales, MLflow, CI/CD).

📝 **Documentación Pedagógica**: Cada celda incluye comentarios explicativos detallados, referencias a conceptos teóricos, y mejores prácticas.

🎓 **Proyecto Final Integrador**: TP08 aplica todo lo aprendido en un pipeline completo Bronze-Silver-Gold con modelo ML, dashboards y documentación profesional.

## 🏗️ Estructura del Proyecto

El curso está organizado en **4 unidades temáticas** + **material complementario avanzado**, cada una con:
- 📖 **Notebooks de Teoría** (`.ipynb`): Fundamentos conceptuales y mejores prácticas
- 💻 **Trabajos Prácticos** (`.ipynb`): Ejercicios hands-on con datasets reales
- 🚀 **Material Complementario**: Temas avanzados opcionales por unidad

```
Laboratorio/
├── 00 - Programa del Curso/
│   └── Programa Completo.ipynb
│
├── 01 - Unidad-01-Analisis-de-Datos/
│   ├── 01 - Teoria/
│   │   └── Teoría - Análisis de Datos.ipynb
│   ├── 02 - Practicas/
│   │   ├── TP01 - Configuración Cloud y Almacenamiento.ipynb
│   │   └── TP02 - Análisis Exploratorio y Limpieza.ipynb
│   └── 03 - Material-Complementario/
│       ├── README-Material-Complementario-Unidad-01.ipynb
│       └── (módulos avanzados opcionales)
│
├── 02 - Unidad-02-Visualizacion-de-Datos/
│   ├── 01 - Teoria/
│   │   └── Teoría - Visualización de Datos.ipynb
│   ├── 02 - Practicas/
│   │   ├── TP03 - Perfilado de Datos.ipynb
│   │   └── TP04 - Agregaciones y Geoespacial.ipynb
│   └── 03 - Material-Complementario/
│       ├── README-Material-Complementario-Unidad-02.ipynb
│       ├── 01-Teoria-Visualizaciones-Interactivas-Plotly.ipynb
│       ├── 01-Practica-Visualizaciones-Interactivas-Plotly.ipynb
│       └── (3 módulos adicionales: Dashboards, Storytelling, Geoespacial)
│
├── 03 - Unidad-03-Modelado-de-Datos/
│   ├── 01 - Teoria/
│   │   └── Teoría - Modelado de Datos.ipynb
│   ├── 02 - Practicas/
│   │   ├── TP05 - Estructuración y Agregación.ipynb
│   │   └── TP06 - Feature Engineering y ML.ipynb
│   └── 03 - Material-Complementario/
│       ├── README-Material-Complementario-Unidad-03.ipynb
│       └── (4 módulos: Ensemble Methods, Redes Neuronales, Series Temporales, NLP)
│
├── 04 - Unidad-04-Proyectos-Integradores/
│   ├── 01 - Teoria/
│   │   └── Teoría - Proyectos Integradores.ipynb
│   ├── 02 - Practicas/
│   │   ├── TP07 - Pipeline Bronze-Silver-Gold.ipynb
│   │   └── TP08 - Proyecto Final Integrador.ipynb
│   └── 03 - Material-Complementario/
│       ├── README-Material-Complementario-Unidad-04.ipynb
│       ├── 02-Practica-Monitoreo-Modelos-MLflow.ipynb
│       └── (3 módulos: MLflow, CI/CD, Arquitecturas Avanzadas)
│
├── 05 - Datasets/
│   ├── productos.csv           (45 productos)
│   ├── clientes.csv            (500 clientes con geolocalización)
│   ├── sucursales.csv          (3 sucursales)
│   ├── ventas.csv              (55,986 transacciones)
│   └── detalles_ventas.csv     (113,523 líneas de detalle)
│
└── 06 - Material-Avanzado/
    └── README.md               (temas avanzados adicionales)
```

---

## 🎯 Material Complementario Avanzado

Cada unidad incluye una sección **03 - Material-Complementario** con temas avanzados opcionales que expanden los conceptos del curso principal.

### 📋 Formato de los Módulos

Cada módulo complementario incluye:
- 📖 **Notebook de Teoría**: Fundamentos conceptuales, comparaciones, casos de uso, mejores prácticas
- 💻 **Notebook Práctico**: Ejercicios resueltos aplicados al dataset de la panadería
- ⏱️ **Duración**: 2-4 horas por módulo (teoría + práctica)
- 🎯 **Nivel**: Intermedio a Avanzado

### 📚 Módulos por Unidad

#### Unidad 1: Análisis de Datos

**Módulos opcionales disponibles** (consultar README de la unidad para lista completa)

---

#### Unidad 2: Visualización de Datos (4 módulos)

**Módulo 01: Visualizaciones Interactivas con Plotly**
- **Contenido**: Introducción a Plotly, gráficos básicos (scatter, line, bar, heatmap), subplots, gráficos 3D, animaciones, personalización avanzada
- **Técnicas**: Express vs Graph Objects, theming, configuración de ejes, leyendas, tooltips interactivos
- **Proyecto práctico**: Dashboard interactivo de ventas con filtros dinámicos
- **Duración**: 3-4 horas

**Módulo 02: Mapas y Visualización Geoespacial**
- **Contenido**: Folium para mapas interactivos, capas de mapas, marcadores, círculos, heatmaps geoespaciales, integración con H3
- **Técnicas**: TileLayer personalizado, cluster markers, choropleth maps, análisis de densidad espacial
- **Proyecto práctico**: Mapa interactivo de sucursales con zonas de influencia y hotspots de clientes
- **Duración**: 3 horas

**Módulo 03: Storytelling con Datos**
- **Contenido**: Principios de narrativa con datos, estructura de presentaciones, diseño de slides con datos, dashboards narrativos
- **Técnicas**: Pirámide invertida, data-ink ratio, progresión de complejidad, contexto y conclusión
- **Proyecto práctico**: Presentación ejecutiva de análisis de ventas
- **Duración**: 2-3 horas

**Módulo 04: Dashboards Profesionales**
- **Contenido**: Diseño de dashboards efectivos, layout y composición, KPIs y métricas clave, actualización en tiempo real
- **Técnicas**: Grid layouts, widget design, color schemes profesionales, responsive design
- **Proyecto práctico**: Dashboard ejecutivo completo de la panadería
- **Duración**: 4 horas

---

#### Unidad 3: Modelado de Datos (4 módulos)

**Módulo 01: Ensemble Methods Avanzados**
- **Contenido**: Random Forest profundo, XGBoost, LightGBM, CatBoost, Stacking, Blending, Voting Classifiers
- **Técnicas**: Hyperparameter tuning avanzado, feature importance, SHAP values para interpretabilidad
- **Proyecto práctico**: Modelo ensemble para predicción de ventas con interpretabilidad SHAP
- **Duración**: 4 horas

**Módulo 02: Redes Neuronales con TensorFlow/Keras**
- **Contenido**: Perceptrones, redes fully-connected, arquitecturas profundas, regularización (Dropout, L1/L2), optimizadores (Adam, RMSprop), callbacks
- **Técnicas**: Data preprocessing para DL, batch training, early stopping, learning rate scheduling
- **Proyecto práctico**: Red neuronal para clasificación de productos por categoría
- **Duración**: 5 horas

**Módulo 03: Detección de Anomalías**
- **Contenido**: Isolation Forest, One-Class SVM, Local Outlier Factor, análisis estadístico de outliers, AutoEncoders
- **Técnicas**: Feature scaling para anomaly detection, umbral de decisión, interpretación de scores
- **Proyecto práctico**: Detección de transacciones fraudulentas en ventas
- **Duración**: 3 horas

**Módulo 04: Modelos de Series Temporales**
- **Contenido**: ARIMA, SARIMA, Prophet (Facebook), LSTM para series temporales, descomposición de series
- **Técnicas**: Stationarity tests, seasonal decomposition, cross-validation temporal, forecasting con incertidumbre
- **Proyecto práctico**: Forecasting de ventas diarias con estacionalidad
- **Duración**: 4-5 horas

---

#### Unidad 4: Proyectos Integradores (4 módulos)

**Módulo 01: MLOps y CI/CD para ML**
- **Contenido**: MLflow tracking avanzado, MLflow Model Registry, deployment de modelos, versionado de datasets, pipelines CI/CD
- **Técnicas**: Automatización de experimentos, registro de artifacts, staging/production workflow
- **Proyecto práctico**: Pipeline automático de reentrenamiento de modelo
- **Duración**: 4 horas

**Módulo 02: Monitoreo de Modelos en Producción**
- **Contenido**: Model drift detection, data drift, performance monitoring, alerting, reentrenamiento automático
- **Técnicas**: Statistical tests (KS test, PSI), distributional shift, logging de inferencias, dashboards de monitoreo
- **Proyecto práctico**: Sistema de monitoreo para modelo de ventas con MLflow
- **Duración**: 3-4 horas

**Módulo 03: A/B Testing y Experimentación**
- **Contenido**: Diseño de experimentos, pruebas de hipótesis, análisis de significancia estadística, tamaño de muestra
- **Técnicas**: Randomización, control groups, métricas de éxito, causal inference básico
- **Proyecto práctico**: A/B test de estrategias de descuento en ventas
- **Duración**: 3 horas

**Módulo 04: Arquitecturas de ML End-to-End**
- **Contenido**: Feature stores, batch vs real-time inference, microservicios de ML, lakehouse architecture completa
- **Técnicas**: REST APIs para modelos, caching de features, model serving patterns
- **Proyecto práctico**: Arquitectura completa de recomendación de productos
- **Duración**: 5 horas

---

### 🎓 Cómo Usar el Material Complementario

1. **Opcionalidad**: No es necesario completar todos los módulos para aprobar el curso principal
2. **Orden flexible**: Puedes elegir los módulos que más te interesen
3. **Prerrequisitos**: Se recomienda completar la unidad base antes de su material complementario
4. **Integración**: Muchos módulos son referenciados en el Proyecto Final (TP08)
5. **Profundidad**: Cada módulo es autocontenido y puede estudiarse independientemente

**Consultar los archivos README-Material-Complementario-Unidad-XX.ipynb** en cada unidad para el índice completo de módulos y contenido detallado específico.

---

## 📚 Contenido por Unidad

### Unidad 1: Análisis de Datos en la Nube
**Teoría:**
- Tipos de análisis de datos (descriptivo, diagnóstico, predictivo, prescriptivo)
- Computación en la nube para Data Science
- Arquitectura de Databricks
- Pandas vs PySpark: ¿cuándo usar cada uno?
- SQL en análisis de datos
- Ciclo de vida de proyectos de datos
- Mejores prácticas: documentación, performance, reproducibilidad

**Prácticos:**
- **TP01 - Configuración Cloud y Almacenamiento**: Setup de entorno, exploración de estructura, configuración de rutas portables, introducción a pathlib
- **TP02 - Análisis Exploratorio y Limpieza**: Carga de datos, análisis exploratorio con Pandas y PySpark, limpieza y transformaciones básicas

**Material Complementario:**
- Módulos avanzados opcionales (consultar README de la unidad)

---

### Unidad 2: Visualización de Datos
**Teoría:**
- Principios de visualización efectiva
- Tipos de gráficos y sus casos de uso
- Matplotlib: visualizaciones personalizables
- Seaborn: visualizaciones estadísticas
- Dashboards en Databricks SQL
- Mejores prácticas de diseño visual
- Data storytelling

**Prácticos:**
- **TP03 - Perfilado de Datos**: Análisis de calidad de datos, distribuciones, valores faltantes, outliers, correlaciones, visualizaciones exploratorias con Matplotlib y Seaborn
- **TP04 - Agregaciones y Geoespacial**: Agregaciones avanzadas, funciones de ventana, análisis geoespacial con H3 (Uber's Hexagonal Hierarchical Spatial Index)

**Material Complementario:**
- **Módulo 01**: Visualizaciones Interactivas con Plotly (scatter, line, bar, heatmap, subplots, animaciones)
- **Módulo 02**: Dashboards Profesionales con Dash
- **Módulo 03**: Data Storytelling Avanzado
- **Módulo 04**: Visualización Geoespacial Avanzada

---

### Unidad 3: Modelado de Datos
**Teoría:**
- Delta Lake: ACID transactions, schema evolution, time travel
- Agregaciones avanzadas y window functions
- Feature engineering (temporal, encoding, scaling, interactions)
- Workflows de Machine Learning en Databricks
- Métricas de evaluación de modelos
- MLflow para tracking y experimentos

**Prácticos:**
- **TP05 - Estructuración y Agregación**: Creación de tablas Delta, agregaciones complejas con groupBy y agg, funciones de ventana (window functions), análisis geoespacial con H3
- **TP06 - Feature Engineering y ML**: Feature engineering temporal, codificación de variables categóricas, escalado, entrenamiento de modelos, evaluación, MLflow tracking

**Material Complementario:**
- **Módulo 01**: Ensemble Methods Avanzados (Random Forest, XGBoost, LightGBM, Stacking, SHAP)
- **Módulo 02**: Redes Neuronales con TensorFlow/Keras
- **Módulo 03**: Series Temporales (ARIMA, Prophet, LSTM)
- **Módulo 04**: Procesamiento de Lenguaje Natural (NLP)

---

### Unidad 4: Proyectos Integradores
**Teoría:**
- Arquitectura Medallion (Bronze-Silver-Gold)
- Pipelines ETL end-to-end
- Feature engineering avanzado
- Análisis predictivo y prescriptivo
- Dashboards ejecutivos
- Documentación y presentación profesional
- Mejores prácticas de producción
- MLflow para producción

**Prácticos:**
- **TP07 - Pipeline Bronze-Silver-Gold**: Implementación completa de arquitectura Medallion, capa Bronze (ingesta cruda), capa Silver (limpieza y estructuración), capa Gold (agregaciones y features para analytics/ML)
- **TP08 - Proyecto Final Integrador**: Análisis end-to-end completo, desde ingesta hasta modelo predictivo, dashboards ejecutivos, documentación profesional, recomendaciones de negocio

**Material Complementario:**
- **Módulo 01**: MLflow Avanzado (tracking, registry, deployment)
- **Módulo 02**: Monitoreo de Modelos en Producción (MLflow practice incluido)
- **Módulo 03**: CI/CD para Data Science
- **Módulo 04**: Arquitecturas de ML Avanzadas

---

## 🛠️ Tecnologías

### Plataforma
- **Databricks Free Edition** (anteriormente Community Edition)
- Notebooks Jupyter en la nube
- Compute clusters serverless y configurables

### Lenguajes
- **Python**: PySpark para procesamiento distribuido, Pandas para análisis local
- **SQL**: Databricks SQL para consultas y Delta Lake
- **Markdown**: Documentación y narrativa en notebooks

### Librerías y Frameworks

**Análisis de Datos:**
- PySpark (procesamiento distribuido)
- Pandas (análisis en memoria)
- NumPy (operaciones numéricas)
- Delta Lake (almacenamiento ACID, time travel)

**Visualización:**
- Matplotlib & Seaborn (visualizaciones estáticas)
- Plotly (visualizaciones interactivas)
- Databricks SQL Dashboards

**Machine Learning:**
- Scikit-learn (modelos clásicos)
- MLlib (ML distribuido en Spark)
- XGBoost, LightGBM (ensemble methods)
- TensorFlow/Keras (deep learning - material complementario)
- MLflow (tracking, registry, deployment)

**Geoespacial:**
- H3 (Uber's Hexagonal Hierarchical Spatial Index)
- Folium (mapas interactivos)

---

## 📝 Requisitos Previos

### Conocimientos Recomendados

✅ **Básicos (Esenciales)**:
- Python básico (variables, funciones, estructuras de control)
- SQL básico (SELECT, WHERE, JOIN, GROUP BY)
- Conceptos de estadística descriptiva

🔸 **Intermedios (Recomendados)**:
- Pandas para análisis de datos
- Visualización con Matplotlib/Seaborn
- Conceptos de Machine Learning (opcional para las primeras unidades)

### Requisitos Técnicos

* Cuenta en [Databricks Free Edition](https://www.databricks.com/try-databricks)
* Navegador web moderno (Chrome, Firefox, Edge, Safari)
* Sin necesidad de instalación local (todo corre en la nube)

### Tiempo de Dedicación

| Material | Tiempo Estimado |
|----------|----------------|
| Curso principal (4 unidades) | 40-50 horas |
| Material complementario opcional | 20-30 horas |
| Proyecto final | 10-15 horas |
| **Total** | **60-80 horas** |

---

## 🚀 Cómo Usar Este Repositorio

### Opción 1: Clonar en Databricks (Recomendado)

1. En tu workspace de Databricks, ir a **Repos**
2. Clic en **Add Repo**
3. Pegar la URL: `https://github.com/cristiandarioortegayubro/Laboratorio`
4. Los notebooks estarán listos para ejecutar

### Opción 2: Importar Notebooks Manualmente

1. Descargar los `.ipynb` desde GitHub
2. En Databricks, ir a **Workspace** → **Import**
3. Seleccionar los notebooks descargados
4. Organizar en carpetas según la estructura del curso

---

## 📖 Orden de Estudio Recomendado

1. **Comenzar con la teoría** de cada unidad antes de los prácticos
2. **Ejecutar todos los notebooks** en orden secuencial
3. **Experimentar modificando el código** para entender los conceptos
4. **Completar los ejercicios** propuestos en cada práctico
5. **Proyecto final (TP08)**: Aplicar todo lo aprendido en un caso real

---


## 📊 Dataset Principal: Panadería

El proyecto utiliza un **dataset realista de ventas de panadería** ubicado en `05 - Datasets/`, que incluye 5 archivos CSV:

### Archivos de Datos

1. **productos.csv** (45 productos)
   - Información de catálogo: producto_id, nombre, categoría, precios, costos, márgenes
   - Categorías: Panadería, Pastelería, Bebidas, Tortas, Snacks
   
2. **clientes.csv** (500 clientes)
   - Datos de clientes: cliente_id, nombre, email, teléfono
   - **Geolocalización**: latitud, longitud, índice H3 (hexagonal spatial index)
   - Datos de comportamiento: preferencia_categoría, es_vip
   
3. **sucursales.csv** (3 sucursales)
   - Información de locales: sucursal_id, nombre, dirección
   - Geolocalización: latitud, longitud, ciudad
   
4. **ventas.csv** (55,986 transacciones)
   - Cabecera de ventas: venta_id, fecha, hora, sucursal_id, cliente_id, total
   - Año completo de datos (2024), permite análisis de series temporales
   
5. **detalles_ventas.csv** (113,523 líneas de detalle)
   - Líneas de venta: venta_id, producto_id, cantidad, precio_unitario, descuento, subtotal
   - Relación 1:N con ventas (promedio ~2 productos por transacción)

### Casos de Uso Ideales

✅ **ETL y Arquitectura Medallion**: Bronze → Silver → Gold  
✅ **Feature Engineering**: Variables temporales, espaciales, agregaciones  
✅ **Análisis Geoespacial**: Clustering con H3, zonas de alta facturación  
✅ **Series Temporales**: Forecasting de ventas, estacionalidad  
✅ **Machine Learning**: Modelos de regresión, clasificación, segmentación  
✅ **Dashboards de Negocio**: Visualizaciones ejecutivas, KPIs  

---

## 🎓 Objetivos de Aprendizaje

Al completar este laboratorio, los estudiantes serán capaces de:

✅ Diseñar y ejecutar pipelines de datos end-to-end  
✅ Aplicar arquitecturas de datos profesionales (Medallion)  
✅ Crear visualizaciones efectivas y dashboards ejecutivos  
✅ Construir modelos de machine learning en producción  
✅ Documentar y presentar proyectos de datos  
✅ Aplicar mejores prácticas de ingeniería de datos y ML  

---

## 📝 Formato de Notebooks

Todos los notebooks de teoría están en formato **Jupyter Notebook estándar (`.ipynb`)** para garantizar:
- ✅ Renderizado correcto en GitHub
- ✅ Compatibilidad con Jupyter, VS Code, Google Colab
- ✅ Versionamiento limpio en Git
- ✅ Visualización de markdown, código y outputs

---

## ❓ Preguntas Frecuentes (FAQ)

### 👤 **¿Puedo usar este material si no soy estudiante de la UDA?**
Sí, el material es de acceso libre para fines académicos. Solo solicitamos mencionar la fuente.

### 💻 **¿Necesito instalar algo en mi computadora?**
No. Todo el curso se ejecuta en Databricks en la nube. Solo necesitas un navegador web y una cuenta gratuita en Databricks.

### 💰 **¿El curso tiene algún costo?**
No. El material educativo es gratuito y la plataforma Databricks Free Edition también es gratuita.

### 📅 **¿En qué orden debo estudiar el material?**
Sigue el orden numérico: Unidad 1 → 2 → 3 → 4. Dentro de cada unidad: Teoría → Prácticos → Material Complementario (opcional).

### ⌛ **¿Cuánto tiempo toma completar el curso?**
- Curso principal (TP01-TP08): 40-50 horas
- Material complementario: 20-30 horas adicionales (opcional)
- Total: 60-80 horas

### 🌎 **¿Puedo usar los notebooks con otros datasets?**
Sí, pero necesitarás adaptar las rutas y esquemas de datos. Los notebooks están diseñados para ser portables y bien documentados para facilitar la adaptación.

### 👥 **¿Hay un foro o comunidad para hacer preguntas?**
Para estudiantes de la UDA, consultar con los docentes. Para otros usuarios, pueden abrir issues en el repositorio de GitHub.

### 📂 **¿Puedo contribuir al repositorio?**
Sí. Pull requests con correcciones, mejoras o contenido adicional son bienvenidos.

---

## 🔗 Recursos Adicionales

- [Documentación oficial de Databricks](https://docs.databricks.com/)
- [PySpark API Reference](https://spark.apache.org/docs/latest/api/python/)
- [Delta Lake Documentation](https://docs.delta.io/)
- [Databricks Academy](https://www.databricks.com/learn/training)

---

## 📄 Licencia

Este material educativo es de uso libre para fines académicos. Se solicita mencionar la fuente al reutilizar el contenido.

---

## 📧 Contacto

Para consultas sobre el curso:
- **Email**: cortega@uda.edu.ar
- **Carrera**: Licenciatura en Analítica de Negocios
- **Facultad**: Ciencias Económicas y Jurídicas
- **Institución**: Universidad del Aconcagua
- **GitHub**: [@cristiandarioortegayubro](https://github.com/cristiandarioortegayubro)

---

**⭐ Si este repositorio te resultó útil, considera darle una estrella en GitHub!**
