# 📚 Unidad 01: Análisis de Datos - Material Complementario
## Índice de Contenidos
### Laboratorio (Herramientas) - Universidad del Aconcagua

---

## 🎯 Visión General

Este material complementario profundiza en técnicas avanzadas de análisis de datos, complementando el contenido principal de la Unidad 01. Incluye **4 módulos temáticos** con teoría + práctica ejecutable.

### 📊 Estadísticas del Material

* **Módulos**: 4
* **Notebooks totales**: 8 (4 teoría + 4 práctica)
* **Celdas de código**: ~55+
* **Duración estimada**: 4-5 horas
* **Nivel**: Intermedio-Avanzado

---

## 📖 Estructura de Módulos

### 🔍 Módulo 01: EDA Avanzado con Profiling

**🎯 Objetivos:**
* Automatizar análisis exploratorio con ydata-profiling
* Generar reportes exhaustivos de calidad de datos
* Comparar datasets con Sweetviz
* Analizar relaciones con variable target

**📓 Notebooks:**

#### [01-Teoria-EDA-Avanzado-con-Profiling](#notebook-1402451318137582)
* **Celdas**: 9
* **Contenido**:
  * Diferencias EDA básico vs avanzado
  * ydata-profiling (instalación, configuración, reportes)
  * Sweetviz (comparaciones, target analysis)
  * Comparación de herramientas
  * Mejores prácticas
  * 4 casos de uso con datasets de panadería
  * Integración con el curso
  * Recursos adicionales

#### [01-Practica-EDA-Avanzado-con-Profiling](#notebook-1402451318137583)
* **Celdas**: 20
* **Ejercicios**:
  1. Profiling completo con ydata-profiling
  2. Comparación Train/Test con Sweetviz
  3. Target Analysis
  4. Interpretación de warnings
  5. Comparación antes/después de limpieza
* **Datasets**: ventas.csv (50,000+ transacciones)

**⏱️ Duración**: 90-120 minutos

---

### 🧹 Módulo 02: Manejo de Datos Faltantes y Outliers

**🎯 Objetivos:**
* Identificar tipos de valores faltantes (MCAR, MAR, MNAR)
* Aplicar estrategias de imputación apropiadas
* Detectar outliers con múltiples métodos
* Decidir tratamiento según contexto de negocio

**📓 Notebooks:**

#### [02-Teoria-Manejo-Datos-Faltantes-Outliers](#notebook-1402451318137584)
* **Celdas**: 7
* **Contenido**:
  * **Parte I: Datos Faltantes**
    * Tipos de mecanismos (MCAR, MAR, MNAR)
    * Análisis de patrones
    * 6 estrategias de imputación (simple, KNN, MICE, etc.)
    * Validación de imputaciones
  * **Parte II: Outliers**
    * 5 métodos de detección (IQR, Z-score, Isolation Forest, DBSCAN, Mahalanobis)
    * Outliers univariados vs multivariados
    * 5 estrategias de tratamiento
    * Workflow de decisión

#### [02-Practica-Manejo-Datos-Faltantes-Outliers](#notebook-1402451318137585)
* **Celdas**: 15
* **Ejercicios**:
  1. Diagnóstico de datos faltantes
  2. Imputación comparativa (Simple vs KNN)
  3. Detección de outliers (IQR, Z-score, Isolation Forest)
  4. Tratamiento de outliers
  5. Validación de resultados
* **Técnicas**: Simula missing MAR y MCAR realistas

**⏱️ Duración**: 90 minutos

---

### 🔗 Módulo 03: Análisis de Correlaciones

**🎯 Objetivos:**
* Entender tipos de correlación (Pearson, Spearman, Kendall)
* Interpretar matrices de correlación
* Detectar multicolinealidad
* Seleccionar features basado en correlación

**📓 Notebooks:**

#### [03-Teoria-Analisis-Correlaciones](#notebook-1402451318137586)
* **Celdas**: 1 (completa)
* **Contenido**:
  * Tipos de correlación y cuándo usar cada una
  * Interpretación de valores (-1 a +1)
  * Multicolinealidad: impacto y detección (VIF)
  * Workflow de selección de features
  * Mejores prácticas

#### [03-Practica-Analisis-Correlaciones](#notebook-1402451318137587)
* **Celdas**: 1 (completa con 4 ejercicios)
* **Ejercicios**:
  1. Matriz de correlación y heatmap
  2. Comparación Pearson vs Spearman
  3. Detección de multicolinealidad (threshold > 0.8)
  4. Selección de features por correlación con target
* **Visualizaciones**: Heatmaps con seaborn

**⏱️ Duración**: 60 minutos

---

### 📊 Módulo 04: Estadística Descriptiva Avanzada

**🎯 Objetivos:**
* Dominar métricas avanzadas (skewness, kurtosis)
* Aplicar estadística robusta (MAD, mediana)
* Realizar tests de normalidad
* Implementar bootstrapping para IC

**📓 Notebooks:**

#### [04-Teoria-Estadistica-Descriptiva-Avanzada](#notebook-1402451318137588)
* **Celdas**: 1 (completa)
* **Contenido**:
  * Métricas de forma (skewness, kurtosis)
  * Estadística robusta (MAD, percentiles)
  * Tests de normalidad (Shapiro-Wilk, KS)
  * Bootstrapping para intervalos de confianza
  * Percentiles y cuartiles
  * Cuándo usar cada métrica

#### [04-Practica-Estadistica-Descriptiva-Avanzada](#notebook-1402451318137589)
* **Celdas**: 1 (completa con 5 ejercicios)
* **Ejercicios**:
  1. Cálculo de skewness y kurtosis
  2. Estadística robusta con MAD
  3. Tests de normalidad
  4. Bootstrapping para IC (media y mediana)
  5. Reporte completo de distribución
* **Visualizaciones**: Histogramas, Q-Q plots, distribuciones bootstrap

**⏱️ Duración**: 60-75 minutos

---

## 🗺️ Mapa de Dependencias

```
Módulo 01 (EDA Avanzado)
    ↓
    ├──> Módulo 02 (Missing/Outliers) ──┐
    │                                    ├──> Módulo 03 (Correlaciones)
    └────────────────────────────────────┘            ↓
                                              Módulo 04 (Est. Avanzada)
```

**Orden recomendado**: 01 → 02 → 03 → 04 (secuencial)

---

## 💡 Conceptos Clave por Módulo

### Módulo 01
* ydata-profiling, Sweetviz
* Profiling automatizado
* Comparaciones de datasets
* Target analysis

### Módulo 02
* MCAR, MAR, MNAR
* Imputación (simple, KNN, MICE)
* IQR, Z-score, Isolation Forest
* Multicolinealidad, VIF

### Módulo 03
* Pearson, Spearman, Kendall
* Matrices de correlación
* Multicolinealidad
* Feature selection

### Módulo 04
* Skewness, Kurtosis
* MAD (Median Absolute Deviation)
* Shapiro-Wilk, Kolmogorov-Smirnov
* Bootstrapping

---

## 🛠️ Herramientas y Librerías

**Análisis de Datos:**
* `pandas`, `numpy`
* `ydata-profiling` (pandas-profiling)
* `sweetviz`

**Visualización:**
* `matplotlib`, `seaborn`
* `plotly`

**Machine Learning:**
* `scikit-learn` (imputers, Isolation Forest, clustering)
* `scipy.stats` (tests estadísticos)

**Databricks:**
* PySpark (para datasets grandes)
* Delta Lake (almacenamiento)
* Databricks widgets (parámetros)

---

## 📁 Estructura de Archivos

```
/Users/cortega@uda.edu.ar/Laboratorio/
└── Unidad-01-Analisis-de-Datos/
    └── Material-Complementario/
        ├── 📄 README.md (este archivo)
        ├── 📓 01-Teoria-EDA-Avanzado-con-Profiling
        ├── 💻 01-Practica-EDA-Avanzado-con-Profiling
        ├── 📓 02-Teoria-Manejo-Datos-Faltantes-Outliers
        ├── 💻 02-Practica-Manejo-Datos-Faltantes-Outliers
        ├── 📓 03-Teoria-Analisis-Correlaciones
        ├── 💻 03-Practica-Analisis-Correlaciones
        ├── 📓 04-Teoria-Estadistica-Descriptiva-Avanzada
        └── 💻 04-Practica-Estadistica-Descriptiva-Avanzada
```

---

## 🎓 Cómo Usar este Material

### Para Estudiantes

1. **Estudia la teoría primero**: Lee el notebook de teoría antes de la práctica
2. **Ejecuta los ejercicios**: Corre todas las celdas, experimenta con parámetros
3. **Aplica en TPs**: Usa estas técnicas en tus Trabajos Prácticos
4. **Integra conceptos**: Conecta con contenido de la unidad principal

### Para Docentes

1. **Material de referencia**: Úsalo como complemento en clases
2. **Ejemplos prácticos**: Muestra notebooks en vivo
3. **Ejercicios adicionales**: Asigna como tarea
4. **Evaluación**: Inspira preguntas de examen/TP

### Sugerencias de Estudio

**Por módulo (2-3 horas cada uno):**
* 30 min: Leer teoría completa
* 60 min: Ejecutar y entender práctica
* 30 min: Aplicar en dataset propio
* 30 min: Repasar conceptos clave

**Intensivo (1 día completo):**
* Mañana: Módulos 01 y 02
* Tarde: Módulos 03 y 04
* Noche: Proyecto integrador mini

---

## 🔗 Conexión con el Curso Principal

### Unidad 01 (Principal)
* **TP01**: Análisis exploratorio básico
* **TP02**: Limpieza de datos
* **TP03**: Visualización

### Material Complementario
* **Módulo 01**: Automatiza EDA del TP01
* **Módulo 02**: Profundiza limpieza del TP02
* **Módulos 03-04**: Añade análisis avanzado a todos los TPs

---

## 📊 Datasets Utilizados

### Dataset Principal: Ventas de Panadería
* **Ubicación**: `/dbfs/FileStore/Laboratorio/Datasets/ventas.csv`
* **Filas**: 50,000+
* **Columnas**: ~15 (cliente_id, producto, monto, cantidad, fecha, etc.)
* **Características**: Datos realistas con missing, outliers, estacionalidad

### Datasets Derivados
* Clientes (con ubicaciones H3)
* Productos (con categorías)
* Transacciones (joins aplicados)

---

## ✅ Checklist de Completitud

### Módulo 01
- ✅ Teoría: ydata-profiling y Sweetviz
- ✅ Práctica: 5 ejercicios ejecutables
- ✅ Visualizaciones: Reportes HTML
- ✅ Integración: Casos de uso panadería

### Módulo 02
- ✅ Teoría: MCAR/MAR/MNAR + 5 métodos outliers
- ✅ Práctica: Imputación comparativa
- ✅ Código: IQR, Z-score, Isolation Forest
- ✅ Decisiones: Workflow antes/después

### Módulo 03
- ✅ Teoría: Pearson, Spearman, Kendall
- ✅ Práctica: Heatmaps y feature selection
- ✅ Multicolinealidad: Detección y VIF
- ✅ Casos: Target correlation analysis

### Módulo 04
- ✅ Teoría: Skewness, kurtosis, MAD, bootstrapping
- ✅ Práctica: Tests normalidad + IC
- ✅ Visualizaciones: Q-Q plots, distribuciones
- ✅ Reporte: Función de análisis completo

---

## 🚀 Próximos Pasos

### Inmediatos
1. ✅ Explorar cada módulo en orden
2. ✅ Ejecutar todos los notebooks de práctica
3. ✅ Aplicar técnicas en TPs

### Avanzados
1. 🔄 Crear pipeline de EDA automatizado propio
2. 🔄 Integrar con Unidad 02 (Visualización)
3. 🔄 Aplicar en proyecto final

---

## 📚 Recursos Adicionales

### Documentación Oficial
* [ydata-profiling](https://docs.profiling.ydata.ai/)
* [Sweetviz](https://github.com/fbdesignpro/sweetviz)
* [scikit-learn Preprocessing](https://scikit-learn.org/stable/modules/preprocessing.html)

### Papers y Referencias
* Little & Rubin (2019): Statistical Analysis with Missing Data
* Isolation Forest (Liu et al., 2008)
* MICE Algorithm (van Buuren & Groothuis-Oudshoorn, 2011)

---

**Universidad del Aconcagua 🇦🇷**