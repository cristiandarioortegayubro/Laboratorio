# 📊 Laboratorio (Herramientas) - Universidad del Aconcagua

Material educativo completo para la materia **Laboratorio (Herramientas)** de la carrera **Licenciatura en Analítica de Negocios**, de la **Facultad de Ciencias Económicas y Jurídicas**, en la Universidad del Aconcagua, Mendoza, Argentina.

## 🎯 Descripción

Este repositorio contiene material teórico y práctico para aprender análisis de datos, visualización, modelado y proyectos integradores usando **Databricks** como plataforma unificada de datos y machine learning.

## 🏗️ Estructura del Proyecto

El curso está organizado en **4 unidades temáticas**, cada una con:
- 📖 **Notebook de Teoría** (`.ipynb`): Fundamentos conceptuales y mejores prácticas
- 💻 **Trabajos Prácticos** (`.ipynb`): Ejercicios hands-on con datasets reales

```
Laboratorio/
├── Programa del Curso/
│   └── Programa Completo.ipynb
│
├── Unidad-01-Analisis-de-Datos/
│   ├── Teoria/
│   │   └── Teoría - Análisis de Datos.ipynb          (9 celdas, 18K)
│   └── TP01 - Introducción a Databricks.ipynb
│
├── Unidad-02-Visualizacion-de-Datos/
│   ├── Teoria/
│   │   └── Teoría - Visualización de Datos.ipynb     (9 celdas, 21K)
│   └── TP02-TP06/                                    (múltiples notebooks)
│
├── Unidad-03-Modelado-de-Datos/
│   ├── Teoria/
│   │   └── Teoría - Modelado de Datos.ipynb          (8 celdas, 28K)
│   └── TP03-TP06/                                    (múltiples notebooks)
│
├── Unidad-04-Proyectos-Integradores/
│   ├── Teoria/
│   │   └── Teoría - Proyectos Integradores.ipynb     (10 celdas, 56K)
│   └── Proyecto-Final/
│       ├── TP07 - Pipeline Integrador.ipynb
│       └── TP08 - Proyecto Final.ipynb
│
└── Material-Complementario/                          ⭐ NUEVO
    ├── 01-Optimizacion-Hiperparametros/              ✅ COMPLETO
    │   ├── Teoría - Optimización de Hiperparámetros con Optuna.ipynb
    │   └── Práctica - Optimización con Optuna.ipynb
    ├── 02-Validacion-Cruzada-Avanzada/               📅 Planificado
    ├── 03-Seleccion-Caracteristicas/                 📅 Planificado
    ├── 04-Interpretabilidad-Modelos/                 📅 Planificado
    ├── 05-AutoML-y-Feature-Store/                    📅 Planificado
    └── 06-MLflow-Tracking/                           📅 Planificado
```

---

## 🎯 Material Complementario (Nuevo)

Sección dedicada a **temas avanzados** que profundizan conceptos del curso principal. Cada módulo incluye:
- 📖 **Notebook de Teoría**: Conceptos, comparaciones, mejores prácticas
- 💻 **Notebook Práctico**: Ejercicios resueltos con el dataset de la panadería

### ✅ Módulos Disponibles

#### 01. Optimización de Hiperparámetros con Optuna
**Estado**: ✅ Completo (2 notebooks, 31 celdas)

**Contenido teórico**:
- Diferencia entre parámetros e hiperparámetros
- Métodos tradicionales: Grid Search y Random Search
- Introducción a Optuna (búsqueda bayesiana con TPE)
- Técnicas avanzadas: Pruning automático, paralelización, persistencia
- Visualizaciones de optimización (historia, importancia, slice plots)
- Mejores prácticas y casos de uso

**Contenido práctico**:
- Ejercicio 1: Optimizar Gradient Boosting Regressor
- Ejercicio 2: Optimización multi-objetivo (precisión vs. velocidad)
- Ejercicio 3: Persistencia en base de datos SQLite
- Ejercicio 4: Optimización con features geoespaciales H3
- Comparación de resultados entre modelos
- Feature importance con datos espaciales

**Por qué es importante**:
- Mejora significativa en rendimiento de modelos ML
- Ahorro de tiempo vs. métodos tradicionales
- Integración con contenido geoespacial H3 del curso
- Preparación para workflows profesionales de ML

---

### 📅 Módulos Planificados

#### 02. Validación Cruzada Avanzada
- Stratified K-Fold para datos desbalanceados
- Time Series Cross-Validation
- Group K-Fold (evitar data leakage)
- Leave-One-Out CV
- Validación espacial (split por zonas geográficas)

#### 03. Selección de Características
- Filter methods (correlación, chi-squared, mutual information)
- Wrapper methods (RFE - Recursive Feature Elimination)
- Embedded methods (Lasso, Ridge, feature importance)
- SHAP values para selección
- Selección con features espaciales

#### 04. Interpretabilidad de Modelos
- SHAP (SHapley Additive exPlanations)
- LIME (Local Interpretable Model-agnostic Explanations)
- Partial Dependence Plots (PDP)
- Feature Importance global y local
- Interpretabilidad de features espaciales

#### 05. AutoML y Feature Store
- AutoML con H2O.ai
- AutoML con PyCaret
- Databricks Feature Store
- Feature engineering automatizado
- MLflow integration para tracking

#### 06. MLflow Tracking y Model Registry
- Logging de experimentos y métricas
- Comparación de runs
- Model Registry de Databricks
- Versionamiento de modelos
- Deployment y A/B testing

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

**Práctico (TP01):**
- Introducción a Databricks workspace
- Carga y exploración de datos
- Análisis exploratorio con Pandas y PySpark
- Consultas SQL básicas

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

**Prácticos (TP02-TP06):**
- Visualizaciones exploratorias
- Gráficos comparativos y de tendencias
- Dashboards interactivos
- Presentaciones con storytelling

---

### Unidad 3: Modelado de Datos
**Teoría:**
- Delta Lake: ACID transactions, schema evolution, time travel
- Agregaciones avanzadas y window functions
- Feature engineering (temporal, encoding, scaling, interactions)
- Workflows de Machine Learning en Databricks
- Métricas de evaluación de modelos

**Prácticos (TP03-TP06):**
- Transformaciones avanzadas con Delta Lake
- Feature engineering para ML
- Entrenamiento y evaluación de modelos
- Despliegue de modelos predictivos

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

**Prácticos (TP07-TP08):**
- **TP07 - Pipeline Integrador**: Implementación completa Bronze → Silver → Gold
- **TP08 - Proyecto Final**: Análisis end-to-end con recomendaciones de negocio

---

## 🛠️ Tecnologías

- **Plataforma**: Databricks Community Edition / Databricks Free Edition
- **Lenguajes**: Python (PySpark), SQL, Markdown
- **Librerías principales**:
  - PySpark (análisis distribuido)
  - Pandas (análisis en memoria)
  - Matplotlib & Seaborn (visualización)
  - MLlib (machine learning)
  - Delta Lake (almacenamiento ACID)

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


## 📊 Dataset Principal

El proyecto utiliza un **dataset realista de ventas de panadería** que incluye:
- Transacciones de clientes
- Información de productos
- Datos temporales (fechas, horas, estacionalidad)
- Variables categóricas y numéricas

Ideal para aprender:
- ETL y limpieza de datos
- Feature engineering
- Análisis de series temporales
- Modelos predictivos (forecasting, segmentación)
- Dashboards de negocio

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
