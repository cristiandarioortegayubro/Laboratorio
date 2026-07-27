# 📚 Material Complementario - Laboratorio (Herramientas)

## 🎯 Propósito

Material avanzado complementario para profundizar temas del curso principal de **Laboratorio (Herramientas)** en la **Licenciatura en Analítica de Negocios** de la Universidad del Aconcagua.

Este contenido está diseñado para:
- 🚀 Estudiantes que quieren ir más allá del programa base
- 💡 Profundizar conceptos de ML y optimización
- 🛠️ Aprender herramientas modernas de la industria
- 📊 Aplicar técnicas avanzadas con el dataset de la panadería

---

## 🏗️ Estructura

Cada módulo incluye:
- **📖 Notebook de Teoría**: Fundamentos conceptuales, comparaciones, mejores prácticas
- **💻 Notebook Práctico**: Ejercicios hands-on con datasets reales del curso
- **🎓 Integración**: Conexión con contenido del curso principal (features H3, pipeline Medallion)

---

## ✅ Módulos Completos

### 01. Optimización de Hiperparámetros con Optuna

**📊 Estadísticas**: 2 notebooks, 31 celdas, ~4 horas de material

**Teoría** (`Teoría - Optimización de Hiperparámetros con Optuna.ipynb`):
- ¿Qué son los hiperparámetros? Diferencia con parámetros
- Métodos tradicionales: Grid Search vs Random Search
- Introducción a Optuna (búsqueda bayesiana con TPE)
- Visualizaciones interactivas (historia, importancia, slice plots)
- Técnicas avanzadas: Pruning automático, paralelización, persistencia
- Mejores prácticas y cuándo usar cada método

**Práctica** (`Práctica - Optimización con Optuna.ipynb`):
- **Ejercicio 1**: Optimizar Gradient Boosting Regressor
  - Configuración de hiperparámetros
  - Validación cruzada
  - Comparación con Random Forest
  
- **Ejercicio 2**: Optimización multi-objetivo
  - Trade-off entre precisión (MAE) y velocidad
  - Pareto front
  - Selección según prioridades de negocio

- **Ejercicio 3**: Persistencia en base de datos
  - Storage con SQLite
  - Reanudar optimizaciones interrumpidas
  - Compartir estudios entre equipos

- **Ejercicio 4**: Features geoespaciales H3
  - Agregar features espaciales (dist_sucursal_min, densidad_zona, facturacion_zona)
  - Re-optimizar modelo con features ampliadas
  - Feature importance de variables espaciales

**🎯 Objetivos de Aprendizaje**:
- ✅ Entender diferencia entre parámetros e hiperparámetros
- ✅ Optimizar modelos de forma eficiente vs. Grid/Random Search
- ✅ Aplicar pruning para ahorrar tiempo de cómputo
- ✅ Visualizar proceso de optimización
- ✅ Integrar features espaciales H3 en workflows de ML

**📦 Librerías Requeridas**:
```python
optuna==3.5.0
h3==4.0.0b5
scikit-learn
pandas
numpy
plotly
```

**🔗 Conexión con el Curso**:
- Relacionado con **Unidad 3 (Modelado de Datos)** - Feature Engineering
- Usa datasets de la panadería del curso principal
- Integra features geoespaciales H3 de TP04-TP08

---

## 📅 Módulos Planificados

### 02. Validación Cruzada Avanzada
**Temas**: Stratified K-Fold, Time Series CV, Group K-Fold, Leave-One-Out CV, validación espacial

### 03. Selección de Características
**Temas**: Filter/Wrapper/Embedded methods, SHAP para selección, features espaciales

### 04. Interpretabilidad de Modelos
**Temas**: SHAP, LIME, Partial Dependence Plots, Feature Importance, interpretabilidad espacial

### 05. AutoML y Feature Store
**Temas**: H2O.ai, PyCaret, Databricks Feature Store, MLflow integration

### 06. MLflow Tracking y Model Registry
**Temas**: Logging, comparación de runs, Model Registry, deployment, A/B testing

---

## 🚀 Cómo Usar Este Material

### Requisitos Previos
- ✅ Completar **Unidad 1 y 2** del curso principal
- ✅ Familiaridad con Python, Pandas y scikit-learn
- ✅ Haber trabajado con el dataset de la panadería

### Orden Sugerido
1. **Después de Unidad 3**: Módulos 01, 02, 03
2. **Después de TP07-TP08**: Módulos 04, 05, 06

### Recomendaciones
- 📖 **Leer teoría primero** antes de ejercicios prácticos
- 💻 **Ejecutar todas las celdas** para ver resultados
- 🔬 **Experimentar** modificando hiperparámetros
- 📊 **Analizar visualizaciones** para entender procesos
- 🎯 **Completar ejercicios** propuestos en cada notebook

---

## 🎓 Objetivos Generales del Material Complementario

Al completar estos módulos, los estudiantes serán capaces de:

✅ **Optimización Avanzada**:
- Configurar búsquedas de hiperparámetros eficientes
- Aplicar técnicas modernas (Optuna, AutoML)
- Balancear precisión, velocidad y recursos

✅ **Validación Robusta**:
- Seleccionar métodos de validación apropiados
- Evitar data leakage
- Validar modelos temporales y espaciales

✅ **Feature Engineering Inteligente**:
- Seleccionar features automáticamente
- Evaluar importancia de variables
- Incorporar información espacial/temporal

✅ **Interpretabilidad**:
- Explicar predicciones de modelos complejos
- Comunicar resultados a stakeholders no técnicos
- Cumplir requisitos de transparencia

✅ **MLOps y Producción**:
- Tracking de experimentos con MLflow
- Versionamiento de modelos
- Deployment y monitoreo

---

## 📊 Dataset

Este material complementario usa los **mismos datasets** del curso principal:
- `ventas.csv`: Transacciones de panadería (50,000+ registros)
- `clientes.csv`: Información de clientes con H3 geoespacial (500 clientes)
- `productos.csv`: Catálogo de productos (100 productos)
- `sucursales.csv`: Ubicaciones con H3 (3 sucursales)
- `detalles_ventas.csv`: Ítems de cada transacción

**Ubicación**: `/Workspace/Users/cortega@uda.edu.ar/Laboratorio/Datasets/`

---

## 🛠️ Tecnologías Adicionales

Además de las herramientas del curso base, este material introduce:

| Herramienta | Uso | Módulo |
|-------------|-----|--------|
| **Optuna** | Optimización de hiperparámetros | 01 |
| **SHAP** | Interpretabilidad de modelos | 03, 04 |
| **LIME** | Explicaciones locales | 04 |
| **H2O.ai** | AutoML | 05 |
| **PyCaret** | AutoML low-code | 05 |
| **MLflow** | Tracking y registry | 06 |

---

## 📖 Orden de Estudio Recomendado

### Track 1: Machine Learning Avanzado
1. **Módulo 01**: Optimización de Hiperparámetros ← ✅ **EMPEZAR AQUÍ**
2. **Módulo 02**: Validación Cruzada Avanzada
3. **Módulo 03**: Selección de Características
4. **Módulo 05**: AutoML y Feature Store

### Track 2: Interpretabilidad y MLOps
1. **Módulo 04**: Interpretabilidad de Modelos
2. **Módulo 06**: MLflow Tracking y Model Registry

### Track Completo (Recomendado)
Seguir el orden numérico de los módulos (01 → 02 → 03 → 04 → 05 → 06)

---

## 🎯 Conexiones con el Curso Principal

| Módulo Complementario | Unidad del Curso | Tema Relacionado |
|----------------------|------------------|------------------|
| 01 - Optuna | Unidad 3 | Feature Engineering y Modelado |
| 02 - Validación CV | Unidad 3 | Evaluación de Modelos |
| 03 - Selección Features | Unidad 3 | Feature Engineering |
| 04 - Interpretabilidad | Unidad 4 | Proyectos Integradores |
| 05 - AutoML | Unidad 3-4 | Modelado y Pipelines |
| 06 - MLflow | Unidad 4 | Pipelines de Producción |

---

## ✅ Criterios de Éxito

Has completado exitosamente este material complementario cuando puedas:

- [ ] Optimizar hiperparámetros con Optuna en <5 minutos
- [ ] Explicar diferencias entre Grid Search, Random Search y Optuna
- [ ] Aplicar validación cruzada apropiada según tipo de datos
- [ ] Seleccionar features automáticamente con métodos justificados
- [ ] Interpretar predicciones de modelos usando SHAP/LIME
- [ ] Configurar AutoML para benchmarking rápido
- [ ] Trackear experimentos con MLflow
- [ ] Versionar y deployar modelos en producción

---

## 📚 Recursos Adicionales

### Documentación Oficial
- [Optuna](https://optuna.readthedocs.io/)
- [SHAP](https://shap.readthedocs.io/)
- [H2O.ai](https://docs.h2o.ai/)
- [PyCaret](https://pycaret.gitbook.io/)
- [MLflow](https://mlflow.org/docs/latest/)

### Papers y Referencias
- "Optuna: A Next-generation Hyperparameter Optimization Framework" (2019)
- "A Unified Approach to Interpreting Model Predictions" (SHAP paper)
- Kaggle competitions best practices

### Cursos Relacionados
- Databricks Academy: Machine Learning
- Fast.ai: Practical Deep Learning
- Coursera: MLOps Specialization

---

## 💡 Contribuciones

Si tienes ideas para nuevos módulos o mejoras:
1. Contactar a: cortega@uda.edu.ar
2. Proponer temas relacionados con las unidades del curso
3. Sugerir ejercicios prácticos con el dataset de la panadería

---

## 📧 Contacto

**Profesor**: Cristian Dario Ortega Yubro  
**Email**: cortega@uda.edu.ar  
**Universidad**: Universidad del Aconcagua  
**Ubicación**: Mendoza, Argentina  
**GitHub**: [@cristiandarioortegayubro](https://github.com/cristiandarioortegayubro)

---

**🌟 Este material complementario sigue en desarrollo. Nuevos módulos serán agregados periódicamente.**

**Universidad del Aconcagua - Facultad de Ciencias Económicas y Jurídicas**  
**Licenciatura en Analítica de Negocios**  
**Mendoza, Argentina 🇦🇷**
