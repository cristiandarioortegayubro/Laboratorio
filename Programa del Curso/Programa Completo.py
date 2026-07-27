# Databricks notebook source
# DBTITLE 1,Encabezado del Programa
# MAGIC %md
# MAGIC # LABORATORIO (HERRAMIENTAS)
# MAGIC ## LICENCIATURA EN ANALÍTICA DE NEGOCIOS
# MAGIC ### Universidad del Aconcagua - Facultad de Ciencias Económicas y Jurídicas
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC <img src="https://www.uda.edu.ar/wp-content/uploads/2021/05/logo-uda.png" width="300" />
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 📚 INFORMACIÓN GENERAL
# MAGIC
# MAGIC | Campo | Información |
# MAGIC |-------|-------------|
# MAGIC | **Ciclo Lectivo** | 2026 |
# MAGIC | **Curso** | Segundo Año |
# MAGIC | **Régimen de Cursado** | Semestral (segundo semestre) |
# MAGIC | **Carga Horaria Total** | 96 horas reloj |
# MAGIC | **Carga Horaria Semanal** | 6 horas reloj |
# MAGIC | **Modalidad** | 100% Virtual |
# MAGIC | **Correlatividad para Cursar** | No tiene |
# MAGIC | **Correlatividad para Rendir** | No tiene |
# MAGIC
# MAGIC ## 👨‍🏫 PROFESORES RESPONSABLES
# MAGIC
# MAGIC * **TITULAR**: ORTEGA YUBRO, Cristian Dario (cortega@uda.edu.ar)
# MAGIC * **ADJUNTO**: MACHIN URBAY, Gustavo Raúl

# COMMAND ----------

# DBTITLE 1,Fundamento de la Asignatura
# MAGIC %md
# MAGIC ## 🎯 FUNDAMENTO DE LA ASIGNATURA
# MAGIC
# MAGIC ### ENCUADRE EN EL PLAN DE ESTUDIO
# MAGIC
# MAGIC La asignatura **Laboratorio** se encuentra ubicada en el segundo semestre de la carrera Licenciatura en Analítica de Negocios. Actúa como el **primer contacto inmersivo y fundacional** del estudiante con el ecosistema tecnológico analítico, proporcionando el soporte instrumental y metodológico necesario para el resto de las asignaturas del plan de estudios.
# MAGIC
# MAGIC Es una materia de carácter **eminentemente práctico y tecnológico**, diseñada para que los alumnos adquieran la capacidad de aplicar conocimientos teóricos mediante el uso de herramientas específicas de análisis, visualización y modelado operando directamente **en la nube**.
# MAGIC
# MAGIC Se distancia de los enfoques tradicionales centrados en software de escritorio local, introduciendo al estudiante en un **flujo de trabajo profesional, colaborativo y escalable**.
# MAGIC
# MAGIC ### EJES INSTRUMENTALES
# MAGIC
# MAGIC El desarrollo de la asignatura se articula en torno a tres ejes instrumentales:
# MAGIC
# MAGIC 1. **Entornos Cloud Colaborativos**: Adopción de plataformas de procesamiento mediante notebooks en la nube que eliminan las barreras de hardware local, permitiendo gestionar de manera eficiente la documentación del código y los recursos computacionales.
# MAGIC
# MAGIC 2. **Manipulación y Modelado Programático**: Introducción al procesamiento estructurado de datos y al modelado estadístico inicial mediante el uso de librerías de alto nivel (como Pandas y Scikit-learn).
# MAGIC
# MAGIC 3. **Visualización Directa y Dinámica**: Desarrollo de paneles de control y reportes interactivos orientados a la analítica de negocios, priorizando la eficiencia del código mediante la utilización exclusiva de la librería **Plotly Express**.
# MAGIC
# MAGIC ### RESULTADO ESPERADO
# MAGIC
# MAGIC Al finalizar el cursado, el estudiante habrá consolidado una **base operativa robusta**, siendo capaz de integrar la extracción, limpieza, análisis exploratorio y comunicación visual de los datos en un único ciclo de trabajo integral.
# MAGIC
# MAGIC Esta formación temprana facilita la resolución de problemas empresariales reales basados en datos y prepara al alumno para abordar con éxito los desafíos de complejidad creciente a lo largo de su formación profesional.

# COMMAND ----------

# DBTITLE 1,Objetivos
# MAGIC %md
# MAGIC ## 🎯 OBJETIVOS GENERALES DE LA ASIGNATURA
# MAGIC
# MAGIC 1. Proporcionar la oportunidad de **aplicar conocimientos teóricos** adquiridos a través de ejercicios y proyectos prácticos utilizando herramientas relevantes.
# MAGIC
# MAGIC 2. **Capacitar a los estudiantes** en el uso de herramientas y software específicos para análisis de datos, visualización y modelado.
# MAGIC
# MAGIC 3. Facilitar la **resolución de problemas empresariales reales** mediante el uso de herramientas avanzadas, fomentando el pensamiento crítico y la capacidad de desarrollar soluciones efectivas y basadas en datos.

# COMMAND ----------

# DBTITLE 1,Contenidos del Curso
# MAGIC %md
# MAGIC ## 📚 CONTENIDOS DEL CURSO
# MAGIC
# MAGIC ### CONTENIDOS MÍNIMOS
# MAGIC
# MAGIC * Herramientas en la nube de Análisis de Datos
# MAGIC * Herramientas en la nube de Visualización de Datos
# MAGIC * Herramientas en la nube de Modelado de Datos
# MAGIC * Desarrollo de proyectos integradores
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 📌 UNIDAD 1: HERRAMIENTAS EN LA NUBE DE ANÁLISIS DE DATOS
# MAGIC
# MAGIC #### Fundamentación
# MAGIC La analítica moderna exige entornos que permitan la reproducibilidad y el procesamiento de alto rendimiento. Se busca que el alumno gestione flujos de trabajo en entornos de computación interactiva basados en notebooks compartidos en la nube.
# MAGIC
# MAGIC #### Objetivos de Aprendizaje
# MAGIC * Gestionar flujos de trabajo en entornos de computación interactiva en la nube
# MAGIC * Extraer, limpiar y explorar conjuntos de datos utilizando sistemas de archivos distribuidos
# MAGIC * Implementar estructuras de datos de baja latencia para el análisis
# MAGIC
# MAGIC #### Contenidos Detallados
# MAGIC 1. **Entornos Operativos**: Configuración de espacios de trabajo en la nube (Databricks), aprovisionamiento de micro-clústeres y gestión colaborativa de notebooks
# MAGIC 2. **Gestión de Almacenamiento**: Utilización de sistemas de archivos distribuidos para la carga, lectura y administración de datos
# MAGIC 3. **Computación con NumPy**: Creación de arreglos y operaciones vectorizadas
# MAGIC 4. **Motores de Datos**: Procesamiento, limpieza y transformación de estructuras de datos utilizando Python (Pandas)
# MAGIC
# MAGIC #### Estrategia Metodológica
# MAGIC Clase expositiva con demostraciones en vivo (Live Coding). Desarrollo de actividades virtuales sincrónicas por Google Meet y asincrónicas en la plataforma Moodle.
# MAGIC
# MAGIC #### Bibliografía Recomendada
# MAGIC * McKinney, W. (2022): Python for Data Analysis
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 📊 UNIDAD 2: HERRAMIENTAS EN LA NUBE DE VISUALIZACIÓN DE DATOS
# MAGIC
# MAGIC #### Fundamentación
# MAGIC La curación de datos y la capacidad de comunicar insights mediante visualizaciones interactivas son fundamentales para que el decisor interactúe con la información.
# MAGIC
# MAGIC #### Objetivos de Aprendizaje
# MAGIC * Diseñar activos visuales dinámicos para el monitoreo de negocios
# MAGIC * Desarrollar paneles de control funcionales integrados en el entorno de desarrollo
# MAGIC
# MAGIC #### Contenidos Detallados
# MAGIC 1. **Visualizaciones Nativas**: Generación de gráficos básicos y perfilado de datos integrados en la plataforma cloud
# MAGIC 2. **Interactividad con Plotly**: Creación de gráficos interactivos y dinámicos utilizando de manera exclusiva la librería **Plotly Express**
# MAGIC 3. **Visualización de Negocios**: Construcción de tableros y empaquetado de celdas de resultados para conformar informes ejecutivos
# MAGIC
# MAGIC #### Estrategia Metodológica
# MAGIC Resolución de casos prácticos y talleres de diseño de visualización interactiva en entornos colaborativos.
# MAGIC
# MAGIC #### Bibliografía Recomendada
# MAGIC * Documentación oficial de Plotly Express
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 🤖 UNIDAD 3: HERRAMIENTAS EN LA NUBE DE MODELADO DE DATOS
# MAGIC
# MAGIC #### Fundamentación
# MAGIC La escala de los datos es una variable crítica. El alumno aprenderá a estructurar datos centralizados y a aplicar modelos predictivos aprovechando el cómputo distribuido en la nube.
# MAGIC
# MAGIC #### Objetivos de Aprendizaje
# MAGIC * Realizar análisis agregados complejos en grandes volúmenes de datos
# MAGIC * Diseñar e implementar modelos predictivos para anticipar escenarios de negocio
# MAGIC
# MAGIC #### Contenidos Detallados
# MAGIC 1. **Estructuración de Datos**: Creación de tablas, vistas y gestión de esquemas lógicos
# MAGIC 2. **Agregación de Datos**: Mecánica de agrupamiento, aplicación de funciones personalizadas y transformación de grupos
# MAGIC 3. **Preparación y Modelado**: Ingeniería de características y entrenamiento de modelos estadísticos integrando Scikit-learn
# MAGIC
# MAGIC #### Estrategia Metodológica
# MAGIC Desarrollo de laboratorios guiados en el entorno cloud para fomentar el análisis de escenarios predictivos.
# MAGIC
# MAGIC #### Bibliografía Recomendada
# MAGIC * Documentación técnica de Scikit-learn
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 🛠️ UNIDAD 4: DESARROLLO DE PROYECTOS INTEGRADORES
# MAGIC
# MAGIC #### Fundamentación
# MAGIC El análisis integral es la esencia de la estrategia. Se busca que el alumno ejecute el ciclo de vida completo del dato para resolver un problema de negocio real.
# MAGIC
# MAGIC #### Objetivos de Aprendizaje
# MAGIC * Facilitar la resolución de problemas empresariales reales mediante el uso de herramientas avanzadas
# MAGIC * Fomentar el pensamiento crítico y la capacidad de desarrollar soluciones efectivas y basadas en datos
# MAGIC
# MAGIC #### Contenidos Detallados
# MAGIC 1. **Diseño del Proyecto**: Formulación del caso de estudio y planificación en el entorno cloud
# MAGIC 2. **Pipeline de Datos**: Ejecución práctica de un flujo de ingesta, limpieza y guardado estructurado
# MAGIC 3. **Análisis y Presentación**: Despliegue de los dashboards interactivos y defensa de la documentación técnica generada
# MAGIC
# MAGIC #### Estrategia Metodológica
# MAGIC Análisis de un caso real integrando metodologías de limpieza, visualización y modelado.

# COMMAND ----------

# DBTITLE 1,Plan de Trabajos Prácticos
# MAGIC %md
# MAGIC ## 📝 PLAN DE TRABAJOS PRÁCTICOS
# MAGIC
# MAGIC | N° | Nombre del TP | Unidad | Objetivos, descripción y/o características |
# MAGIC |---|---|---|---|
# MAGIC | **1** | Configuración Cloud y Almacenamiento | 1 | Configurar el espacio de trabajo en la nube y gestionar la carga de un dataset tabular mediante el sistema de archivos distribuido |
# MAGIC | **2** | Manipulación Programática y Exploración | 1 | Transformar estructuras de datos utilizando Pandas y explorar la información procesada integrando celdas SQL |
# MAGIC | **3** | Perfilado de Datos | 2 | Utilizar herramientas de profiling integradas en el notebook para reconocer distribuciones iniciales del dataset |
# MAGIC | **4** | Dashboards | 2 | Diseñar activos visuales interactivos programando y agruparlos en un panel consolidado |
# MAGIC | **5** | Estructuración y Agregación | 3 | Guardar dataframes como tablas estructuradas y aplicar funciones de agregación sobre métricas comerciales clave |
# MAGIC | **6** | Feature Engineering y Modelado | 3 | Estandarizar variables numéricas y entrenar un modelo predictivo |
# MAGIC | **7** | Diseño y Pipeline Integrador | 4 | Formular el caso de estudio empresarial y ejecutar el flujo práctico de ingesta, limpieza y guardado estructurado |
# MAGIC | **8** | Presentación del Proyecto Final | 4 | Resolver el problema analítico presentando un informe ejecutivo apoyado en el dashboard y la documentación técnica |
# MAGIC
# MAGIC ### Dataset de Trabajo: Panadería La Espiga Dorada
# MAGIC
# MAGIC Todos los trabajos prácticos utilizarán un **dataset unificado de ventas de panadería** que contiene:
# MAGIC * Datos de ventas diarias
# MAGIC * Productos (panes, facturas, tortas, bebidas)
# MAGIC * Clientes y sus preferencias
# MAGIC * Información de sucursales
# MAGIC * Datos de stock y producción
# MAGIC * Métricas de rentabilidad
# MAGIC
# MAGIC Este dataset permite desarrollar análisis realistas de un negocio de retail, aplicando todas las técnicas aprendidas en cada unidad.

# COMMAND ----------

# DBTITLE 1,Metodología y Evaluación
# MAGIC %md
# MAGIC ## 🏛️ METODOLOGÍA DE TRABAJO
# MAGIC
# MAGIC ### Modalidad: 100% Virtual
# MAGIC
# MAGIC La Facultad de Ciencias Económicas y Jurídicas, teniendo en cuenta normativas nacionales del sistema universitario y la experiencia adquirida en los últimos años, ha dispuesto para las clases de este ciclo lectivo, un modelo de cursado organizado con un **100% de virtualidad**.
# MAGIC
# MAGIC En el transcurso del semestre, la cátedra llevará adelante las siguientes actividades virtuales:
# MAGIC * Clases sincrónicas por **Google Meet**
# MAGIC * Videos subidos al aula virtual de la plataforma **Moodle**
# MAGIC * Clases de consulta
# MAGIC * Realización y evaluación de Trabajos Prácticos
# MAGIC * Utilización del aula virtual de Moodle: material escrito de la disciplina, cuestionarios, foro, enlaces URL, etc.
# MAGIC
# MAGIC ### Metodología: "Aprender Haciendo" (Learning by Doing)
# MAGIC
# MAGIC La asignatura se describe como un **Laboratorio de herramientas** para el desarrollo de Analítica Avanzada. El dictado de clases no se limita a la exposición teórica, sino que se desarrolla mediante:
# MAGIC
# MAGIC * **Live Coding**: Resolución en tiempo real de problemas de negocio donde el docente modela el proceso de pensamiento y resolución de errores
# MAGIC * **Entornos Colaborativos**: Uso intensivo de **Databricks Free Edition** y otras herramientas, permitiendo que la experimentación con datos ocurra en un entorno basado en la nube, eliminando barreras de hardware
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 📊 SISTEMA DE EVALUACIÓN
# MAGIC
# MAGIC ### Escala de Calificación
# MAGIC
# MAGIC Según la Res. 260/16 de CS, a partir del ciclo lectivo 2017, la nueva escala de calificaciones será ordinal, de calificación numérica, en la que el **mínimo exigible para aprobar** equivaldrá al **SESENTA POR CIENTO (60%)**. Este porcentaje mínimo se traducirá, en la escala numérica, a un **SEIS (6)**.
# MAGIC
# MAGIC | Escala porcentual % | Escala numérica Nota | Resultado |
# MAGIC |---|---|---|
# MAGIC | 0% | 0 | NO APROBADO |
# MAGIC | 1 a 12% | 1 | NO APROBADO |
# MAGIC | 13 a 24% | 2 | NO APROBADO |
# MAGIC | 25 a 35% | 3 | NO APROBADO |
# MAGIC | 36 a 47% | 4 | NO APROBADO |
# MAGIC | 48 a 59% | 5 | NO APROBADO |
# MAGIC | **60 a 64%** | **6** | **APROBADO** |
# MAGIC | 65 a 74% | 7 | APROBADO |
# MAGIC | 75 a 84% | 8 | APROBADO |
# MAGIC | 85 a 94% | 9 | APROBADO |
# MAGIC | 95 a 100% | 10 | APROBADO |
# MAGIC
# MAGIC ### Condiciones de Regularidad
# MAGIC
# MAGIC Las condiciones de regularidad se obtendrán de la siguiente manera:
# MAGIC
# MAGIC * **Alumno regular**: asistencia a por lo menos el **70%** de las clases y aprobación de **dos (2) exámenes parciales**
# MAGIC * **Alumno no regular**: asistencia a por lo menos el **40%** de las clases y aprobación de **un (1) examen parcial**
# MAGIC * **Alumno recursante**: aquel que no alcance los requisitos mencionados anteriormente

# COMMAND ----------

# DBTITLE 1,Bibliografía
# MAGIC %md
# MAGIC ## 📚 BIBLIOGRAFÍA
# MAGIC
# MAGIC ### Bibliografía Obligatoria
# MAGIC
# MAGIC * **McKinney, W. (2022)**. *Python for Data Analysis: Data Wrangling with pandas, NumPy, and Jupyter* (3rd ed.). O'Reilly Media.
# MAGIC
# MAGIC ### Bibliografía Complementaria
# MAGIC
# MAGIC * **Plotly Express Documentation**: Plotly Open Source Graphing Library for Python. [plotly.com/python/plotly-express/](https://plotly.com/python/plotly-express/)
# MAGIC * **Pandas User Guide**: pandas: powerful Python data analysis toolkit. [pandas.pydata.org/docs/user_guide/](https://pandas.pydata.org/docs/user_guide/)
# MAGIC * **Scikit-Learn Documentation**: scikit-learn: machine learning in Python. [scikit-learn.org/stable/documentation.html](https://scikit-learn.org/stable/documentation.html)
# MAGIC
# MAGIC ### Bibliografía en Línea
# MAGIC
# MAGIC * [www.uda.edu.ar/bibliotecadigital](http://www.uda.edu.ar/bibliotecadigital)
# MAGIC * E-MAIL DE BIBLIOTECA: biblioteca@uda.edu.ar

# COMMAND ----------

# DBTITLE 1,Estructura del Repositorio
# MAGIC %md
# MAGIC ## 📁 ESTRUCTURA DEL REPOSITORIO
# MAGIC
# MAGIC ### Organización de Carpetas
# MAGIC
# MAGIC Este curso está organizado en un repositorio Git sincronizado con GitHub. Toda la estructura del curso se encuentra dentro de la carpeta:
# MAGIC
# MAGIC **`/Workspace/Users/cortega@uda.edu.ar/Laboratorio/`**
# MAGIC
# MAGIC ### 📋 Contenidos por Carpeta
# MAGIC
# MAGIC #### 📚 Programa del Curso
# MAGIC Contiene este notebook con toda la información del programa de la asignatura.
# MAGIC
# MAGIC #### 🔹 Unidad 1: Análisis de Datos
# MAGIC * **Teoria/**: Material teórico sobre entornos cloud, NumPy y Pandas
# MAGIC * **Practicas/**: 
# MAGIC   * **TP01 - Configuración Cloud y Almacenamiento**: Configurar el entorno Databricks y gestionar archivos
# MAGIC   * **TP02 - Manipulación Programática y Exploración**: Transformación de datos con Pandas y SQL
# MAGIC
# MAGIC #### 🔹 Unidad 2: Visualización de Datos
# MAGIC * **Teoria/**: Material teórico sobre Plotly Express y visualizaciones
# MAGIC * **Practicas/**:
# MAGIC   * **TP03 - Perfilado de Datos**: Análisis exploratorio y profiling de datasets
# MAGIC   * **TP04 - Dashboards Interactivos**: Creación de paneles interactivos con Plotly
# MAGIC
# MAGIC #### 🔹 Unidad 3: Modelado de Datos
# MAGIC * **Teoria/**: Material teórico sobre agregación y Scikit-learn
# MAGIC * **Practicas/**:
# MAGIC   * **TP05 - Estructuración y Agregación**: Tablas estructuradas y agregaciones
# MAGIC   * **TP06 - Feature Engineering y Modelado**: Ingeniería de características y modelado predictivo
# MAGIC
# MAGIC #### 🔹 Unidad 4: Proyectos Integradores
# MAGIC * **TP07 - Pipeline Integrador**: Pipeline completo de análisis de datos
# MAGIC * **TP08 - Proyecto Final**: Proyecto final integrador
# MAGIC
# MAGIC > **📌 Nota Importante**: Cada trabajo práctico está ubicado dentro de la carpeta `Practicas/` de su unidad correspondiente, facilitando la organización por temas y siguiendo la estructura pedagógica del curso. Los TPs se desarrollan progresivamente a medida que se avanza en cada unidad.
# MAGIC
# MAGIC #### 📂 Datasets
# MAGIC Contiene todos los datasets utilizados en el curso, principalmente:
# MAGIC * **ventas_panaderia.csv**: Dataset principal de ventas de panadería
# MAGIC * **productos.csv**: Catálogo de productos
# MAGIC * **clientes.csv**: Información de clientes
# MAGIC * **sucursales.csv**: Datos de sucursales
# MAGIC
# MAGIC #### 📖 Bibliografía
# MAGIC Material bibliográfico del curso en formato PDF

# COMMAND ----------

