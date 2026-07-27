# Databricks notebook source
# DBTITLE 1,Encabezado TP01
# MAGIC %md
# MAGIC # TP01: Configuración Cloud y Almacenamiento
# MAGIC ## Laboratorio (Herramientas) - Universidad del Aconcagua
# MAGIC ### Unidad 1: Herramientas en la Nube de Análisis de Datos
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 🎯 Objetivos del Trabajo Práctico
# MAGIC
# MAGIC En este primer trabajo práctico aprenderemos a:
# MAGIC
# MAGIC 1. **Configurar el espacio de trabajo** en Databricks Free Edition
# MAGIC 2. **Gestionar archivos** en el sistema de archivos de Databricks
# MAGIC 3. **Cargar datasets** desde archivos CSV
# MAGIC 4. **Leer y explorar datos** utilizando Python y Pandas
# MAGIC 5. **Realizar operaciones básicas** de inspección de datos
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 📁 Caso de Estudio: Panadería La Espiga Dorada
# MAGIC
# MAGIC Trabajaremos con datos reales de la **Panadería La Espiga Dorada**, un negocio de retail ubicado en Mendoza, Argentina, que opera con 3 sucursales.
# MAGIC
# MAGIC **Datasets disponibles:**
# MAGIC * `productos.csv` - Catálogo de productos
# MAGIC * `sucursales.csv` - Información de sucursales  
# MAGIC * `clientes.csv` - Base de clientes
# MAGIC * `ventas.csv` - Transacciones de ventas
# MAGIC * `detalles_ventas.csv` - Detalle de productos por venta
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 🕰️ Duración Estimada: 2 horas

# COMMAND ----------

# DBTITLE 1,Parte 1: Configuración del Entorno
# MAGIC %md
# MAGIC ## Parte 1: Configuración del Entorno Databricks
# MAGIC
# MAGIC ### ✅ Verificar configuración del Workspace
# MAGIC
# MAGIC Databricks Free Edition ya está configurado y listo para usar. Vamos a verificar:
# MAGIC
# MAGIC 1. **Compute**: Este notebook usa serverless compute (se selecciona automáticamente)
# MAGIC 2. **Workspace**: Tu workspace está en `/Workspace/Users/cortega@uda.edu.ar/`
# MAGIC 3. **Lenguajes soportados**: Python, SQL, sh (R y Scala NO están soportados en serverless)

# COMMAND ----------

# DBTITLE 1,Verificar entorno Python
# Verificar las librerías disponibles
import sys
import pandas as pd
import numpy as np

print("✅ Entorno Python configurado correctamente")
print(f"Python version: {sys.version}")
print(f"Pandas version: {pd.__version__}")
print(f"NumPy version: {np.__version__}")
print(f"\nWorkspace path: /Workspace/Users/cortega@uda.edu.ar/")

# COMMAND ----------

