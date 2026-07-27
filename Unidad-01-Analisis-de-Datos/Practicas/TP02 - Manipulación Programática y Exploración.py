# Databricks notebook source
# DBTITLE 1,Encabezado TP02
# MAGIC %md
# MAGIC # TP02: Manipulación Programática y Exploración
# MAGIC ## Laboratorio (Herramientas) - Universidad del Aconcagua
# MAGIC ### Unidad 1: Herramientas en la Nube de Análisis de Datos
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 🎯 Objetivos del Trabajo Práctico
# MAGIC
# MAGIC En este trabajo práctico aprenderemos a:
# MAGIC
# MAGIC 1. **Cargar datasets** desde archivos CSV usando Pandas
# MAGIC 2. **Transformar estructuras de datos** con operaciones de Pandas
# MAGIC 3. **Limpiar y preparar datos** para análisis
# MAGIC 4. **Explorar información** integrando celdas SQL
# MAGIC 5. **Realizar análisis exploratorio** básico (EDA)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 📁 Caso de Estudio: Análisis de Ventas de Panadería
# MAGIC
# MAGIC Continuamos trabajando con los datos de la **Panadería La Espiga Dorada**. En este TP nos enfocaremos en:
# MAGIC
# MAGIC * Cargar y unir múltiples datasets (ventas, productos, clientes)
# MAGIC * Limpiar datos inconsistentes
# MAGIC * Crear nuevas variables derivadas
# MAGIC * Explorar patrones de ventas
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 🕰️ Duración Estimada: 2 horas

# COMMAND ----------

# DBTITLE 1,Parte 1: Carga de Datos
# MAGIC %md
# MAGIC ## Parte 1: Carga de Datos con Pandas
# MAGIC
# MAGIC ### 📂 Lectura de múltiples archivos CSV
# MAGIC
# MAGIC Vamos a cargar todos los datasets de la panadería usando Pandas.

# COMMAND ----------

# DBTITLE 1,Importar librerías y cargar datos
import pandas as pd
import numpy as np
from datetime import datetime

# Ruta base de los datasets
ruta_datos = '/Workspace/Users/cortega@uda.edu.ar/Laboratorio/Datasets/'

# Cargar todos los datasets
print("📂 Cargando datasets...\n")

df_productos = pd.read_csv(ruta_datos + 'productos.csv')
print(f"✅ Productos: {len(df_productos)} registros")

df_sucursales = pd.read_csv(ruta_datos + 'sucursales.csv')
print(f"✅ Sucursales: {len(df_sucursales)} registros")

df_clientes = pd.read_csv(ruta_datos + 'clientes.csv')
print(f"✅ Clientes: {len(df_clientes)} registros")

df_ventas = pd.read_csv(ruta_datos + 'ventas.csv')
print(f"✅ Ventas: {len(df_ventas)} registros")

df_detalles_ventas = pd.read_csv(ruta_datos + 'detalles_ventas.csv')
print(f"✅ Detalles de ventas: {len(df_detalles_ventas)} registros")

print("\n✅ Todos los datasets cargados exitosamente")

# COMMAND ----------

