# Databricks notebook source
# DBTITLE 1,Encabezado TP01
# MAGIC %md
# MAGIC # TP01: Configuración Cloud y Almacenamiento
# MAGIC ## Laboratorio (Herramientas) - Universidad del Aconcagua
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

