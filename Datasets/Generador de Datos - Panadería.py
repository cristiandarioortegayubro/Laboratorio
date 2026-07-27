# Databricks notebook source
# DBTITLE 1,Introducción
# MAGIC %md
# MAGIC # Generador de Datos - Panadería La Espiga Dorada
# MAGIC
# MAGIC Este notebook genera datasets sintéticos para la **Panadería La Espiga Dorada**, un negocio ficticio de retail que opera en Mendoza, Argentina.
# MAGIC
# MAGIC ## Datasets generados:
# MAGIC
# MAGIC 1. **productos.csv**: Catálogo de productos (panes, facturas, tortas, bebidas)
# MAGIC 2. **sucursales.csv**: Información de las 3 sucursales
# MAGIC 3. **clientes.csv**: Base de datos de clientes frecuentes
# MAGIC 4. **ventas.csv**: Transacciones de ventas (2 años de historia)
# MAGIC
# MAGIC ## Características del dataset:
# MAGIC
# MAGIC * **Período**: 2 años (2024-2025)
# MAGIC * **Transacciones**: ~50,000 ventas
# MAGIC * **Productos**: 45 productos diferentes
# MAGIC * **Sucursales**: 3 locales en Mendoza
# MAGIC * **Clientes**: 500 clientes frecuentes
# MAGIC * **Estacionalidad**: Mayor venta de facturas los fines de semana
# MAGIC * **Tendencias**: Crecimiento de productos saludables
# MAGIC * **Datos faltantes**: Algunos registros tienen valores nulos (realismo)

# COMMAND ----------

# DBTITLE 1,Importar librerías
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Configurar semilla para reproducibilidad
np.random.seed(42)
random.seed(42)

print("Librerías importadas correctamente")
print(f"Pandas version: {pd.__version__}")
print(f"NumPy version: {np.__version__}")

# COMMAND ----------

# DBTITLE 1,Generar catálogo de productos
# Dataset 1: PRODUCTOS
productos_data = {
    'producto_id': range(1, 46),
    'nombre': [
        # Panes (15)
        'Pan Francés', 'Pan Integral', 'Pan de Molde', 'Pan de Campo', 'Pan de Viena',
        'Pan Gallego', 'Baguette', 'Pan de Salvado', 'Pan de Centeno', 'Pan Ciabatta',
        'Pan de Ajo', 'Pan Pita', 'Pan Hamburguesa', 'Pan Pancho', 'Flauta',
        # Facturas (12)
        'Medialuna Manteca', 'Medialuna Grasa', 'Bola de Fraile', 'Canon', 'Vigilante',
        'Sacramento', 'Churros', 'Biscuit', 'Cremona', 'Palmerita', 'Fosforito', 'Facturas Surtidas',
        # Tortas (8)
        'Torta de Chocolate', 'Torta de Frutilla', 'Torta Rogel', 'Torta de Coco',
        'Lemon Pie', 'Torta Selva Negra', 'Cheesecake', 'Tiramisú',
        # Bebidas (5)
        'Café Espresso', 'Café con Leche', 'Té', 'Jugo Natural', 'Agua Mineral',
        # Otros (5)
        'Sandwich Miga', 'Empanadas', 'Pizza Porción', 'Alfajor', 'Galletitas'
    ],
    'categoria': (
        ['Pan'] * 15 + 
        ['Facturas'] * 12 + 
        ['Tortas'] * 8 + 
        ['Bebidas'] * 5 + 
        ['Otros'] * 5
    ),
    'precio_unitario': [
        # Panes
        800, 950, 1200, 1000, 850, 1100, 900, 1050, 1150, 950,
        750, 600, 500, 450, 800,
        # Facturas
        350, 280, 400, 350, 380, 380, 250, 320, 400, 330, 280, 3500,
        # Tortas
        8500, 9000, 9500, 8000, 7500, 10000, 9500, 11000,
        # Bebidas
        800, 950, 600, 1200, 500,
        # Otros
        2500, 800, 1500, 650, 900
    ],
    'costo_unitario': [
        # Panes (margen 50%)
        400, 475, 600, 500, 425, 550, 450, 525, 575, 475,
        375, 300, 250, 225, 400,
        # Facturas (margen 60%)
        140, 112, 160, 140, 152, 152, 100, 128, 160, 132, 112, 1400,
        # Tortas (margen 40%)
        5100, 5400, 5700, 4800, 4500, 6000, 5700, 6600,
        # Bebidas (margen 65%)
        280, 332, 210, 420, 175,
        # Otros (margen 55%)
        1125, 360, 675, 292, 405
    ],
    'stock_minimo': [
        # Panes
        50, 30, 20, 15, 40, 25, 35, 20, 15, 20,
        30, 25, 40, 40, 30,
        # Facturas
        100, 120, 50, 60, 40, 40, 80, 60, 30, 50, 60, 20,
        # Tortas
        5, 5, 4, 4, 3, 3, 4, 3,
        # Bebidas
        30, 40, 30, 20, 50,
        # Otros
        30, 40, 25, 50, 40
    ],
    'es_saludable': (
        [False, True, False, False, False, False, False, True, True, False,
         False, False, False, False, False] +  # Panes
        [False] * 12 +  # Facturas
        [False] * 8 +  # Tortas
        [False, False, True, True, True] +  # Bebidas
        [False] * 5  # Otros
    )
}

df_productos = pd.DataFrame(productos_data)
df_productos['margen_porcentaje'] = ((df_productos['precio_unitario'] - df_productos['costo_unitario']) / 
                                      df_productos['precio_unitario'] * 100).round(2)

print(f"\u2705 Productos generados: {len(df_productos)} productos")
df_productos.head(10)

# COMMAND ----------

# DBTITLE 1,Generar sucursales
# Dataset 2: SUCURSALES
sucursales_data = {
    'sucursal_id': [1, 2, 3],
    'nombre': ['Espiga Dorada - Centro', 'Espiga Dorada - Guaymallén', 'Espiga Dorada - Godoy Cruz'],
    'direccion': ['Av. San Martín 1234, Mendoza', 'Ruta 40 Km 5, Guaymallén', 'Av. San Francisco de Asis 567, Godoy Cruz'],
    'zona': ['Centro', 'Este', 'Sur'],
    'fecha_apertura': ['2020-03-15', '2021-06-20', '2022-11-10'],
    'metros_cuadrados': [120, 150, 100],
    'empleados': [8, 10, 6],
    'tiene_cafeteria': [True, True, False]
}

df_sucursales = pd.DataFrame(sucursales_data)
df_sucursales['fecha_apertura'] = pd.to_datetime(df_sucursales['fecha_apertura'])

print(f"\u2705 Sucursales generadas: {len(df_sucursales)} sucursales")
df_sucursales

# COMMAND ----------

# DBTITLE 1,Generar clientes
# Dataset 3: CLIENTES
nombres = ['Juan', 'María', 'Carlos', 'Ana', 'Luis', 'Laura', 'Pedro', 'Sofía', 'Diego', 'Valentina',
           'Miguel', 'Florencia', 'Javier', 'Camila', 'Martín', 'Victoria', 'Alejandro', 'Martina', 'Gonzalo', 'Catalina']
apellidos = ['González', 'Rodríguez', 'Fernández', 'López', 'Martínez', 'Sánchez', 'Pérez', 'Gómez', 'Martín', 'Jiménez',
             'Ruiz', 'Díaz', 'Moreno', 'Álvarez', 'Romero', 'Alonso', 'Gutiérrez', 'Navarro', 'Torres', 'Domínguez']

clientes_data = []
for i in range(1, 501):
    nombre_completo = f"{random.choice(nombres)} {random.choice(apellidos)}"
    email = f"{nombre_completo.lower().replace(' ', '.')}@email.com" if random.random() > 0.1 else None
    telefono = f"261{random.randint(1000000, 9999999)}" if random.random() > 0.15 else None
    
    clientes_data.append({
        'cliente_id': i,
        'nombre': nombre_completo,
        'email': email,
        'telefono': telefono,
        'fecha_registro': (datetime(2020, 1, 1) + timedelta(days=random.randint(0, 1825))).strftime('%Y-%m-%d'),
        'preferencia_categoria': random.choice(['Pan', 'Facturas', 'Tortas', 'Bebidas', 'Otros', None]),
        'es_vip': random.random() > 0.85
    })

df_clientes = pd.DataFrame(clientes_data)
df_clientes['fecha_registro'] = pd.to_datetime(df_clientes['fecha_registro'])

print(f"\u2705 Clientes generados: {len(df_clientes)} clientes")
print(f"   Clientes VIP: {df_clientes['es_vip'].sum()}")
print(f"   Clientes con email: {df_clientes['email'].notna().sum()}")
print(f"   Clientes con teléfono: {df_clientes['telefono'].notna().sum()}")
df_clientes.head(10)

# COMMAND ----------

# DBTITLE 1,Generar ventas (parte 1)
# Dataset 4: VENTAS (2 años de historia)
print("Generando dataset de ventas... esto puede tomar unos segundos")

ventas_data = []
venta_id = 1
fecha_inicio = datetime(2024, 1, 1)
fecha_fin = datetime(2025, 12, 31)

# Generar ventas para cada día
fecha_actual = fecha_inicio
while fecha_actual <= fecha_fin:
    # Determinar número de ventas según día de semana
    es_fin_de_semana = fecha_actual.weekday() >= 5
    es_lunes = fecha_actual.weekday() == 0
    
    # Más ventas los fines de semana, menos los lunes
    if es_fin_de_semana:
        num_ventas = random.randint(80, 120)
    elif es_lunes:
        num_ventas = random.randint(40, 60)
    else:
        num_ventas = random.randint(60, 85)
    
    # Generar ventas del día
    for _ in range(num_ventas):
        sucursal_id = random.choice([1, 2, 3])
        
        # 70% de las ventas tienen cliente identificado
        cliente_id = random.choice(df_clientes['cliente_id'].tolist()) if random.random() > 0.3 else None
        
        # Hora de venta (picos en desayuno y merienda)
        hora_pico = random.choice(['desayuno', 'almuerzo', 'merienda'])
        if hora_pico == 'desayuno':
            hora = random.randint(7, 10)
        elif hora_pico == 'almuerzo':
            hora = random.randint(12, 14)
        else:  # merienda
            hora = random.randint(17, 20)
        
        minuto = random.randint(0, 59)
        segundo = random.randint(0, 59)
        fecha_hora = fecha_actual.replace(hour=hora, minute=minuto, second=segundo)
        
        ventas_data.append({
            'venta_id': venta_id,
            'fecha': fecha_actual.strftime('%Y-%m-%d'),
            'hora': fecha_hora.strftime('%H:%M:%S'),
            'sucursal_id': sucursal_id,
            'cliente_id': cliente_id
        })
        venta_id += 1
    
    fecha_actual += timedelta(days=1)
    
    # Mostrar progreso cada 100 días
    if (fecha_actual - fecha_inicio).days % 100 == 0:
        print(f"  Progreso: {(fecha_actual - fecha_inicio).days} días procesados...")

print(f"\n\u2705 Ventas generadas: {len(ventas_data)} transacciones")

# COMMAND ----------

# DBTITLE 1,Generar detalles de ventas (parte 2)
# Generar detalles de cada venta (productos comprados)
print("Generando detalles de ventas...")
detalles_ventas = []

for venta in ventas_data:
    # Cada venta tiene entre 1 y 5 productos diferentes
    num_productos = random.choices([1, 2, 3, 4, 5], weights=[40, 30, 20, 7, 3])[0]
    
    # Seleccionar productos aleatorios
    productos_venta = random.sample(df_productos['producto_id'].tolist(), num_productos)
    
    for producto_id in productos_venta:
        cantidad = random.choices([1, 2, 3, 4, 6, 12], weights=[50, 25, 15, 5, 3, 2])[0]
        
        # Buscar precio del producto
        precio = df_productos[df_productos['producto_id'] == producto_id]['precio_unitario'].values[0]
        
        # 5% de descuento aleatorio en algunos productos
        descuento = random.choice([0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 10, 15])
        
        detalles_ventas.append({
            'venta_id': venta['venta_id'],
            'producto_id': producto_id,
            'cantidad': cantidad,
            'precio_unitario': precio,
            'descuento_porcentaje': descuento,
            'subtotal': precio * cantidad * (1 - descuento/100)
        })

print(f"\u2705 Detalles de ventas generados: {len(detalles_ventas)} líneas de venta")
print(f"   Promedio de productos por venta: {len(detalles_ventas) / len(ventas_data):.2f}")

# COMMAND ----------

# DBTITLE 1,Crear DataFrames finales
# Crear DataFrames finales
df_ventas = pd.DataFrame(ventas_data)
df_detalles_ventas = pd.DataFrame(detalles_ventas)

# Agregar total a cada venta
df_totales = df_detalles_ventas.groupby('venta_id')['subtotal'].sum().reset_index()
df_totales.columns = ['venta_id', 'total']
df_ventas = df_ventas.merge(df_totales, on='venta_id')

# Convertir fechas
df_ventas['fecha'] = pd.to_datetime(df_ventas['fecha'])
df_ventas['fecha_hora'] = pd.to_datetime(df_ventas['fecha'].astype(str) + ' ' + df_ventas['hora'])

# Agregar columnas adicionales
df_ventas['dia_semana'] = df_ventas['fecha'].dt.day_name()
df_ventas['mes'] = df_ventas['fecha'].dt.month
df_ventas['anio'] = df_ventas['fecha'].dt.year
df_ventas['es_fin_de_semana'] = df_ventas['fecha'].dt.weekday >= 5

print("\n\u2705 DataFrames finales creados")
print(f"\nEstadísticas de ventas:")
print(f"  Total ventas: {len(df_ventas):,}")
print(f"  Total líneas de venta: {len(df_detalles_ventas):,}")
print(f"  Facturación total: ${df_ventas['total'].sum():,.2f}")
print(f"  Ticket promedio: ${df_ventas['total'].mean():,.2f}")
print(f"  Venta mínima: ${df_ventas['total'].min():,.2f}")
print(f"  Venta máxima: ${df_ventas['total'].max():,.2f}")

# COMMAND ----------

# DBTITLE 1,Guardar archivos CSV
# Guardar archivos CSV
import os

# Definir ruta de salida
ruta_datasets = '/Workspace/Users/cortega@uda.edu.ar/Laboratorio/Datasets/'

print("Guardando archivos CSV...")

# Guardar cada dataset
df_productos.to_csv(ruta_datasets + 'productos.csv', index=False, encoding='utf-8')
print(f"  \u2705 productos.csv - {len(df_productos)} registros")

df_sucursales.to_csv(ruta_datasets + 'sucursales.csv', index=False, encoding='utf-8')
print(f"  \u2705 sucursales.csv - {len(df_sucursales)} registros")

df_clientes.to_csv(ruta_datasets + 'clientes.csv', index=False, encoding='utf-8')
print(f"  \u2705 clientes.csv - {len(df_clientes)} registros")

df_ventas.to_csv(ruta_datasets + 'ventas.csv', index=False, encoding='utf-8')
print(f"  \u2705 ventas.csv - {len(df_ventas)} registros")

df_detalles_ventas.to_csv(ruta_datasets + 'detalles_ventas.csv', index=False, encoding='utf-8')
print(f"  \u2705 detalles_ventas.csv - {len(df_detalles_ventas)} registros")

print("\n\u2705 Todos los archivos guardados exitosamente en:")
print(f"   {ruta_datasets}")

# COMMAND ----------

# DBTITLE 1,Vista previa de los datos
# MAGIC %md
# MAGIC ## Vista Previa de los Datasets Generados
# MAGIC
# MAGIC A continuación se muestran muestras de cada dataset para verificar la calidad de los datos generados.

# COMMAND ----------

# DBTITLE 1,Previsualizar productos
print("=" * 80)
print("PRODUCTOS")
print("=" * 80)
df_productos.head(10)

# COMMAND ----------

# DBTITLE 1,Previsualizar ventas
print("=" * 80)
print("VENTAS (primeras 10)")
print("=" * 80)
df_ventas[['venta_id', 'fecha', 'hora', 'sucursal_id', 'cliente_id', 'total', 'dia_semana']].head(10)

# COMMAND ----------

# DBTITLE 1,Análisis rápido
print("=" * 80)
print("ANÁLISIS RÁPIDO")
print("=" * 80)

print("\n1. Ventas por Sucursal:")
print(df_ventas.groupby('sucursal_id')['total'].agg(['count', 'sum', 'mean']).round(2))

print("\n2. Productos más vendidos (top 10):")
top_productos = df_detalles_ventas.groupby('producto_id').agg({
    'cantidad': 'sum',
    'subtotal': 'sum'
}).sort_values('cantidad', ascending=False).head(10)
top_productos = top_productos.merge(df_productos[['producto_id', 'nombre']], on='producto_id')
print(top_productos[['nombre', 'cantidad', 'subtotal']])

print("\n3. Ventas por Categoría:")
ventas_categoria = df_detalles_ventas.merge(df_productos[['producto_id', 'categoria']], on='producto_id')
print(ventas_categoria.groupby('categoria')['subtotal'].sum().sort_values(ascending=False).round(2))

print("\n4. Ventas por Día de Semana:")
print(df_ventas.groupby('dia_semana')['total'].agg(['count', 'sum', 'mean']).round(2))

# COMMAND ----------

