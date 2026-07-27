# Databricks notebook source
# DBTITLE 1,10. Casos Prácticos con H3
# MAGIC %md
# MAGIC ## 🎯 Casos Prácticos: Features H3 y Temporales
# MAGIC
# MAGIC ### 🗺️ Caso 1: Interpretar Features H3
# MAGIC
# MAGIC **Problema**: Feature H3 encoded es importante, pero ¿qué zonas contribuyen más?
# MAGIC
# MAGIC **Solución con SHAP**:
# MAGIC
# MAGIC ```python
# MAGIC import shap
# MAGIC import pandas as pd
# MAGIC import h3
# MAGIC
# MAGIC # Entrenar modelo
# MAGIC model = RandomForestRegressor(n_estimators=100, random_state=42)
# MAGIC model.fit(X_train, y_train)
# MAGIC
# MAGIC # SHAP values
# MAGIC explainer = shap.TreeExplainer(model)
# MAGIC shap_values = explainer(X_test)
# MAGIC
# MAGIC # Analizar feature H3
# MAGIC h3_col = 'zona_h3_encoded'
# MAGIC h3_idx = X_test.columns.tolist().index(h3_col)
# MAGIC
# MAGIC # Top zonas por SHAP value promedio
# MAGIC df_h3_shap = pd.DataFrame({
# MAGIC     'h3_encoded': X_test[h3_col],
# MAGIC     'h3_original': X_test['zona_h3'],  # Sin encodear
# MAGIC     'shap_value': shap_values.values[:, h3_idx]
# MAGIC })
# MAGIC
# MAGIC top_zonas = df_h3_shap.groupby('h3_original')['shap_value'].mean().sort_values(ascending=False)
# MAGIC
# MAGIC print("Top 10 zonas H3 por impacto positivo:")
# MAGIC for h3_hex, shap_val in top_zonas.head(10).items():
# MAGIC     lat, lon = h3.h3_to_geo(h3_hex)
# MAGIC     print(f"  {h3_hex}: SHAP = {shap_val:+.2f}  (lat={lat:.4f}, lon={lon:.4f})")
# MAGIC
# MAGIC print("\nTop 10 zonas H3 por impacto negativo:")
# MAGIC for h3_hex, shap_val in top_zonas.tail(10).items():
# MAGIC     lat, lon = h3.h3_to_geo(h3_hex)
# MAGIC     print(f"  {h3_hex}: SHAP = {shap_val:+.2f}  (lat={lat:.4f}, lon={lon:.4f})")
# MAGIC ```
# MAGIC
# MAGIC 💡 **Insight**: Identificar zonas geográficas con mayor impacto en ventas.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 📅 Caso 2: Interpretar Features Temporales
# MAGIC
# MAGIC **Problema**: ¿Cómo interactúan día de semana y mes?
# MAGIC
# MAGIC **Solución con SHAP Dependence Plot**:
# MAGIC
# MAGIC ```python
# MAGIC # Dependence plot con interacción
# MAGIC shap.dependence_plot(
# MAGIC     'dia_semana',
# MAGIC     shap_values.values,
# MAGIC     X_test,
# MAGIC     interaction_index='mes'
# MAGIC )
# MAGIC plt.title('Interacción: Día de Semana x Mes')
# MAGIC plt.show()
# MAGIC ```
# MAGIC
# MAGIC **Insights típicos**:
# MAGIC - Viernes tiene mayor impacto en diciembre (temporada alta)
# MAGIC - Lunes tiene impacto similar todo el año
# MAGIC - Fin de semana es más importante en verano
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 🔄 Caso 3: Explicar Predicción Anómala
# MAGIC
# MAGIC **Problema**: Predicción muy alta (¿por qué?).
# MAGIC
# MAGIC **Solución con SHAP Waterfall**:
# MAGIC
# MAGIC ```python
# MAGIC # Encontrar predicciones altas
# MAGIC predictions = model.predict(X_test)
# MAGIC high_preds = predictions.argsort()[-5:]  # Top 5 predicciones
# MAGIC
# MAGIC # Explicar cada una
# MAGIC for i in high_preds:
# MAGIC     print(f"\nPredicción #{i}: ${predictions[i]:.2f}")
# MAGIC     shap.plots.waterfall(shap_values[i])
# MAGIC ```
# MAGIC
# MAGIC **Interpretación**:
# MAGIC ```
# MAGIC Predicción #542: $280.50
# MAGIC
# MAGIC E[f(X)] = $120.00
# MAGIC   ├─ zona_h3 = microcentro  +$45.00  ⬆️
# MAGIC   ├─ dia_semana = viernes   +$35.00  ⬆️
# MAGIC   ├─ mes = diciembre        +$50.00  ⬆️  (Navidad!)
# MAGIC   ├─ segmento = premium     +$25.00  ⬆️
# MAGIC   └─ cliente_frecuente      +$5.50   ⬆️
# MAGIC f(x) = $280.50
# MAGIC ```
# MAGIC
# MAGIC 💡 **Insight**: Predicción alta debido a **múltiples factores positivos** alineados (zona premium + viernes + diciembre).
# MAGIC
# MAGIC ---

# COMMAND ----------

# DBTITLE 1,11. Mejores Prácticas
# MAGIC %md
# MAGIC ## ✅ Mejores Prácticas de Interpretabilidad
# MAGIC
# MAGIC ### ✅ DO: Buenas Prácticas
# MAGIC
# MAGIC #### 1. **Siempre Usar Test Set para Interpretabilidad**
# MAGIC ```python
# MAGIC # ✅ BIEN: Calcular SHAP en test
# MAGIC shap_values = explainer(X_test)
# MAGIC
# MAGIC # ❌ MAL: Calcular SHAP en train (sobreajustado)
# MAGIC shap_values = explainer(X_train)
# MAGIC ```
# MAGIC
# MAGIC #### 2. **Validar con Múltiples Métodos**
# MAGIC ```python
# MAGIC # ✅ BIEN: Comparar Feature Importance, Permutation y SHAP
# MAGIC fi = model.feature_importances_
# MAGIC pi = permutation_importance(model, X_test, y_test)
# MAGIC shap_global = np.abs(shap_values.values).mean(axis=0)
# MAGIC
# MAGIC # ¿Coinciden? Si no, investigar por qué
# MAGIC ```
# MAGIC
# MAGIC #### 3. **Documentar Explicaciones**
# MAGIC ```python
# MAGIC # ✅ BIEN: Guardar explicaciones
# MAGIC import pickle
# MAGIC
# MAGIC with open('shap_values.pkl', 'wb') as f:
# MAGIC     pickle.dump(shap_values, f)
# MAGIC
# MAGIC # Guardar imágenes
# MAGIC shap.summary_plot(shap_values.values, X_test, show=False)
# MAGIC plt.savefig('shap_summary.png', dpi=300, bbox_inches='tight')
# MAGIC ```
# MAGIC
# MAGIC #### 4. **Explicar Predicciones Críticas**
# MAGIC ```python
# MAGIC # ✅ BIEN: Explicar decisiones importantes
# MAGIC # - Predicciones altas (oportunidades)
# MAGIC # - Predicciones bajas (riesgos)
# MAGIC # - Predicciones anómalas (outliers)
# MAGIC
# MAGIC high_pred_idx = predictions.argsort()[-10:]  # Top 10
# MAGIC for i in high_pred_idx:
# MAGIC     shap.plots.waterfall(shap_values[i])
# MAGIC ```
# MAGIC
# MAGIC #### 5. **Comunicar a Audiencias Diferentes**
# MAGIC ```python
# MAGIC # Para data scientists: SHAP summary plots, dependence plots
# MAGIC # Para negocio: PDP (más simples), waterfall para casos específicos
# MAGIC # Para usuarios finales: Waterfall con lenguaje natural
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### ❌ DON'T: Errores Comunes
# MAGIC
# MAGIC #### 1. **NO Confiar Solo en Feature Importance**
# MAGIC ```python
# MAGIC # ❌ MAL: Solo feature importance (sesgado)
# MAGIC importances = model.feature_importances_
# MAGIC
# MAGIC # ✅ BIEN: Validar con Permutation o SHAP
# MAGIC ```
# MAGIC
# MAGIC #### 2. **NO Interpretar Features Enco ded Sin Contexto**
# MAGIC ```python
# MAGIC # ❌ MAL: "zona_h3_encoded = 42 tiene SHAP = +$20"
# MAGIC # ¿Qué zona es 42?
# MAGIC
# MAGIC # ✅ BIEN: Mapear a valores originales
# MAGIC zone_mapping = dict(zip(df['zona_h3_encoded'], df['zona_h3']))
# MAGIC print(f"Zona {42} = {zone_mapping[42]}")
# MAGIC ```
# MAGIC
# MAGIC #### 3. **NO Asumir Causalidad**
# MAGIC ```python
# MAGIC # ❌ MAL: "Si cambiamos zona H3 → ventas subirán $20"
# MAGIC # Interpretabilidad != Causalidad
# MAGIC
# MAGIC # ✅ BIEN: "Zona H3 está ASOCIADA con +$20 en ventas"
# MAGIC ```
# MAGIC
# MAGIC #### 4. **NO Ignorar Interacciones**
# MAGIC ```python
# MAGIC # ❌ MAL: Analizar features aisladamente
# MAGIC
# MAGIC # ✅ BIEN: Usar SHAP dependence con interaction_index
# MAGIC shap.dependence_plot('dia_semana', shap_values.values, X_test,
# MAGIC                       interaction_index='mes')
# MAGIC ```
# MAGIC
# MAGIC #### 5. **NO Olvidar Incertidumbre**
# MAGIC ```python
# MAGIC # ❌ MAL: "Este feature aporta exactamente +$15.23"
# MAGIC
# MAGIC # ✅ BIEN: "Este feature aporta aproximadamente +$15 (±$2)"
# MAGIC # Reportar intervalos de confianza
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 📊 Checklist de Interpretabilidad
# MAGIC
# MAGIC Antes de presentar un modelo:
# MAGIC
# MAGIC - [ ] Calculaste feature importance (si tree-based)
# MAGIC - [ ] Validaste con permutation importance
# MAGIC - [ ] Generaste SHAP summary plot
# MAGIC - [ ] Analizaste PDP para top 5 features
# MAGIC - [ ] Explicaste al menos 5 predicciones individuales (SHAP waterfall)
# MAGIC - [ ] Investigaste predicciones anómalas
# MAGIC - [ ] Verificaste interacciones con SHAP dependence
# MAGIC - [ ] Documentaste hallazgos clave
# MAGIC - [ ] Preparaste visualizaciones para stakeholders
# MAGIC - [ ] Mapeaste features encoded a valores originales
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 💡 Tips Finales
# MAGIC
# MAGIC 1. ✅ **Empieza simple** (Feature Importance) → profundiza (SHAP)
# MAGIC 2. ✅ **Visual > Números**: Usa plots siempre que sea posible
# MAGIC 3. ✅ **Samplea con datasets grandes**: SHAP en 1000-5000 registros es suficiente
# MAGIC 4. ✅ **Itera**: Interpretabilidad es un proceso, no un pasoúnicos
# MAGIC 5. ✅ **Cuenta historias**: Conecta insights con problemas de negocio
# MAGIC 6. ✅ **Sé honesto**: Si no entiendes algo, invéstigalo
# MAGIC
# MAGIC ---

# COMMAND ----------

# DBTITLE 1,Setup
# Instalar dependencias
%pip install shap lime --quiet

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
import shap
import lime
import lime.lime_tabular

# Configurar visualizaciones
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
shap.initjs()

print("✅ Librerías importadas correctamente")

# COMMAND ----------

# DBTITLE 1,Cargar datos
# Cargar datasets
ruta_datos = '/Workspace/Users/cortega@uda.edu.ar/Laboratorio/Datasets/'

df_ventas = pd.read_csv(ruta_datos + 'ventas.csv')
df_clientes = pd.read_csv(ruta_datos + 'clientes.csv')

print("✅ Datasets cargados:")
print(f"   Ventas: {len(df_ventas):,} registros")
print(f"   Clientes: {len(df_clientes):,} registros")

# Preparar datos
df_ventas['fecha'] = pd.to_datetime(df_ventas['fecha'])
df_ventas = df_ventas.sort_values('fecha').reset_index(drop=True)

# Features temporales
df_ventas['dia_semana'] = df_ventas['fecha'].dt.dayofweek
df_ventas['mes'] = df_ventas['fecha'].dt.month
df_ventas['es_fin_semana'] = df_ventas['dia_semana'].isin([5, 6]).astype(int)

# Merge con clientes
df = df_ventas.merge(df_clientes, on='cliente_id', how='left')
df['segmento_encoded'] = df['segmento'].astype('category').cat.codes

print("\n✅ Features temporales creadas")

# COMMAND ----------

# DBTITLE 1,Ejemplo ejecutable: SHAP completo
print("="*80)
print("EJEMPLO COMPLETO: INTERPRETABILIDAD CON SHAP")
print("="*80)

# Preparar datos
df_ml = df[df['cliente_id'].notna()].copy()

features = ['sucursal_id', 'dia_semana', 'mes', 'es_fin_semana', 'segmento_encoded']
X = df_ml[features]
y = df_ml['total']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\n📊 Dataset: {len(X):,} registros, {len(features)} features")
print(f"   Train: {len(X_train):,}")
print(f"   Test: {len(X_test):,}")

# Entrenar modelo
print("\n1️⃣ Entrenando Random Forest...")
model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)

from sklearn.metrics import mean_absolute_error, r2_score
mae = mean_absolute_error(y_test, model.predict(X_test))
r2 = r2_score(y_test, model.predict(X_test))
print(f"   MAE: ${mae:.2f}")
print(f"   R²: {r2:.3f}")

# Feature Importance
print("\n2️⃣ Feature Importance (RF):")
for feat, imp in sorted(zip(features, model.feature_importances_), key=lambda x: -x[1]):
    print(f"   {feat:20s}: {imp:.4f}")

# SHAP
print("\n3️⃣ Calculando SHAP values...")
explainer = shap.TreeExplainer(model)
shap_values = explainer(X_test[:500])  # Samplear para velocidad

print(f"   SHAP values shape: {shap_values.values.shape}")
print(f"   Base value: ${explainer.expected_value:.2f}")

# Summary plot
print("\n4️⃣ SHAP Summary Plot:")
shap.summary_plot(shap_values.values, X_test[:500], show=False)
plt.tight_layout()
plt.show()

# Waterfall para una predicción
print("\n5️⃣ SHAP Waterfall para una predicción:")
shap.plots.waterfall(shap_values[0])

print(f"\n" + "="*80)
print("✅ Ejemplo completado!")
print("="*80)

# COMMAND ----------

# DBTITLE 1,Conclusiones
# MAGIC %md
# MAGIC ## ✅ Conclusiones
# MAGIC
# MAGIC ### 🎯 Resumen del Módulo
# MAGIC
# MAGIC **Lo que aprendimos**:
# MAGIC
# MAGIC 1. ✅ **Por qué interpretabilidad** (confianza, compliance, debugging, bias)
# MAGIC 2. ✅ **Interpretabilidad global vs. local** (entender modelo vs. predicción)
# MAGIC 3. ✅ **Feature Importance** (rápido pero limitado)
# MAGIC 4. ✅ **Permutation Importance** (más confiable)
# MAGIC 5. ✅ **PDP e ICE** (visualizar efectos de features)
# MAGIC 6. ✅ **SHAP** (método más robusto y completo)
# MAGIC 7. ✅ **LIME** (alternativa local)
# MAGIC 8. ✅ **Comparación** de métodos y cuándo usar cada uno
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 💡 Mensajes Clave
# MAGIC
# MAGIC 1. 🔑 **Interpretabilidad != Causalidad** - Asociación no implica causalidad
# MAGIC 2. 🎯 **SHAP es el gold standard** - Más completo, robusto y rápido (para trees)
# MAGIC 3. ⚠️ **No confiar solo en Feature Importance** - Validar con Permutation/SHAP
# MAGIC 4. 📊 **Visual > Números** - Usar plots para comunicar
# MAGIC 5. 🔄 **Combinar global + local** - Entendimiento completo requiere ambos
# MAGIC 6. 💡 **Samplear con datos grandes** - 1000-5000 registros suficientes para SHAP
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 📦 Guía Rápida
# MAGIC
# MAGIC **¿Qué método usar?**
# MAGIC
# MAGIC - 🚀 **Exploración rápida** → Feature Importance
# MAGIC - ⚖️ **Importancia confiable** → Permutation Importance
# MAGIC - 📊 **Efecto de features** → PDP / ICE
# MAGIC - 🎯 **Explicaciones completas** (trees) → **SHAP TreeExplainer**
# MAGIC - 👤 **Explicaciones individuales** (no-trees) → LIME
# MAGIC - 🔍 **Debugging y validación** → Múltiples métodos
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 📚 Recursos Adicionales
# MAGIC
# MAGIC - [SHAP Documentation](https://shap.readthedocs.io/)
# MAGIC - [Interpretable Machine Learning Book](https://christophm.github.io/interpretable-ml-book/)
# MAGIC - [LIME Paper](https://arxiv.org/abs/1602.04938)
# MAGIC - [SHAP Paper](https://arxiv.org/abs/1705.07874)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🎓 ¡Felicitaciones!
# MAGIC
# MAGIC **Has completado el módulo de Interpretabilidad de Modelos.**
# MAGIC
# MAGIC Ahora puedes:
# MAGIC - ✅ Explicar por qué un modelo hizo una predicción
# MAGIC - ✅ Identificar features más importantes (confiablemente)
# MAGIC - ✅ Visualizar efectos de features con PDP/ICE
# MAGIC - ✅ Usar SHAP para explicaciones globales y locales
# MAGIC - ✅ Comparar métodos y elegir el adecuado
# MAGIC - ✅ Detectar data leakage y bias
# MAGIC - ✅ Comunicar insights a stakeholders
# MAGIC
# MAGIC **Próximo paso**: Notebook Práctico con ejercicios hands-on.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC **Universidad del Aconcagua**  
# MAGIC **Laboratorio (Herramientas)**  
# MAGIC **Mendoza, Argentina**

# COMMAND ----------

# DBTITLE 1,8. LIME
# MAGIC %md
# MAGIC ## 8️⃣ LIME (Local Interpretable Model-agnostic Explanations)
# MAGIC
# MAGIC ### 📖 Concepto
# MAGIC
# MAGIC **LIME** explica **predicciones individuales** aproximando el modelo complejo con un **modelo simple local**.
# MAGIC
# MAGIC **Idea**:
# MAGIC 1. Tomar una predicción a explicar
# MAGIC 2. Generar **datos sintéticos** cerca de esa predicción
# MAGIC 3. Predecir con el modelo original
# MAGIC 4. Entrenar **modelo lineal simple** en esos datos
# MAGIC 5. Usar el modelo lineal para explicar
# MAGIC
# MAGIC **Analogía**: El modelo complejo es un **mapa 3D** complicado. LIME crea un **plano tangente** en un punto específico para entenderlo localmente.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 💻 Instalación
# MAGIC
# MAGIC ```python
# MAGIC %pip install lime
# MAGIC
# MAGIC import lime
# MAGIC import lime.lime_tabular
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 💻 Implementación
# MAGIC
# MAGIC ```python
# MAGIC import lime
# MAGIC import lime.lime_tabular
# MAGIC import numpy as np
# MAGIC
# MAGIC # Entrenar modelo
# MAGIC model = RandomForestRegressor(n_estimators=100, random_state=42)
# MAGIC model.fit(X_train, y_train)
# MAGIC
# MAGIC # Crear explicador LIME
# MAGIC explainer = lime.lime_tabular.LimeTabularExplainer(
# MAGIC     training_data=np.array(X_train),
# MAGIC     feature_names=X_train.columns.tolist(),
# MAGIC     mode='regression',  # 'classification' para clasificación
# MAGIC     random_state=42
# MAGIC )
# MAGIC
# MAGIC # Explicar una predicción
# MAGIC i = 0  # Índice de la predicción a explicar
# MAGIC explanation = explainer.explain_instance(
# MAGIC     data_row=X_test.iloc[i].values,
# MAGIC     predict_fn=model.predict,
# MAGIC     num_features=5  # Top 5 features
# MAGIC )
# MAGIC
# MAGIC # Visualizar
# MAGIC explanation.show_in_notebook()
# MAGIC
# MAGIC # O como lista
# MAGIC print(f"Predicción: ${model.predict(X_test.iloc[[i]])[0]:.2f}")
# MAGIC print("\nContribuciones (LIME):")
# MAGIC for feat, weight in explanation.as_list():
# MAGIC     print(f"  {feat}: {weight:+.2f}")
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 📊 Visualización LIME
# MAGIC
# MAGIC **Output típico**:
# MAGIC ```
# MAGIC Predicción: $152.30
# MAGIC Intercept: $120.00
# MAGIC
# MAGIC Contribuciones:
# MAGIC   dia_semana = 5        +$16.50
# MAGIC   zona_h3 = microcentro +$14.80
# MAGIC   mes = 1               -$3.20
# MAGIC   segmento = VIP        +$4.20
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### ✅ Ventajas de LIME
# MAGIC
# MAGIC 1. ✅ **Model-agnostic**: Funciona con CUALQUIER modelo
# MAGIC 2. ✅ **Intuitivo**: Modelo lineal fácil de entender
# MAGIC 3. ✅ **Flexible**: Funciona con tablas, texto, imágenes
# MAGIC 4. ✅ **Local**: Enfocado en explicar UNA predicción
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### ❌ Desventajas de LIME
# MAGIC
# MAGIC 1. ❌ **Inestable**: Diferentes ejecuciones → diferentes explicaciones
# MAGIC 2. ❌ **Muestreo**: Depende de cómo se generan datos sintéticos
# MAGIC 3. ❌ **Lento**: Más lento que SHAP TreeExplainer
# MAGIC 4. ❌ **Solo local**: No da visión global del modelo
# MAGIC 5. ❌ **No suma**: Contribuciones no suman exactamente a la predicción
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### ⚖️ LIME vs. SHAP
# MAGIC
# MAGIC | Aspecto | LIME | SHAP |
# MAGIC |---------|------|------|
# MAGIC | **Scope** | Solo local | Global + Local |
# MAGIC | **Velocidad** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ (TreeExplainer) |
# MAGIC | **Estabilidad** | ❌ Inestable | ✅ Estable |
# MAGIC | **Sumas** | ❌ No | ✅ Sí |
# MAGIC | **Teoría** | Heurística | Shapley values |
# MAGIC | **Visualizaciones** | Básicas | ✅ Ricas |
# MAGIC
# MAGIC **Conclusión**: 🎯 **SHAP es generalmente mejor** (más robusto, rápido para trees, mejor fundamentado).
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 📊 Cuándo Usar LIME
# MAGIC
# MAGIC ✅ **Usar cuando**:
# MAGIC - Modelo no es tree-based (ej: SVM, redes neuronales) Y SHAP KernelExplainer es muy lento
# MAGIC - Necesitas explicar **texto o imágenes** (LIME tiene soporte especial)
# MAGIC - Ya tienes pipeline con LIME
# MAGIC
# MAGIC ❌ **No usar cuando**:
# MAGIC - Tienes Random Forest/XGBoost → Usar SHAP TreeExplainer
# MAGIC - Necesitas explicaciones **globales** → Usar SHAP
# MAGIC - Necesitas **estabilidad** → Usar SHAP
# MAGIC
# MAGIC ---

# COMMAND ----------

# DBTITLE 1,9. Comparación de Métodos
# MAGIC %md
# MAGIC ## 9️⃣ Comparación Completa de Métodos
# MAGIC
# MAGIC ### 📊 Tabla Comparativa
# MAGIC
# MAGIC | Método | Scope | Model-Agnostic | Velocidad | Estabilidad | Visualizaciones | Mejor Para |
# MAGIC |---------|-------|----------------|-----------|-------------|-----------------|------------|
# MAGIC | **Feature Importance** | Global | ❌ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | Exploración rápida (trees) |
# MAGIC | **Permutation Importance** | Global | ✅ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | Importancia confiable |
# MAGIC | **PDP** | Global | ✅ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Efecto de 1 feature |
# MAGIC | **ICE** | Local | ✅ | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | Heterogeneidad |
# MAGIC | **SHAP** | Ambos | ✅ | ⭐⭐⭐⭐⭐* | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | **TODO** (más completo) |
# MAGIC | **LIME** | Local | ✅ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐ | Texto, imágenes |
# MAGIC
# MAGIC \* SHAP TreeExplainer es muy rápido; KernelExplainer es lento.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### ⚖️ Trade-offs
# MAGIC
# MAGIC **Velocidad vs. Completitud**:
# MAGIC ```
# MAGIC Feature Importance (rápido, básico)
# MAGIC   ↓
# MAGIC Permutation Importance
# MAGIC   ↓
# MAGIC PDP / ICE
# MAGIC   ↓
# MAGIC SHAP (completo, más lento)
# MAGIC   ↓
# MAGIC LIME (lento, inestable)
# MAGIC ```
# MAGIC
# MAGIC **Global vs. Local**:
# MAGIC ```
# MAGIC Global                 Ambos              Local
# MAGIC   │                    │                  │
# MAGIC Feature Imp.          SHAP              LIME
# MAGIC Permutation                             ICE
# MAGIC PDP
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 🧩 Árbol de Decisión: ¿Qué Método Usar?
# MAGIC
# MAGIC ```
# MAGIC ¿Tienes modelo tree-based (RF, XGBoost)?
# MAGIC │
# MAGIC ├── SÍ → ¿Qué necesitas?
# MAGIC │   ├── Exploración rápida → Feature Importance
# MAGIC │   ├── Importancia confiable → Permutation Importance
# MAGIC │   ├── Efecto de features → PDP / ICE
# MAGIC │   └── Explicaciones completas → SHAP TreeExplainer 🎯
# MAGIC │
# MAGIC └── NO → ¿Qué necesitas?
# MAGIC     ├── Importancia global → Permutation Importance
# MAGIC     ├── Efecto de features → PDP / ICE
# MAGIC     ├── Explicaciones locales → SHAP KernelExplainer o LIME
# MAGIC     └── Texto/Imágenes → LIME
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 📊 Workflow Recomendado
# MAGIC
# MAGIC **Fase 1: Exploración Rápida** (⏱️ 5 minutos)
# MAGIC 1. Feature Importance (si tree-based)
# MAGIC 2. Permutation Importance
# MAGIC
# MAGIC **Fase 2: Análisis Global** (⏱️ 30 minutos)
# MAGIC 3. PDP para top 5 features
# MAGIC 4. SHAP Summary Plot
# MAGIC 5. SHAP Dependence Plots para interacciones
# MAGIC
# MAGIC **Fase 3: Explicaciones Locales** (⏱️ 15 minutos por caso)
# MAGIC 6. SHAP Waterfall para casos específicos
# MAGIC 7. SHAP Force plots para comparar predicciones
# MAGIC
# MAGIC **Fase 4: Presentación** (⏱️ 1 hora)
# MAGIC 8. Preparar visualizaciones SHAP
# MAGIC 9. PDP para stakeholders
# MAGIC 10. Waterfall para casos de negocio
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 💡 Recomendación General
# MAGIC
# MAGIC 🎯 **Para la mayoría de casos con Random Forest/XGBoost**:
# MAGIC
# MAGIC 1. ✅ **Usa SHAP** como método principal (TreeExplainer es rápido)
# MAGIC 2. ✅ **Complementa con PDP** para visualizaciones simples
# MAGIC 3. ✅ **Permutation Importance** para validar feature importance
# MAGIC
# MAGIC 🎯 **Para modelos no tree-based**:
# MAGIC
# MAGIC 1. ✅ **Permutation Importance** para importancia global
# MAGIC 2. ✅ **PDP / ICE** para efectos de features
# MAGIC 3. ✅ **SHAP KernelExplainer** si tiempo lo permite (lento)
# MAGIC 4. ✅ **LIME** como alternativa más rápida
# MAGIC
# MAGIC ---

# COMMAND ----------

# DBTITLE 1,7. SHAP Values - Introducción
# MAGIC %md
# MAGIC ## 7️⃣ SHAP Values (SHapley Additive exPlanations)
# MAGIC
# MAGIC ### 🎯 ¿Qué es SHAP?
# MAGIC
# MAGIC **SHAP** es el método de interpretabilidad **más robusto y completo** actualmente.
# MAGIC
# MAGIC **Ventajas**:
# MAGIC 1. ✅ **Global Y local**: Explica modelo completo + predicciones individuales
# MAGIC 2. ✅ **Model-agnostic**: Funciona con cualquier modelo
# MAGIC 3. ✅ **Fundamentado teóricamente**: Basado en Shapley values (teoría de juegos)
# MAGIC 4. ✅ **Propiedades deseables**: Consistencia, simetría, dummy
# MAGIC
# MAGIC **Pregunta que responde**:
# MAGIC > "Para esta predicción, ¿cuánto contribuyó cada feature?"
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 📚 Shapley Values: Intuición
# MAGIC
# MAGIC **Analogía**: Repartir crédito en un equipo.
# MAGIC
# MAGIC **Problema**: Tu equipo de fútbol ganó 3-1. ¿Cuánto contribuyó cada jugador?
# MAGIC
# MAGIC **Shapley Value**: Valor promedio de un jugador **considerando todas las coaliciones posibles**.
# MAGIC
# MAGIC **Ejemplo ML**:
# MAGIC ```python
# MAGIC # Predicción: $150
# MAGIC # Baseline (sin features): $100
# MAGIC # Diferencia a explicar: +$50
# MAGIC
# MAGIC # SHAP reparte los $50 entre features:
# MAGIC dia_viernes:    +$20  (40% del crédito)
# MAGIC zona_h3_centro: +$15  (30%)
# MAGIC cliente_VIP:    +$10  (20%)
# MAGIC mes_diciembre:  +$5   (10%)
# MAGIC -----------------
# MAGIC Total:          +$50  (100%)
# MAGIC ```
# MAGIC
# MAGIC 🔑 **Propiedad clave**: SHAP values **suman** al total.
# MAGIC
# MAGIC ```
# MAGIC Predicción = Baseline + Σ(SHAP values)
# MAGIC $150 = $100 + ($20 + $15 + $10 + $5)
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 📊 SHAP Value Formula
# MAGIC
# MAGIC **Definición matemática** (no te asustes, la librería lo calcula):
# MAGIC
# MAGIC ```
# MAGIC SHAPᵢ = Σ |S|! (N - |S| - 1)! / N! [f(S ∪ {i}) - f(S)]
# MAGIC       S
# MAGIC ```
# MAGIC
# MAGIC Donde:
# MAGIC - S = subconjunto de features
# MAGIC - i = feature de interés
# MAGIC - N = total de features
# MAGIC - f(S) = predicción con features en S
# MAGIC
# MAGIC 💡 **En palabras**: Promedio de **contribución marginal** de la feature sobre **todas las posibles coaliciones**.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 💻 Instalación y Setup
# MAGIC
# MAGIC ```python
# MAGIC # Instalar SHAP
# MAGIC %pip install shap
# MAGIC
# MAGIC import shap
# MAGIC import matplotlib.pyplot as plt
# MAGIC
# MAGIC # Inicializar visualizaciones de SHAP
# MAGIC shap.initjs()
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 🔧 Tipos de Explicadores SHAP
# MAGIC
# MAGIC SHAP tiene **diferentes explicadores** según el tipo de modelo:
# MAGIC
# MAGIC | Explicador | Modelos | Velocidad | Precisión |
# MAGIC |------------|---------|-----------|------------|
# MAGIC | **TreeExplainer** | RF, XGBoost, LightGBM | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
# MAGIC | **LinearExplainer** | Regresión Lineal, Logística | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
# MAGIC | **KernelExplainer** | Cualquier modelo | ⭐ | ⭐⭐⭐⭐ |
# MAGIC | **DeepExplainer** | Redes Neuronales (TensorFlow/PyTorch) | ⭐⭐⭐ | ⭐⭐⭐⭐ |
# MAGIC
# MAGIC 💡 **Recomendación**: Usar **TreeExplainer** para RF/XGBoost (rápido y exacto).
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 💻 Implementación Básica
# MAGIC
# MAGIC ```python
# MAGIC import shap
# MAGIC from sklearn.ensemble import RandomForestRegressor
# MAGIC
# MAGIC # Entrenar modelo
# MAGIC model = RandomForestRegressor(n_estimators=100, random_state=42)
# MAGIC model.fit(X_train, y_train)
# MAGIC
# MAGIC # Crear explicador SHAP
# MAGIC explainer = shap.TreeExplainer(model)
# MAGIC
# MAGIC # Calcular SHAP values
# MAGIC shap_values = explainer(X_test)
# MAGIC
# MAGIC print(f"SHAP values shape: {shap_values.values.shape}")
# MAGIC print(f"Base value: {explainer.expected_value:.2f}")
# MAGIC ```
# MAGIC
# MAGIC **Estructura de shap_values**:
# MAGIC ```python
# MAGIC shap_values.values       # Array (N_samples, N_features)
# MAGIC shap_values.base_values  # Predicción baseline
# MAGIC shap_values.data         # Valores originales de features
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 📊 SHAP Value para Una Predicción
# MAGIC
# MAGIC ```python
# MAGIC # Seleccionar una predicción
# MAGIC i = 0  # Primera predicción de test
# MAGIC
# MAGIC # SHAP values para esa predicción
# MAGIC shap_sample = shap_values[i]
# MAGIC
# MAGIC print(f"Predicción: ${model.predict(X_test.iloc[[i]])[0]:.2f}")
# MAGIC print(f"Baseline: ${shap_sample.base_values:.2f}")
# MAGIC print(f"\nContribuciones por feature:")
# MAGIC
# MAGIC for feat, val in zip(X_test.columns, shap_sample.values):
# MAGIC     sign = '+' if val >= 0 else ''
# MAGIC     print(f"  {feat:20s}: {sign}${val:.2f}")
# MAGIC
# MAGIC print(f"\nSuma: ${shap_sample.base_values + shap_sample.values.sum():.2f}")
# MAGIC ```
# MAGIC
# MAGIC **Salida**:
# MAGIC ```
# MAGIC Predicción: $152.30
# MAGIC Baseline: $120.00
# MAGIC
# MAGIC Contribuciones por feature:
# MAGIC   dia_semana          : +$15.20
# MAGIC   zona_h3_encoded     : +$18.50
# MAGIC   mes                 : -$2.40
# MAGIC   segmento_encoded    : +$1.00
# MAGIC
# MAGIC Suma: $152.30  # ✅ Suma exactamente!
# MAGIC ```
# MAGIC
# MAGIC ---

# COMMAND ----------

# DBTITLE 1,7. SHAP - Visualizaciones
# MAGIC %md
# MAGIC ## 7️⃣ SHAP Values - Visualizaciones
# MAGIC
# MAGIC ### 📊 1. Waterfall Plot (Explicación Local)
# MAGIC
# MAGIC **Uso**: Explicar **UNA predicción específica**.
# MAGIC
# MAGIC **Visualiza**: Cómo cada feature mueve la predicción desde baseline hasta el valor final.
# MAGIC
# MAGIC ```python
# MAGIC import shap
# MAGIC
# MAGIC # Waterfall para una predicción
# MAGIC shap.plots.waterfall(shap_values[0])
# MAGIC ```
# MAGIC
# MAGIC **Interpretación**:
# MAGIC ```
# MAGIC E[f(X)] = $120.00  (baseline)
# MAGIC   │
# MAGIC   ├─ dia_semana=5         +$15.20  ⬆️
# MAGIC   ├─ zona_h3=microcentro  +$18.50  ⬆️
# MAGIC   ├─ mes=1               -$2.40   ⬇️
# MAGIC   └─ segmento=VIP        +$1.00   ⬆️
# MAGIC   │
# MAGIC f(x) = $152.30  (predicción)
# MAGIC ```
# MAGIC
# MAGIC 💡 **Uso**: Explicar predicciones a usuarios finales.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 📊 2. Force Plot (Explicación Local)
# MAGIC
# MAGIC **Similar a waterfall** pero visualización horizontal.
# MAGIC
# MAGIC ```python
# MAGIC # Force plot para una predicción
# MAGIC shap.plots.force(shap_values[0])
# MAGIC ```
# MAGIC
# MAGIC **Force plot para múltiples predicciones**:
# MAGIC ```python
# MAGIC # Apilar múltiples force plots
# MAGIC shap.plots.force(shap_values[:100])  # Primeras 100 predicciones
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 📊 3. Summary Plot (Interpretabilidad Global)
# MAGIC
# MAGIC **Uso**: Visualizar **importancia global** + **distribución** de SHAP values.
# MAGIC
# MAGIC ```python
# MAGIC # Summary plot (beeswarm)
# MAGIC shap.summary_plot(shap_values.values, X_test)
# MAGIC ```
# MAGIC
# MAGIC **Interpretación**:
# MAGIC - **Eje Y**: Features ordenadas por importancia
# MAGIC - **Eje X**: SHAP value (impacto en predicción)
# MAGIC - **Color**: Valor de la feature (rojo=alto, azul=bajo)
# MAGIC - **Puntos**: Cada punto es una predicción
# MAGIC
# MAGIC **Ejemplo**:
# MAGIC ```
# MAGIC Feature         SHAP value
# MAGIC             -20  -10   0  +10  +20
# MAGIC zona_h3     ●●●●|●●●●●●  (rojo a derecha → zonas buenas aumentan ventas)
# MAGIC dia_semana  ●●●|●●●●  (disperso → efecto depende del día)
# MAGIC mes         ●●|●●●  (centrado → efecto variable)
# MAGIC ```
# MAGIC
# MAGIC 💡 **Insights**:
# MAGIC - Features en **top** son más importantes
# MAGIC - **Dispersión horizontal** → efecto variable
# MAGIC - **Color** muestra dirección del efecto
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 📊 4. Bar Plot (Feature Importance Global)
# MAGIC
# MAGIC **Uso**: Importancia global (similar a feature importance).
# MAGIC
# MAGIC ```python
# MAGIC # Bar plot (importancia promedio absoluta)
# MAGIC shap.summary_plot(shap_values.values, X_test, plot_type='bar')
# MAGIC ```
# MAGIC
# MAGIC **Interpretación**: Muestra **|SHAP value| promedio** por feature.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 📊 5. Dependence Plot (Relación Feature-Predicción)
# MAGIC
# MAGIC **Uso**: Ver cómo **una feature** afecta predicciones (similar a PDP pero con SHAP).
# MAGIC
# MAGIC ```python
# MAGIC # Dependence plot para una feature
# MAGIC shap.dependence_plot(
# MAGIC     'dia_semana',  # Feature principal
# MAGIC     shap_values.values,
# MAGIC     X_test,
# MAGIC     interaction_index='mes'  # Color por interacción con otra feature
# MAGIC )
# MAGIC ```
# MAGIC
# MAGIC **Interpretación**:
# MAGIC - **Eje X**: Valor de feature
# MAGIC - **Eje Y**: SHAP value (impacto)
# MAGIC - **Color**: Interacción con otra feature
# MAGIC
# MAGIC **Ejemplo**: 
# MAGIC ```
# MAGIC SHAP value
# MAGIC    │
# MAGIC +20│      🔴🔴  (viernes + diciembre)
# MAGIC    │    🔵
# MAGIC +10│  🔵    🔵
# MAGIC    │🔵      🔴
# MAGIC   0│🔴    🔵
# MAGIC    │
# MAGIC  -10│🔵
# MAGIC    └──────────────> dia_semana
# MAGIC     Lun       Vie      Dom
# MAGIC     🔵=verano 🔴=invierno
# MAGIC ```
# MAGIC
# MAGIC 💡 **Insight**: Viernes tiene mayor impacto en invierno que en verano.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 📊 6. Decision Plot (Trayectoria de Predicción)
# MAGIC
# MAGIC **Uso**: Ver **cómo se acumula** la predicción feature por feature.
# MAGIC
# MAGIC ```python
# MAGIC # Decision plot
# MAGIC shap.decision_plot(
# MAGIC     explainer.expected_value,
# MAGIC     shap_values.values[:10],  # Primeras 10 predicciones
# MAGIC     X_test.iloc[:10]
# MAGIC )
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### ⚖️ Qué Visualización Usar
# MAGIC
# MAGIC | Objetivo | Visualización |
# MAGIC |----------|----------------|
# MAGIC | **Explicar 1 predicción** | Waterfall o Force |
# MAGIC | **Importancia global** | Summary (beeswarm) o Bar |
# MAGIC | **Efecto de 1 feature** | Dependence Plot |
# MAGIC | **Comparar predicciones** | Force (múltiples) o Decision |
# MAGIC | **Interacciones** | Dependence (con color) |
# MAGIC
# MAGIC ---

# COMMAND ----------

# DBTITLE 1,5. Partial Dependence Plots
# MAGIC %md
# MAGIC ## 5️⃣ Partial Dependence Plots (PDP)
# MAGIC
# MAGIC ### 📖 Concepto
# MAGIC
# MAGIC **PDP** muestra **cómo afecta una feature** a las predicciones del modelo, **promediando** sobre todas las demás features.
# MAGIC
# MAGIC **Pregunta**: ¿Cómo cambia la predicción cuando cambio esta feature, **manteniendo todo lo demás constante**?
# MAGIC
# MAGIC **Proceso**:
# MAGIC 1. Seleccionar feature de interés (ej: `dia_semana`)
# MAGIC 2. Para cada valor posible de `dia_semana` (0-6):
# MAGIC    - Crear dataset donde TODOS los registros tienen `dia_semana = 0`
# MAGIC    - Predecir y promediar
# MAGIC    - Crear dataset donde TODOS los registros tienen `dia_semana = 1`
# MAGIC    - Predecir y promediar
# MAGIC    - ... repetir para 2, 3, 4, 5, 6
# MAGIC 3. Graficar: `dia_semana` vs. predicción promedio
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 💻 Implementación
# MAGIC
# MAGIC ```python
# MAGIC from sklearn.inspection import PartialDependenceDisplay
# MAGIC from sklearn.ensemble import RandomForestRegressor
# MAGIC import matplotlib.pyplot as plt
# MAGIC
# MAGIC # Entrenar modelo
# MAGIC model = RandomForestRegressor(n_estimators=100, random_state=42)
# MAGIC model.fit(X_train, y_train)
# MAGIC
# MAGIC # PDP para una feature
# MAGIC fig, ax = plt.subplots(figsize=(10, 6))
# MAGIC PartialDependenceDisplay.from_estimator(
# MAGIC     model, X_train,
# MAGIC     features=['dia_semana'],  # Feature de interés
# MAGIC     ax=ax
# MAGIC )
# MAGIC plt.title('PDP: Efecto de Día de la Semana en Ventas')
# MAGIC plt.show()
# MAGIC ```
# MAGIC
# MAGIC **PDP para múltiples features**:
# MAGIC ```python
# MAGIC # PDP para top 4 features
# MAGIC fig, ax = plt.subplots(figsize=(14, 10))
# MAGIC PartialDependenceDisplay.from_estimator(
# MAGIC     model, X_train,
# MAGIC     features=['dia_semana', 'mes', 'zona_h3_encoded', 'segmento_encoded'],
# MAGIC     ax=ax
# MAGIC )
# MAGIC plt.tight_layout()
# MAGIC plt.show()
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 📊 Interpretación de PDP
# MAGIC
# MAGIC **Ejemplo**: PDP de `dia_semana` para ventas
# MAGIC
# MAGIC ```
# MAGIC Predicción
# MAGIC    $
# MAGIC    │
# MAGIC 150│             ┌───┐
# MAGIC    │             │   │
# MAGIC 140│         ┌───┤   ├──┐
# MAGIC    │     ┌───┤   │   │  │
# MAGIC 130│     │   │   │   │  │
# MAGIC    │─────┤   │   │   │  ├─────
# MAGIC 120│     │   └───┘   └──┘
# MAGIC    └───────────────────────────> Día
# MAGIC     Lun Mar Mie Jue Vie Sab Dom
# MAGIC ```
# MAGIC
# MAGIC 💡 **Insights**:
# MAGIC - Lunes: ventas bajas ($120)
# MAGIC - Viernes: ventas altas ($150)
# MAGIC - Fin de semana: caída ($140)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### ✅ Ventajas
# MAGIC
# MAGIC 1. ✅ **Visual e intuitivo**: Fácil de explicar
# MAGIC 2. ✅ **Model-agnostic**: Funciona con cualquier modelo
# MAGIC 3. ✅ **Muestra relación**: Lineal, no lineal, monotonicity
# MAGIC 4. ✅ **Causal (parcial)**: Efecto de cambiar UNA feature
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### ❌ Desventajas
# MAGIC
# MAGIC #### 1. **Asume Independencia de Features**
# MAGIC
# MAGIC ⚠️ **Problema**: PDP crea combinaciones **imposibles** de features.
# MAGIC
# MAGIC **Ejemplo**:
# MAGIC ```python
# MAGIC # Dataset:
# MAGIC # - Zona H3 A → siempre sucursal 1
# MAGIC # - Zona H3 B → siempre sucursal 2
# MAGIC
# MAGIC # PDP de zona H3:
# MAGIC # Crea combinaciones imposibles:
# MAGIC # - Zona H3 A + sucursal 2  → Nunca ocurre en realidad!
# MAGIC ```
# MAGIC
# MAGIC 💡 **Solución**: Usar **ICE plots** (Individual Conditional Expectation).
# MAGIC
# MAGIC #### 2. **Oculta Heterogeneidad**
# MAGIC
# MAGIC ```python
# MAGIC # PDP muestra promedio:
# MAGIC # - 50% clientes: ventas suben con precio
# MAGIC # - 50% clientes: ventas bajan con precio
# MAGIC # PDP promedio: Sin efecto (falso!)
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 🔄 PDP de 2 Features (2D)
# MAGIC
# MAGIC **PDP 2D** muestra **interacción** entre dos features.
# MAGIC
# MAGIC ```python
# MAGIC from sklearn.inspection import PartialDependenceDisplay
# MAGIC
# MAGIC fig, ax = plt.subplots(figsize=(10, 8))
# MAGIC PartialDependenceDisplay.from_estimator(
# MAGIC     model, X_train,
# MAGIC     features=[('dia_semana', 'mes')],  # Tuple para 2D
# MAGIC     ax=ax
# MAGIC )
# MAGIC plt.title('PDP 2D: Interacción Día-Mes')
# MAGIC plt.show()
# MAGIC ```
# MAGIC
# MAGIC **Interpretación**: Heatmap muestra predicción para cada combinación.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 📊 Cuándo Usar PDP
# MAGIC
# MAGIC ✅ **Usar cuando**:
# MAGIC - Quieres entender **efecto global** de una feature
# MAGIC - Presentar a stakeholders (visual)
# MAGIC - Detectar **relaciones no lineales**
# MAGIC - Comparar modelos
# MAGIC
# MAGIC ❌ **No usar cuando**:
# MAGIC - Features están **muy correlacionadas** (usar ICE)
# MAGIC - Quieres explicar **predicciones individuales** (usar SHAP/LIME)
# MAGIC
# MAGIC ---

# COMMAND ----------

# DBTITLE 1,6. ICE Plots
# MAGIC %md
# MAGIC ## 6️⃣ Individual Conditional Expectation (ICE)
# MAGIC
# MAGIC ### 📖 Concepto
# MAGIC
# MAGIC **ICE plots** son como PDP pero **sin promediar** → muestran una línea **por cada registro**.
# MAGIC
# MAGIC **Diferencia con PDP**:
# MAGIC - **PDP**: Una línea (promedio de todos los registros)
# MAGIC - **ICE**: N líneas (una por registro)
# MAGIC
# MAGIC **Ventaja**: Revela **heterogeneidad** (diferentes efectos en diferentes registros).
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 💻 Implementación
# MAGIC
# MAGIC ```python
# MAGIC from sklearn.inspection import PartialDependenceDisplay
# MAGIC import matplotlib.pyplot as plt
# MAGIC
# MAGIC # ICE plot
# MAGIC fig, ax = plt.subplots(figsize=(10, 6))
# MAGIC PartialDependenceDisplay.from_estimator(
# MAGIC     model, X_train,
# MAGIC     features=['dia_semana'],
# MAGIC     kind='individual',  # 🔑 ICE en lugar de PDP
# MAGIC     ax=ax
# MAGIC )
# MAGIC plt.title('ICE: Efecto de Día de Semana (por registro)')
# MAGIC plt.show()
# MAGIC ```
# MAGIC
# MAGIC **ICE + PDP juntos**:
# MAGIC ```python
# MAGIC fig, ax = plt.subplots(figsize=(10, 6))
# MAGIC PartialDependenceDisplay.from_estimator(
# MAGIC     model, X_train,
# MAGIC     features=['dia_semana'],
# MAGIC     kind='both',  # 🔑 ICE (líneas) + PDP (promedio)
# MAGIC     ax=ax
# MAGIC )
# MAGIC plt.title('ICE + PDP: Día de Semana')
# MAGIC plt.show()
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 📊 Interpretación de ICE
# MAGIC
# MAGIC **Ejemplo 1: Efecto Homogéneo**
# MAGIC ```
# MAGIC Predicción
# MAGIC    $
# MAGIC    │      Todas las líneas
# MAGIC 150│      van en la misma
# MAGIC    │      dirección →
# MAGIC 140│     ┌─────────┐
# MAGIC    │    /          \
# MAGIC 130│   /            \
# MAGIC    │  /              \
# MAGIC 120│ /                \
# MAGIC    └────────────────────> Día
# MAGIC     Lun       Vie      Dom
# MAGIC ```
# MAGIC 💡 **Efecto consistente**: Viernes aumenta ventas para TODOS.
# MAGIC
# MAGIC **Ejemplo 2: Efecto Heterogéneo**
# MAGIC ```
# MAGIC Predicción
# MAGIC    $
# MAGIC    │  Líneas van en
# MAGIC 150│  direcciones opuestas!
# MAGIC    │    ┌──────
# MAGIC 140│   /
# MAGIC    │  /   \
# MAGIC 130│ /     \
# MAGIC    │/       \
# MAGIC 120│         ╰─────
# MAGIC    └──────────────────> Día
# MAGIC     Lun       Vie      Dom
# MAGIC ```
# MAGIC 💡 **Efecto heterogéneo**: 
# MAGIC - 50% clientes: ventas suben los viernes
# MAGIC - 50% clientes: ventas bajan los viernes
# MAGIC - PDP mostraría promedio (sin efecto) → **engañoso**!
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### ⚖️ PDP vs. ICE
# MAGIC
# MAGIC | Aspecto | PDP | ICE |
# MAGIC |---------|-----|-----|
# MAGIC | **Visualización** | 1 línea (promedio) | N líneas (individuales) |
# MAGIC | **Muestra heterogeneidad** | ❌ No | ✅ Sí |
# MAGIC | **Fácil de interpretar** | ✅ Sí | ❌ Puede ser confuso con muchos datos |
# MAGIC | **Velocidad** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ (más plots) |
# MAGIC
# MAGIC 💡 **Recomendación**: Usar **ambos juntos** (kind='both').
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 📊 Cuándo Usar ICE
# MAGIC
# MAGIC ✅ **Usar cuando**:
# MAGIC - Sospechas **interacciones** complejas
# MAGIC - Quieres ver **variabilidad** individual
# MAGIC - Dataset tiene **subgrupos** diferentes
# MAGIC
# MAGIC ⚠️ **Cuidado con**:
# MAGIC - Datasets muy grandes (demasiadas líneas) → samplear
# MAGIC - Features discretas (líneas superpuestas)
# MAGIC
# MAGIC ---

# COMMAND ----------

# DBTITLE 1,3. Feature Importance
# MAGIC %md
# MAGIC ## 3️⃣ Feature Importance (Tree-based)
# MAGIC
# MAGIC ### 📖 Concepto
# MAGIC
# MAGIC **Feature Importance** mide **cuánto contribuye cada feature** a reducir la impureza en árboles de decisión.
# MAGIC
# MAGIC **Cálculo** (para Random Forest):
# MAGIC 1. Para cada árbol, calcular reducción de impureza por feature
# MAGIC 2. Promediar sobre todos los árboles
# MAGIC 3. Normalizar a suma = 1
# MAGIC
# MAGIC **Impureza**: Gini (clasificación) o MSE (regresión)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 💻 Implementación
# MAGIC
# MAGIC ```python
# MAGIC from sklearn.ensemble import RandomForestRegressor
# MAGIC import pandas as pd
# MAGIC import matplotlib.pyplot as plt
# MAGIC
# MAGIC # Entrenar modelo
# MAGIC model = RandomForestRegressor(n_estimators=100, random_state=42)
# MAGIC model.fit(X_train, y_train)
# MAGIC
# MAGIC # Feature importance
# MAGIC importances = pd.DataFrame({
# MAGIC     'feature': X_train.columns,
# MAGIC     'importance': model.feature_importances_
# MAGIC }).sort_values('importance', ascending=False)
# MAGIC
# MAGIC print(importances)
# MAGIC
# MAGIC # Visualizar
# MAGIC plt.figure(figsize=(10, 6))
# MAGIC plt.barh(importances['feature'][:10], importances['importance'][:10])
# MAGIC plt.xlabel('Importancia')
# MAGIC plt.title('Top 10 Features más Importantes')
# MAGIC plt.gca().invert_yaxis()
# MAGIC plt.show()
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### ✅ Ventajas
# MAGIC
# MAGIC 1. ✅ **Rápido**: Calculado durante entrenamiento
# MAGIC 2. ✅ **Fácil de entender**: Un número por feature
# MAGIC 3. ✅ **Incorporado**: Built-in en sklearn
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### ❌ Desventajas y Limitaciones
# MAGIC
# MAGIC #### 1. **Sesgo con Features de Alta Cardinalidad**
# MAGIC
# MAGIC ```python
# MAGIC # Feature con muchos valores únicos tiene ventaja injusta
# MAGIC df['id_transaccion']  # 10,000 valores únicos → Alta importancia (falsa)
# MAGIC df['dia_semana']      # 7 valores → Baja importancia (subestimada)
# MAGIC ```
# MAGIC
# MAGIC ⚠️ **Problema**: Features con más valores tienen más oportunidades de split.
# MAGIC
# MAGIC #### 2. **Sesgo con Features Correlacionadas**
# MAGIC
# MAGIC ```python
# MAGIC # Si feature_A y feature_B están correlacionadas:
# MAGIC # - Importancia se "reparte" entre ambas
# MAGIC # - Una puede dominar arbitrariamente
# MAGIC ```
# MAGIC
# MAGIC #### 3. **Solo para Modelos Tree-based**
# MAGIC
# MAGIC ❌ No funciona con:
# MAGIC - Regresión lineal
# MAGIC - SVM
# MAGIC - Redes neuronales
# MAGIC - KNN
# MAGIC
# MAGIC #### 4. **No Captura Dirección**
# MAGIC
# MAGIC ```python
# MAGIC # Feature importance dice: "mes" es importante (30%)
# MAGIC # Pero NO dice:
# MAGIC # - ¿Más ventas en verano o invierno?
# MAGIC # - ¿Relación lineal o no lineal?
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 📊 Ejemplo: Panadería
# MAGIC
# MAGIC ```python
# MAGIC import pandas as pd
# MAGIC from sklearn.ensemble import RandomForestRegressor
# MAGIC
# MAGIC # Cargar datos
# MAGIC df = pd.read_csv('ventas.csv')
# MAGIC X = df[['sucursal_id', 'dia_semana', 'mes', 'zona_h3_encoded', 'segmento_encoded']]
# MAGIC y = df['total']
# MAGIC
# MAGIC # Entrenar
# MAGIC model = RandomForestRegressor(n_estimators=100, random_state=42)
# MAGIC model.fit(X, y)
# MAGIC
# MAGIC # Feature importance
# MAGIC for feat, imp in zip(X.columns, model.feature_importances_):
# MAGIC     print(f"{feat:20s}: {imp:.4f}  {'=' * int(imp * 100)}")
# MAGIC ```
# MAGIC
# MAGIC **Salida**:
# MAGIC ```
# MAGIC zona_h3_encoded     : 0.3521  ===================================
# MAGIC dia_semana          : 0.2845  ============================
# MAGIC mes                 : 0.1832  ==================
# MAGIC segmento_encoded    : 0.1204  ============
# MAGIC sucursal_id         : 0.0598  =====
# MAGIC ```
# MAGIC
# MAGIC 💡 **Interpretación**: La zona H3 es la feature más importante (35%), seguida del día de la semana (28%).
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 🔑 Mensaje Clave
# MAGIC
# MAGIC ⚠️ **Feature Importance es útil como primer paso**, pero tiene **limitaciones**.
# MAGIC
# MAGIC ✅ **Mejor**: Combinar con **Permutation Importance** (más confiable).
# MAGIC
# MAGIC ---

# COMMAND ----------

# DBTITLE 1,4. Permutation Importance
# MAGIC %md
# MAGIC ## 4️⃣ Permutation Importance
# MAGIC
# MAGIC ### 📖 Concepto
# MAGIC
# MAGIC **Permutation Importance** mide **cuánto empeora** el rendimiento cuando se **permutan (mezclan)** los valores de una feature.
# MAGIC
# MAGIC **Proceso**:
# MAGIC 1. Entrenar modelo y calcular rendimiento baseline
# MAGIC 2. Para cada feature:
# MAGIC    - **Permutar** sus valores aleatoriamente
# MAGIC    - Predecir con feature permutada
# MAGIC    - Calcular nuevo rendimiento
# MAGIC    - Importancia = baseline - nuevo rendimiento
# MAGIC 3. Repetir N veces (ej: 10) y promediar
# MAGIC
# MAGIC **Intuición**: Si permutar una feature **empeora mucho** el modelo → feature es **importante**.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 💻 Implementación
# MAGIC
# MAGIC ```python
# MAGIC from sklearn.inspection import permutation_importance
# MAGIC from sklearn.ensemble import RandomForestRegressor
# MAGIC from sklearn.model_selection import train_test_split
# MAGIC
# MAGIC # Split
# MAGIC X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# MAGIC
# MAGIC # Entrenar
# MAGIC model = RandomForestRegressor(n_estimators=100, random_state=42)
# MAGIC model.fit(X_train, y_train)
# MAGIC
# MAGIC # Permutation importance
# MAGIC result = permutation_importance(
# MAGIC     model, X_test, y_test,
# MAGIC     n_repeats=10,  # Repeticiones
# MAGIC     random_state=42,
# MAGIC     n_jobs=-1
# MAGIC )
# MAGIC
# MAGIC # Resultados
# MAGIC importances = pd.DataFrame({
# MAGIC     'feature': X.columns,
# MAGIC     'importance': result.importances_mean,
# MAGIC     'std': result.importances_std
# MAGIC }).sort_values('importance', ascending=False)
# MAGIC
# MAGIC print(importances)
# MAGIC
# MAGIC # Visualizar con error bars
# MAGIC import matplotlib.pyplot as plt
# MAGIC
# MAGIC plt.figure(figsize=(10, 6))
# MAGIC plt.barh(
# MAGIC     importances['feature'][:10],
# MAGIC     importances['importance'][:10],
# MAGIC     xerr=importances['std'][:10]  # Error bars
# MAGIC )
# MAGIC plt.xlabel('Permutation Importance')
# MAGIC plt.title('Top 10 Features por Permutation Importance')
# MAGIC plt.gca().invert_yaxis()
# MAGIC plt.show()
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### ✅ Ventajas
# MAGIC
# MAGIC 1. ✅ **Model-agnostic**: Funciona con **cualquier modelo**
# MAGIC 2. ✅ **Más confiable** que RF importance (sin sesgo de cardinalidad)
# MAGIC 3. ✅ **Maneja correlaciones** mejor
# MAGIC 4. ✅ **Incluye varianza** (error bars)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### ❌ Desventajas
# MAGIC
# MAGIC 1. ❌ **Más lento**: Requiere N × M predicciones (N features, M repeticiones)
# MAGIC 2. ❌ **Requiere test set**: No usar train (sobreestima importancia)
# MAGIC 3. ❌ **Puede ser inestable**: Con features muy correlacionadas
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### ⚖️ RF Importance vs. Permutation Importance
# MAGIC
# MAGIC **Comparación**:
# MAGIC
# MAGIC | Aspecto | RF Importance | Permutation Importance |
# MAGIC |---------|---------------|-------------------------|
# MAGIC | **Velocidad** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
# MAGIC | **Confiabilidad** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
# MAGIC | **Sesgo cardinalidad** | ❌ Sí | ✅ No |
# MAGIC | **Model-agnostic** | ❌ No | ✅ Sí |
# MAGIC | **Incertidumbre** | ❌ No | ✅ Sí (std) |
# MAGIC
# MAGIC **Ejemplo de Diferencia**:
# MAGIC
# MAGIC ```python
# MAGIC # Dataset con feature correlacionadas
# MAGIC df['peso_kg'] = ...   # Peso en kilogramos
# MAGIC df['peso_lb'] = df['peso_kg'] * 2.205  # Peso en libras (perfectamente correlacionado)
# MAGIC
# MAGIC # RF Importance:
# MAGIC peso_kg: 0.15
# MAGIC peso_lb: 0.35  # ⚠️ Sobrestimado (alta cardinalidad por decimales)
# MAGIC
# MAGIC # Permutation Importance:
# MAGIC peso_kg: 0.25
# MAGIC peso_lb: 0.25  # ✅ Similar (correctamente)
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 📊 Cuándo Usar
# MAGIC
# MAGIC **Usa RF Importance cuando**:
# MAGIC - Exploración rápida
# MAGIC - Solo usas Random Forest/XGBoost
# MAGIC - Dataset grande (velocidad crítica)
# MAGIC
# MAGIC **Usa Permutation Importance cuando**:
# MAGIC - Quieres **máxima confiabilidad**
# MAGIC - Tienes features **correlacionadas**
# MAGIC - Usas modelos **no tree-based**
# MAGIC - Presentación a stakeholders
# MAGIC
# MAGIC 💡 **Recomendación**: **Siempre usar Permutation** para análisis final.
# MAGIC
# MAGIC ---

# COMMAND ----------

# DBTITLE 1,Título del notebook
# MAGIC %md
# MAGIC # 🔍 Interpretabilidad de Modelos de Machine Learning
# MAGIC ## Material Complementario - Laboratorio (Herramientas)
# MAGIC ### Universidad del Aconcagua - Mendoza, Argentina
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 🎯 Objetivos de Aprendizaje
# MAGIC
# MAGIC 1. Comprender **por qué la interpretabilidad** es crucial
# MAGIC 2. Diferenciar entre **interpretabilidad global vs. local**
# MAGIC 3. Dominar **Feature Importance** y Permutation Importance
# MAGIC 4. Aplicar **Partial Dependence Plots (PDP)** e **ICE**
# MAGIC 5. Usar **SHAP values** para explicaciones robustas
# MAGIC 6. Implementar **LIME** para explicaciones locales
# MAGIC 7. Comparar métodos y elegir el adecuado
# MAGIC 8. Aplicar a **features H3 y temporales**
# MAGIC
# MAGIC ### 📁 Contenido
# MAGIC
# MAGIC 1. ¿Por qué Interpretabilidad?
# MAGIC 2. Interpretabilidad Global vs. Local
# MAGIC 3. Feature Importance (Tree-based)
# MAGIC 4. Permutation Importance
# MAGIC 5. Partial Dependence Plots (PDP)
# MAGIC 6. Individual Conditional Expectation (ICE)
# MAGIC 7. SHAP Values (SHapley Additive exPlanations)
# MAGIC 8. LIME (Local Interpretable Model-agnostic Explanations)
# MAGIC 9. Casos Prácticos con H3
# MAGIC 10. Comparación de Métodos
# MAGIC 11. Mejores Prácticas
# MAGIC
# MAGIC ### ⏱️ Duración Estimada: 2-3 horas
# MAGIC
# MAGIC ---

# COMMAND ----------

# DBTITLE 1,1. ¿Por qué Interpretabilidad?
# MAGIC %md
# MAGIC ## 1️⃣ ¿Por qué Necesitamos Interpretabilidad?
# MAGIC
# MAGIC ### 🎭 El Problema: Modelos "Black Box"
# MAGIC
# MAGIC **Escenario típico**:
# MAGIC ```python
# MAGIC model = RandomForestRegressor()
# MAGIC model.fit(X_train, y_train)
# MAGIC y_pred = model.predict(X_test)
# MAGIC
# MAGIC # ¿Pero CÓMO llegó a esta predicción?
# MAGIC # ¿QUÉ features fueron importantes?
# MAGIC # ¿POR QUÉ predijo $150 y no $100?
# MAGIC ```
# MAGIC
# MAGIC ❌ **Problema**: El modelo funciona pero **no sabemos por qué**.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 💡 ¿Qué es Interpretabilidad?
# MAGIC
# MAGIC **Interpretabilidad** = Capacidad de **explicar** o **entender** las decisiones de un modelo de ML.
# MAGIC
# MAGIC **Dos dimensiones**:
# MAGIC
# MAGIC 1. **Interpretabilidad Global**: ¿Cómo funciona el modelo en general?
# MAGIC    - ¿Qué features son más importantes?
# MAGIC    - ¿Cómo afecta cada feature a las predicciones?
# MAGIC
# MAGIC 2. **Interpretabilidad Local**: ¿Por qué el modelo hizo ESTA predicción específica?
# MAGIC    - Para este cliente, ¿por qué predijo $150?
# MAGIC    - ¿Qué cambiaría para obtener una predicción diferente?
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 🎯 ¿Por qué es Importante?
# MAGIC
# MAGIC #### 1. **Confianza y Adopción**
# MAGIC
# MAGIC ```python
# MAGIC # Stakeholder: "¿Por qué debo confiar en este modelo?"
# MAGIC # Tú: "Porque tiene 95% de accuracy"
# MAGIC # Stakeholder: "Pero ¿CÓMO funciona?"
# MAGIC ```
# MAGIC
# MAGIC ✅ **Con interpretabilidad**: "El modelo usa principalmente el historial de compras y la ubicación geográfica. Para este cliente, su alta frecuencia de compra y zona H3 urbana aumentan la predicción en $30."
# MAGIC
# MAGIC #### 2. **Cumplimiento Legal y Regulatorio**
# MAGIC
# MAGIC 📜 **GDPR (Europa)**: "Derecho a explicación" de decisiones automatizadas.
# MAGIC 📜 **Fair Lending (USA)**: Explicar decisiones de crédito.
# MAGIC 📜 **BCRA (Argentina)**: Transparencia en modelos de riesgo.
# MAGIC
# MAGIC #### 3. **Debugging y Mejora del Modelo**
# MAGIC
# MAGIC ```python
# MAGIC # Feature importance revela:
# MAGIC # - "ID de transacción" tiene alta importancia → 🚨 Data leakage!
# MAGIC # - "Mes" no es importante → 💡 Estacionalidad no capturada
# MAGIC ```
# MAGIC
# MAGIC #### 4. **Detección de Bias**
# MAGIC
# MAGIC ```python
# MAGIC # SHAP muestra:
# MAGIC # - Género afecta predicción salarial → 🚨 Sesgo discriminatorio
# MAGIC # - Código postal (proxy de raza) predice crédito → 🚨 Red-lining
# MAGIC ```
# MAGIC
# MAGIC #### 5. **Conocimiento del Negocio**
# MAGIC
# MAGIC 💡 **Descubrir patrones** que el negocio no conocía:
# MAGIC - "Clientes que compran pan los lunes tienen 2x más probabilidad de volver"
# MAGIC - "La zona H3 del microcentro tiene ventas 30% mayores de 7-9am"
# MAGIC
# MAGIC #### 6. **Acción y Recomendaciones**
# MAGIC
# MAGIC ```python
# MAGIC # Para aumentar la predicción de ventas:
# MAGIC # SHAP dice: "Aumentar inventario en zona H3 X los viernes +15%"
# MAGIC # LIME dice: "Este cliente compraría más si le ofrecemos descuento en medialunas"
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### ⚖️ Trade-off: Accuracy vs. Interpretabilidad
# MAGIC
# MAGIC **Espectro de modelos**:
# MAGIC
# MAGIC ```
# MAGIC Interpretables                       Black Box
# MAGIC │                                        │
# MAGIC │  Regresión Lineal                      │
# MAGIC │  │                                     │
# MAGIC │  │  Regresión Logística                │
# MAGIC │  │  │                                  │
# MAGIC │  │  │  Árbol de Decisión               │
# MAGIC │  │  │  │                               │
# MAGIC │  │  │  │  Random Forest                │
# MAGIC │  │  │  │  │                            │
# MAGIC │  │  │  │  │  XGBoost                   │
# MAGIC │  │  │  │  │  │                         │
# MAGIC │  │  │  │  │  │  Redes Neuronales       │
# MAGIC │  │  │  │  │  │  │                      │
# MAGIC ⭐⭐⭐  ⭐⭐  ⭐  ⭐  ⭐  ⭐  Accuracy           
# MAGIC ```
# MAGIC
# MAGIC **Dilemma**:
# MAGIC - Modelos simples (regresión lineal) → **Interpretables** pero **menos precisos**
# MAGIC - Modelos complejos (redes neuronales) → **Más precisos** pero **menos interpretables**
# MAGIC
# MAGIC 🎯 **Solución**: Usar **técnicas de interpretabilidad** (SHAP, LIME, PDP) para explicar modelos complejos.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 📈 Ejemplo Motivador
# MAGIC
# MAGIC **Caso**: Modelo predice ventas de panadería.
# MAGIC
# MAGIC **Sin interpretabilidad**:
# MAGIC ```
# MAGIC Predicción: $152.30
# MAGIC ```
# MAGIC
# MAGIC **Con interpretabilidad (SHAP)**:
# MAGIC ```
# MAGIC Predicción Base: $120.00
# MAGIC + Día viernes:        +$15.00
# MAGIC + Zona H3 microcentro: +$20.00
# MAGIC + Cliente frecuente:   +$8.00
# MAGIC - Mes enero (verano):  -$10.70
# MAGIC = Predicción Final:    $152.30
# MAGIC ```
# MAGIC
# MAGIC 💡 **Ahora sabemos**:
# MAGIC - **Por qué** la predicción es alta
# MAGIC - **Qué features** contribuyeron más
# MAGIC - **Cómo cambiar** la predicción (ej: mover a otra zona)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 📊 Tipos de Preguntas que Podemos Responder
# MAGIC
# MAGIC **Interpretabilidad Global**:
# MAGIC 1. ¿Cuáles son las 5 features más importantes?
# MAGIC 2. ¿Cómo afecta "día de la semana" a las predicciones?
# MAGIC 3. ¿Qué relación hay entre "zona H3" y "ventas"?
# MAGIC
# MAGIC **Interpretabilidad Local**:
# MAGIC 1. Para **este cliente**, ¿por qué predijimos $150?
# MAGIC 2. ¿Qué cambiaría para que la predicción sea $200?
# MAGIC 3. ¿Es esta predicción confiable?
# MAGIC
# MAGIC ---

# COMMAND ----------

# DBTITLE 1,2. Interpretabilidad Global vs. Local
# MAGIC %md
# MAGIC ## 2️⃣ Interpretabilidad Global vs. Local
# MAGIC
# MAGIC ### 🌍 Interpretabilidad Global
# MAGIC
# MAGIC **Pregunta**: ¿Cómo funciona el modelo **en general**?
# MAGIC
# MAGIC **Métodos**:
# MAGIC - Feature Importance
# MAGIC - Permutation Importance
# MAGIC - Partial Dependence Plots (PDP)
# MAGIC - SHAP Global (promedios)
# MAGIC
# MAGIC **Ejemplo**: "El modelo usa principalmente 'zona H3' (30%) y 'día de semana' (25%) para predecir ventas."
# MAGIC
# MAGIC **Cuándo usar**:
# MAGIC - Entender el modelo completo
# MAGIC - Comunicar a stakeholders
# MAGIC - Detectar data leakage
# MAGIC - Feature selection
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 🎯 Interpretabilidad Local
# MAGIC
# MAGIC **Pregunta**: ¿Por qué el modelo hizo **esta predicción específica**?
# MAGIC
# MAGIC **Métodos**:
# MAGIC - SHAP Local (valores individuales)
# MAGIC - LIME
# MAGIC - Individual Conditional Expectation (ICE)
# MAGIC
# MAGIC **Ejemplo**: "Para el cliente #12345, la predicción de $150 se debe a: zona H3 microcentro (+$20), viernes (+$15), cliente frecuente (+$8)."
# MAGIC
# MAGIC **Cuándo usar**:
# MAGIC - Explicar predicciones individuales
# MAGIC - Debugging de casos anómalos
# MAGIC - Compliance (explicar decisiones)
# MAGIC - Recomendaciones personalizadas
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 📐 Comparación Visual
# MAGIC
# MAGIC **Interpretabilidad Global**:
# MAGIC ```
# MAGIC 🌍 Vista del bosque completo
# MAGIC
# MAGIC    Feature Importance:
# MAGIC    │
# MAGIC    ├── zona_h3        ███████████████ 30%
# MAGIC    ├── dia_semana     ████████████ 25%
# MAGIC    ├── mes            ████████ 20%
# MAGIC    └── segmento       █████ 15%
# MAGIC ```
# MAGIC
# MAGIC **Interpretabilidad Local**:
# MAGIC ```
# MAGIC 🎯 Vista de un árbol específico
# MAGIC
# MAGIC    Cliente #12345 - Predicción: $152.30
# MAGIC    │
# MAGIC    ├── Base:              $120.00
# MAGIC    ├── zona_h3:           +$20.00  ⬆️
# MAGIC    ├── dia_semana:        +$15.00  ⬆️
# MAGIC    ├── cliente_frecuente: +$8.00   ⬆️
# MAGIC    └── mes:               -$10.70  ⬇️
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 📊 Matriz de Métodos
# MAGIC
# MAGIC | Método | Global | Local | Model-Agnostic | Velocidad |
# MAGIC |---------|--------|-------|----------------|------------|
# MAGIC | **Feature Importance** | ✅ | ❌ | ❌ (solo trees) | ⭐⭐⭐⭐⭐ |
# MAGIC | **Permutation Importance** | ✅ | ❌ | ✅ | ⭐⭐⭐ |
# MAGIC | **PDP** | ✅ | ❌ | ✅ | ⭐⭐⭐ |
# MAGIC | **ICE** | ❌ | ✅ | ✅ | ⭐⭐ |
# MAGIC | **SHAP** | ✅ | ✅ | ✅ | ⭐⭐ |
# MAGIC | **LIME** | ❌ | ✅ | ✅ | ⭐⭐⭐ |
# MAGIC
# MAGIC **Model-Agnostic** = Funciona con cualquier modelo (no solo árboles)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 🧠 Cuándo Usar Cada Tipo
# MAGIC
# MAGIC **Usa Interpretabilidad Global cuando**:
# MAGIC - 📊 Presentas el modelo a stakeholders
# MAGIC - 🔍 Debuggeas el modelo (detectar leakage)
# MAGIC - ⚖️ Comparas features (feature selection)
# MAGIC - 📚 Documentas el modelo para producción
# MAGIC
# MAGIC **Usa Interpretabilidad Local cuando**:
# MAGIC - 👤 Explicas una predicción a un usuario
# MAGIC - 🚨 Investigas casos anómalos (outliers)
# MAGIC - ⚖️ Cumples regulaciones (right to explanation)
# MAGIC - 🎯 Generas recomendaciones personalizadas
# MAGIC
# MAGIC **Usa AMBAS cuando**:
# MAGIC - 🎯 Quieres entendimiento completo del modelo
# MAGIC - 📈 Presentación a negocio (global) + casos de uso (local)
# MAGIC
# MAGIC ---