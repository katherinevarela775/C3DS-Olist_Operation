# Guía de Estudio: Conceptos del Challenge Aplicados en Cada Sección

> Para estudiar con los notebooks: cada sección del proyecto corresponde a uno o más conceptos del
> challenge, y cada línea de código está anotada inline (`#`) dentro del propio notebook para leerlo
> de corrido. Esta guía es el mapa que conecta sección ↔ concepto.

---

## Mapa general challenge ↔ notebooks

El challenge ofrece 5 misiones + bonus. Este proyecto resuelve **3 misiones + bonus** — no en el
orden del enunciado, sino por afinidad de datos (el dataset de Olist tiene lo necesario):

| Reto del challenge                   | Concepto central                 | Notebook del proyecto                |
| ------------------------------------ | -------------------------------- | ------------------------------------- |
| 🛍️ Misión 2: Recomendaciones         | Clustering (K-Means) + PCA       | `01_catalog_segmentation.ipynb`       |
| 🚚 Misión 3: Predicción de entregas  | Árboles (DT / RF / GB)           | `02_plazo_entrega.ipynb`              |
| 🧠 Misión 5: Análisis de sentimientos| NLP + Naive Bayes + SVM          | `03_opiniones_reviews.ipynb`          |
| 🌟 Bonus: automatizar experimentos   | GridSearchCV + CV + comparación  | `04_automacion_experimentos.ipynb`    |
| 🧿 Previo común                       | Inventario / calidad de datos    | `00_inventary_data.ipynb`             |

**Nota:** el notebook 01 se titula "Misión 1" y el 02 "Misión 2" por el orden interno de trabajo,
pero frente al enunciado corresponden a las misiones 2 y 3 del reto.

---

## Notebook 00 — Inventario de datos (previa común)

Concepto del challenge: **ninguna misión concreta, pero es el prerrequisito de TODAS**. El challenge
exige entregables bien estructurados y "comparar / evaluar / optimizar": eso solo es posible si se
conoce el dato.

| Sección                                     | Concepto del challenge aplicado                                                        |
| ------------------------------------------- | -------------------------------------------------------------------------------------- |
| Recuento de tablas y su tamaño              | Conocer el dataset (filas × columnas) antes de modelar → evita errores de merge.        |
| Mapa de relaciones entre tablas             | Esquema relacional de Olist: claves entre pedidos/items/pagos/reviews → bases del feature engineering. |
| Perfil de tipos y valores faltantes         | Calidad de datos: nulos y `dtype` por columna → define estrategias de limpieza por misión. |
| Hoja de ruta de las misiones                | Alcance del proyecto → honestidad técnica (qué se resolverá y qué no).                  |

---

## Notebook 01 — Segmentando el Catálogo para Recomendar

Misión del challenge: **🛍️ Misión 2: Recomendaciones Inteligentes** (clustering + PCA).
Objetivo: agrupar productos para recomendar cosas coherentes (nada de bikini a alguien con laptop gamer).

| Sección                                        | Concepto del challenge aplicado                                                          |
| ---------------------------------------------- | ---------------------------------------------------------------------------------------- |
| Consolidar ventas por producto                 | Feature engineering: agregar órdenes/item por producto (unidades, ingresos).              |
| Anexar ficha técnica de cada producto          | Joins con dimensión → enriquecer features (peso, volumen).                               |
| Limpieza de registros incompletos              | Calidad de datos: nulos y outliers → reglas de negocio para descartar.                    |
| Qué tan sesgadas están las variables           | **Distribuciones / skew**: dato asimétrico rompe la distancia euclidiana de K-Means.       |
| Domar la cola larga con logaritmo              | **Transformación `log1p`**: logaritmo comprime colas largas (precio, peso) → concepto de normalización de datos. |
| Llevar variables a la misma escala             | **`StandardScaler`** (media 0, desvío 1): K-Means es sensible a la escala.                |
| Cuánto explica cada componente                 | **PCA / varianza explicada**: cuánta información conserva cada componente.                |
| Proyectar en 2D para explorar                  | **Reducción de dimensionalidad** con PCA → permite visualizar nubes de puntos.            |
| Métodos del codo y de la silueta para elegir K | **Selección de K**: inercia (codo) + `silhouette_score` → no hay K "mágica", se justifica. |
| Comparar codo y silueta en un gráfico          | Visualización interpretable (requisito del reto).                                        |
| Ajustar el K-Means definitivo                  | **K-Means** con `random_state` fijo → reproducibilidad.                                  |
| Perfil de cada cluster                         | **Interpretación de resultados**: promedio de features por grupo → lectura de negocio.    |
| Ubicar los clusters en el plano PCA            | Proyección de clusters sobre los componentes → storytelling visual.                      |
| Recomendador por cluster                       | Aplicación de negocio: sugerir productos del mismo cluster (recomendación coherente).    |

---

## Notebook 02 — Horizonte de Entrega

Misión del challenge: **🚚 Misión 3: Predicción de Entregas**.
Objetivo: predecir tiempos de entrega y **comparar modelos** (Árboles, RF, GB) para elegir el mejor.

| Sección                               | Concepto del challenge aplicado                                                               |
| ------------------------------------- | --------------------------------------------------------------------------------------------- |
| Construir la variable objetivo        | Definición de **target** (días de entrega) a partir de timestamps → `pd.to_datetime`.          |
| Observar la cola larga antes de recortar | EDA: ver la distribución antes de podar (no borrar a ciegas).                              |
| Poda de casos atípicos                | Manejo de outliers con criterio de negocio (días imposibles / extremos).                      |
| Fecha de compra y promesa de Olist    | **Feature engineering temporal** (mes, día de semana) y `estimated_days` sin fuga de datos.   |
| Combinar las tablas del pedido        | Joins múltiples (items, pagos, clientes) → consolidar features por pedido.                    |
| Codificar categorías y armar la matriz | **LabelEncoder** para variables categóricas; matriz X y vector objetivo y. (Sin leakage.)      |
| Repartir entre train y test           | **Train/test split** con seed → honestidad de la evaluación (no usar test para decidir).      |
| Duelo de árboles: DT, RF y GB         | **Comparación de modelos**: el challenge pide enfrentar modelos entre sí.                     |
| Detectando sobreajuste                | **Overfitting**: brecha entre métricas de train y test (≤0.05 OK; >0.15 severo).               |
| Gráfico comparativo de modelos        | Visualización MAE / R2 / tiempo por modelo → selección según contexto.                        |

---

## Notebook 03 — Voz del Cliente en las Reviews

Misión del challenge: **🧠 Misión 5: Análisis de Sentimientos**.
Objetivo: clasificar reseñas positivas/negativas con **Naive Bayes** vs **SVM** y detectar patrones
recurrentes (extra points por detectar señales estratégicas).

| Sección                                 | Concepto del challenge aplicado                                                              |
| --------------------------------------- | -------------------------------------------------------------------------------------------- |
| Reviews con texto y etiqueta de sentimiento | Etiquetado: `review_score` como proxy de sentimiento (≥4 positivo, <4 negativo; descartar 3). |
| Normalización del texto                 | **Preprocesamiento NLP**: minúsculas, URLs, dinero, solo letras, stopwords PT + stemming `RSLPStemmer`, conservar negaciones. |
| Reparto estratificado train/test        | **`stratify`**: mantener proporción de clases en train/test → evaluación honesta en desbalance. |
| Combate: Bayes vs SVM                   | **Comparación de modelos + pipelines**: `TfidfVectorizer` → `ComplementNB` vs `LinearSVC`.    |
| Matrices de confusión por modelo        | **Evaluación con matriz de confusión**: errores de tipo I/II según el modelo.                 |
| Validación cruzada de ambos             | **Cross-validation**: promedio ± desvío → robustez (no depender de un solo split).            |
| Términos que delatan cada sentimiento  | **Interpretabilidad del modelo**: coeficientes SVM / log-probabilidad NB → "señal estratégica" (bonus del reto). |

---

## Notebook 04 — Laboratorio de Experimentos (Bonus)

Concepto del challenge: **🌟 Bonus**: automatizar experimentos, tabla comparativa elegante y la
sección "Modelos que parecían prometedores pero me traicionaron".

| Sección                                  | Concepto del challenge aplicado                                                               |
| ---------------------------------------- | --------------------------------------------------------------------------------------------- |
| El motor ExperimentRunner                | **Abstracción / automatización**: clase que ejecuta GridSearchCV + CV y registra métricas.     |
| Datos de regresión (misión 2)            | Reutilización de los datasets ya preparados → reproducibilidad.                               |
| Datos de texto (misión 3)                | Idem para clasificación de texto.                                                            |
| Ronda 1: modelos de regresión            | **GridSearchCV por modelo**: DT, RF, GB con `param_grid` → optimización de hiperparámetros.   |
| Ronda 2: modelos de texto                | **GridSearchCV**: ComplementNB y LinearSVC con pipeline vectorizador → optimización NLP.      |
| Tabla de resultados y modelos traicioneros | **Comparación sistemática + tabla elegante** y la sección de modelos "traidores" (overfitting detectado). |
| Gráficos de la comparación               | Visualización train vs test por modelo y tarea → selección final con contexto.               |

---

## Skills del challenge que se desbloquean aquí

| Skill del reto                        | Dónde se practica                                                            |
| ------------------------------------- | ---------------------------------------------------------------------------- |
| Pipelines en scikit-learn             | 01 (escalar), 03 (TF-IDF → clf), 04 (pipeline en GridSearchCV)               |
| Comparación sistemática de modelos    | 02 (DT/RF/GB), 03 (NB/SVM), 04 (ExperimentRunner)                            |
| Detección de overfitting/underfitting | 02 (gap R2 train-test), 04 (sección "traidores")                             |
| Cross-validation                      | 03 (cross_val_score), 04 (cv=5 en GridSearchCV)                              |
| Visualización clara de resultados     | 01 (PCA y clusters), 02 (barras por modelo), 03 (matrices), 04 (train vs test)|
| Selección del mejor modelo según contexto | decisiones al cierre de cada notebook                                     |