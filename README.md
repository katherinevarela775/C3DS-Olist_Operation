# Olist: entregas, reviews y catálogo bajo el microscopio

Un recorrido de Machine Learning aplicado sobre el
[Brazilian E-Commerce Public Dataset de Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
(Kaggle). En vez de un ejercicio de "entrenar un modelo y listo", acá se recorre el ciclo real de un
proyecto: limpiar con criterio, construir features con su justificación, comparar modelos midiendo
el sobreajuste y cerrar la historia con conclusiones accionables para el negocio.

---

## El tablero del proyecto

El dataset de Olist permite leer tres problemas de negocio al mismo tiempo:

| # | Pregunta de negocio | Cómo se atacó | Notebook |
|---|---|---|---|
| 1 | ¿Existen grupos naturales de productos para segmentar el catálogo y recomendar? | `log1p` + `StandardScaler` + **K-Means (K=4)** + PCA | `01_catalog_segmentation.ipynb` |
| 2 | ¿Cuánto va a demorar el pedido, sabiendo solo lo que existe al momento de comprarlo? | Regresión: Decision Tree vs Random Forest vs Gradient Boosting | `02_plazo_entrega.ipynb` |
| 3 | ¿Se puede inferir el sentimiento de las reviews en portugués sin etiqueta de sentimiento? | NLP: TF-IDF + `ComplementNB` vs `LinearSVC` | `03_opiniones_reviews.ipynb` |

El hilo entre las tres misiones: la logística condiciona la experiencia, la experiencia se mide en
las reviews, y la estructura del catálogo explica qué productos son estratégicos. Por eso el proyecto
se lee como una sola historia, no como tres ejercicios sueltos.

---

## Cómo se trabajó cada problema

**Misión 1 — Catálogo.** El 97% de los clientes de Olist compra una sola vez: no hay historial
recurrente como para segmentar clientes. Se segmenta entonces el **catálogo** (~32.3 mil productos)
en base a precio, flete, peso, volumen y popularidad. Antes de clusterizar, `log1p` doma la cola
larga (razón máximos/mediana alta) y `StandardScaler` pone todas las features en la misma escala.
El número de clusters se elige con codo de inercia + silhouette (**K=4, silhouette 0.249**) y la
visualización usa **3 componentes de PCA (85.8% de varianza)**: PC1 = tamaño físico, PC2 =
popularidad, PC3 = precio relativo.

**Misión 2 — Entregas.** Se predice la demora en días con variables disponibles al momento del
pedido (estacionalidad, peso, flete, promesa estimada y zona geográfica). Los tres modelos de árbol
corren dentro de pipelines con `StandardScaler` y CV de 5 pliegues; se reportan métricas de train y
test para diagnosticar el sobreajuste, no solo el score ganador.

**Misión 3 — Sentimiento.** No existe columna de sentimiento: se usa `review_score` como proxy
(4-5 positivo, 1-2 negativo). El texto pasa por un preprocesamiento en portugués — minúsculas,
stemming RSLP y recuperación explícita de negaciones ("não", "nunca"…) — y se comparan Naive Bayes
`ComplementNB` (variante para clases desbalanceadas) con SVM lineal.

---

## Resultados

| Problema | Modelo elegido | Lo que importa |
|---|---|---|
| **Entregas** | **Gradient Boosting** | MAE ≈ **4.74 días** en test; R² train/test casi idéntico (0.343 / 0.344): el único sin sobreajuste |
| **Sentimiento** | **LinearSVC** | Accuracy **0.92** (CV 0.918 ± 0.002); recall de clase negativa **0.92** — prioridad para no perder quejas |
| **Catálogo** | **K-Means K=4 + PCA 3D** | Silhouette **0.249**; cuatro segmentos balanceados (11.6% a 40.7%) |

Dos observaciones que se escapan en un resumen de una línea:

- **Decision Tree es el "modelo traicionero".** MAE train 0.019 pero test 6.544, R² test negativo:
  memoriza el entrenamiento y predice peor que un modelo constante. Random Forest mejora pero
  todavía sobreajusta (gap de R² 0.560). El diagnóstico honesto del sobreajuste es parte tan
  importante del resultado como el score final.
- **El techo del problema de entregas.** Con el mejor modelo, R² test llega a ~0.34: alrededor del
  ~66% de la varianza de una demora ocurre en el tramo del transportista (rutas, clima, estado del
  correo), información que el dataset no contiene. Es un límite informativo, no una falla de tuning.

---

## Los hallazgos, en criollo

**Catálogo.** Los 4 clusters cuentan una historia comercial clara: chicos y económicos (34%), grandes
y premium de alto flete (14%), best sellers de alta rotación (12%) y una long-tail de precio medio y
bajo movimiento (40%). En el plano PCA, el grupo de best sellers es el que más se aísla (su demanda lo
separa del resto), mientras los clusters chico y medio se traslapan en tamaño y solo se despegan con
la tercera componente (precio relativo, +13.1% de varianza).

**Entregas.** El estimado de Olist (`estimated_days`) es una de las features más informativas, y la
pregunta que deja abierta es si la promesa es una **profecía autocumplida**: si el modelo copia la
promesa del vendedor, entonces la "decisión" de negocio sobre el SLA ya está implícita en los datos.

**Sentimiento.** El vocabulario que más tira a la clase negativa gira en torno a negaciones ("não") y
reclamos de calidad o urgencia (falsif-, incomplet-, urg-), mientras la positiva se ancla en
rapidez, atención y facilidad (ráp-, atenci-, fác-). Para Olist, el valor operativo no es acertar la
mayoría positiva: es **no perder reviews negativas reales** para escalar la queja antes de que
dañe la reputación pública.

---

## Qué le convendría hacer a Olist

1. **SLA dinámico en vez de estático (Misión 2).** Con una predicción usable al momento del pedido,
   se pueden ofrecer plazos ajustados por estado y tipo de pedido en vez de una promesa genérica
   (y probablemente inflada como margen de seguridad).
2. **Mesa de ayuda proactiva (Misión 3).** Desplegar el clasificador en tiempo real para disparar
   tickets de retención apenas una review negativa entre, cuando todavía hay margen de respuesta.
3. **Operación por segmento (Misión 1).** Garantizar stock y despacho inmediato para los best
   sellers, y alianzas de flete dedicado para los productos voluminosos, donde la demora y el costo
   de envío golpean la satisfacción.

---

## Limitaciones que asumimos con los ojos abiertos

- **Reviews de 3 estrellas fuera del modelo (8.7% del texto).** El puntaje 3 es una franja ambigua:
  quien puntúa así ni reclamó ni celebró. Forzarlo a una clase binaria ensuciaría las etiquetas con
  patrones neutros, así que el modelo solo aprende el espectro claro (1-2 vs 4-5). Costo: no existe
  categoría "neutro" y, en la práctica, una review de 3 estrellas caería del lado positivo. Para el
  caso de uso (detectar quejas temprano) el riesgo es bajo porque el recall negativo se mantiene en
  0.92 (SVM) / 0.94 (Naive Bayes).
- **Recomendador por misma-cluster ≠ afinidad real.** Agrupar por perfil comercial (tamaño, precio,
  demanda) es útil para estrategia de inventario y precios, pero no genera sustitutos funcionales
  ("compraste una cafetera, te recomiendo café"). Eso pediría reglas de co-ocurrencia dentro de un
  mismo pedido.
- **Techo de datos en entregas.** ~66% de la varianza queda fuera del dataset por pertenecer al
  tramo del transportista; ninguna transformación de las variables actuales lo va a recuperar.

---

## Próximos pasos

- **Catálogo:** sumar Market Basket Analysis (productos que se compran juntos en una misma orden)
  para recomendaciones cruzadas tipo *"quién compró esto también agregó"*.
- **Entregas:** evaluar Target Encoding *out-of-fold* con el historial de cumplimiento de cada
  vendedor, con validación cruzada rigurosa para evitar fuga de datos.
- **Sentimiento:** pasar a una arquitectura de 3 clases (positivo/neutro/negativo) incorporando el
  3 estrellas, y agregar bigramas al TF-IDF para capturar negaciones compuestas.

---

## Organización del repositorio

```
├── notebooks/          # Código reproducible con outputs ejecutados
│   ├── 00_inventary_data.ipynb
│   ├── 01_catalog_segmentation.ipynb
│   ├── 02_plazo_entrega.ipynb
│   └── 03_opiniones_reviews.ipynb
├── docs/               # Informes por misión y guía conceptual del challenge
├── src/                # Módulos compartidos
├── data/raw/           # CSVs de Kaggle (sin versionar)
├── requirements.txt    # Dependencias
└── README.md
```

> **Nota sobre los datos:** los CSVs de `data/` se descargan desde Kaggle y no se suben al repo;
> los notebooks esperan encontrarlos en `data/raw/`.