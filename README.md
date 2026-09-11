# Olist E-Commerce — Recomendaciones, Entregas y Sentimiento

Proyecto de Machine Learning end-to-end sobre el
[Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
(Kaggle), que explora tres problemas de negocio distintos sobre una plataforma real de
e-commerce brasileña: personalización de catálogo, logística y experiencia de cliente.

El objetivo es recorrer el ciclo completo de un proyecto de ML aplicado: desde la limpieza y el
feature engineering riguroso hasta la comparación de múltiples modelos, la detección de sobreajuste
(overfitting) y la traducción de los hallazgos técnicos en conclusiones accionables para el negocio.
Como bonus, se incluye un sistema de automatización de experimentos (GridSearchCV + cross-validation)
que hace reproducible la comparación entre modelos.

---

## Preguntas que explora el proyecto

1. **Recomendaciones y segmentación** (aprendizaje no supervisado) — ¿Existen grupos naturales de
   productos que permitan pensar en recomendaciones o segmentación estratégica de catálogo?
   `log1p` + `StandardScaler` + K-Means ($K=4$) y reducción de dimensionalidad con PCA para
   interpretación y visualización.
2. **Tiempos de entrega** (regresión supervisada) — ¿Se puede predecir con exactitud cuánto va a
   tardar un pedido en llegar a partir de datos disponibles al momento de la compra? Comparación
   entre Árbol de Decisión, Random Forest y Gradient Boosting.
3. **Sentimiento en reviews** (NLP / clasificación de texto) — ¿Se puede inferir si una reseña es
   positiva o negativa a partir del texto en portugués, usando `review_score` como proxy ante la
   ausencia de una etiqueta explícita? TF-IDF + Naive Bayes (`ComplementNB`) vs. Support Vector
   Machine (`LinearSVC`).

---

## Enfoque técnico

- Pipelines reproducibles de `scikit-learn` para preprocesamiento y modelado.
- Comparación sistemática de múltiples modelos por misión con reporte de métricas en train y test.
- Diagnóstico honesto de overfitting mediante la brecha entre métricas de train y test.
- Preprocesamiento específico para texto en portugués (`RSLPStemmer`, rescate explícito de negaciones).
- Transformaciones robustas para datos asimétricos en clustering (`log1p` antes de estandarizar).
- Bonus de automatización de experimentos: `ExperimentRunner` + GridSearchCV que ejecuta, compara y
  guarda los resultados de todos los experimentos en JSON.

---

## Estructura del proyecto

```
.
├── data/
│   ├── raw/          # CSVs originales de Olist (no versionados)
│   ├── processed/    # Datasets intermedios limpios (no versionados)
│   └── recursos/     # Documentación técnica y guías de estudio por misión
├── notebooks/        # Notebooks reproducibles con outputs ejecutados
│   ├── 00_inventary_data.ipynb
│   ├── 01_catalog_segmentation.ipynb
│   ├── 02_plazo_entrega.ipynb
│   ├── 03_opiniones_reviews.ipynb
│   └── 04_automacion_experimentos.ipynb
├── docs/             # Documentación profunda por misión (la "memoria de las decisiones")
│   ├── guia_conceptos_challenge.md
│   ├── informe1_segmentacion_catalogo.md
│   ├── informe2_plazo_entrega.md
│   └── informe3_opiniones_reviews.md
├── outputs/          # Gráficos y resultados exportados (no versionados)
├── requirements.txt  # Dependencias del proyecto
└── README.md
```

> **Nota sobre los datos:** Los archivos dentro de `data/raw/`, `data/processed/` y `outputs/` no se
> versionan en Git (ver `.gitignore`); el dataset se descarga directamente desde Kaggle y se ubica
> en `data/raw/`.

---

## Resultados y Comparación de Modelos Ganadores

Cada problema se evaluó contra métricas alineadas con su naturaleza matemática y su impacto en el negocio:

| Misión | Tipo de Aprendizaje | Modelo Ganador | Métrica Principal (Test) | Diagnóstico Metodológico |
|---|---|---|---|---|
| **1. Catálogo** | No Supervisado (Clustering) | **K-Means ($K=4$) + PCA** | **Silhouette = 0.249** (K=4); inercia 74,434.5 | Clusters balanceados (11.6% a 40.7%), perfiles comerciales diferenciados; estructura moderada por features continuas. |
| **2. Entregas** | Supervisado (Regresión) | **Gradient Boosting** | **MAE = 4.74 días** (CV 4.78) | Único con R² train (0.343) ≈ R² test (0.344): no sobreajusta. Decision Tree colapsa (R² test negativo) y Random Forest sobreajusta (gap R² 0.56). |
| **3. Opiniones** | Supervisado (NLP / Clasificación) | **SVM (`LinearSVC`)** | **Accuracy = 0.92** (CV 0.918) | Supera a ComplementNB (0.90); monitorea la clase negativa (F1 0.86, recall 0.92) para no perder quejas. |

---

## La Historia de Negocio: Conexión entre las Tres Misiones

En lugar de tres ejercicios técnicos aislados, los hallazgos componen una narrativa completa sobre las
operaciones y la experiencia de usuario en Olist:

```
┌─────────────────────────────────────────┐
│         Misión 1: Catálogo              │
│  Segmentar productos, no clientes:      │
│  97% de los compradores adquieren 1 vez.│
└────────────────────┬────────────────────┘
                     │ Define el mix de oferta
                     ▼
┌─────────────────────────────────────────┐
│         Misión 2: Logística             │
│  Predecir demoras ANTES de que ocurran  │
│  con features del momento del pedido.   │
└────────────────────┬────────────────────┘
                     │ Condiciona la satisfacción
                     ▼
┌─────────────────────────────────────────┐
│         Misión 3: Sentimiento           │
│  Detectar quejas en reviews para        │
│  escalarlas y responder temprano.       │
└─────────────────────────────────────────┘
```

1. **Del Catálogo a la Estrategia Comercial (Misión 1 $\to$ Negocio):**
   - El análisis de los datos reveló que el **97% de los compradores de Olist adquiere un producto
     una sola vez**: no hay historial recurrente por cliente para segmentar. La decisión analítica
     correcta fue segmentar el **catálogo de productos** (~32 mil artículos) mediante variables
     físicas, económicas y de demanda (`log1p` + K-Means + PCA).
   - Resultado: 4 perfiles comerciales accionables (chicos/económicos, grandes/premium de alto flete,
     best sellers de alta rotación, y long-tail de bajo movimiento).

2. **De la Logística a la Reputación (Misión 2 $\to$ Misión 3):**
- En la Misión 2 se demostró que el tiempo de entrega se puede predecir con features disponibles
      **al momento del pedido** (estacionalidad, peso, flete, promesa del vendedor), con MAE de ~4.7
      días y sin overfitting severo. `estimated_days` resultó clave — y plantea la pregunta de si la
      promesa de entrega es una profecía autocumplida.
   - En la Misión 3, el análisis de texto con TF-IDF + SVM y Naive Bayes permite separar reviews
     positivas de negativas en portugués. Combinadas, las dos misiones responden al mismo circuito
     de negocio: la logística condiciona la experiencia, y la experiencia se mide en las reviews.

3. **El Bonus: experimentos automatizados entre misiones:**
   - El `ExperimentRunner` reutiliza los mismos datos preparados de las misiones 2 y 3 para
     recorrer grids de hiperparámetros automáticamente, comparar modelos en una tabla única y
     detectar overfitting por la brecha train-test, guardando todo en `outputs/experiment_results.json`.

---

## Sentimiento y Reviews: lo que dice el texto

El clasificador de sentimiento se apoya en marcadores léxicos claros entre ambas clases: la clase
negativa se ancla en términos de queja y logística (demoras, no recibido, defecto), mientras la
positiva se asocia a rapidez y satisfacción. Para Olist, el valor operativo no es acertar la
mayoría positiva, sino **capturar la mayor cantidad posible de reviews negativas reales** (recall
sobre la clase negativa) para escalar quejas antes de que dañen la reputación pública.

---

## Conclusiones Estratégicas y Recomendaciones para Olist

1. **Estrategia de Catálogo por Segmento de Producto (Misión 1):**
   - **Best sellers (alta rotación):** asegurar stock y despacho inmediato con acuerdos de
     fulfillment de los vendedores correspondientes.
   - **Grandes y pesados (alto flete):** el costo de envío es el principal riesgo de satisfacción;
     conviene alianzas de flete dedicado para mitigar demoras y fricción.
2. **Promesa de Entrega Más Realista (Misión 2):**
   - Con un predictor de demoras usable al momento del pedido, Olist puede pasar de un SLA
     estático (posiblemente inflado como margen de seguridad) a plazos dinámicos ajustados por
     estado y tipo de pedido, sin incumplir.
3. **Mesa de Ayuda Proactiva (Misión 3):**
   - Desplegar el clasificador de reviews en tiempo real para disparar tickets automáticos de
     retención apenas un cliente publique una queja logística, reduciendo el daño reputacional.
4. **Automatización de Experimentos (Bonus):**
   - La tabla comparativa única, la detección de "modelos que parecían prometedores pero
     traicionaron" y el guardado en JSON convierten el experimento ML en un proceso reproducible
     y auditable, no en un score aislado.

---

## Trabajo Futuro Documentado

En los notebooks y reportes técnicos se documentaron oportunidades de mejora metodológica para
iteraciones posteriores:

- **Misión 1:** complementar la segmentación con un modelo de afinidad funcional y reglas de
  asociación (Market Basket Analysis / co-ocurrencia de productos en una misma orden) para
  habilitar recomendaciones cruzadas tipo *"quien compró esto también agregó..."*.
- **Misión 2:** evaluar Target Encoding out-of-fold sobre el historial de cumplimiento de cada
  vendedor para subir el techo de varianza explicada, con validación cruzada rigurosa que evite
  fuga de datos (data leakage).
- **Misión 3:** probar una arquitectura de 3 clases (incorporando el 3 estrellas neutral) e incluir
  bigramas en el vectorizador TF-IDF para capturar estructuras compuestas del lenguaje y
  negaciones multicutérminos.