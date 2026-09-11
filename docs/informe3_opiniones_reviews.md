# Misión 3: Análisis de Sentimientos — Documentación Completa

## Resumen ejecutivo

Se clasifican reviews de Olist como positivas o negativas usando NLP. Como no existe una
columna de sentimiento etiquetada, se construyó el target a partir del `review_score`
acompañado de texto real. Se compararon Naive Bayes (ComplementNB) contra SVM lineal
(LinearSVC) sobre un pipeline de TF-IDF, y la decisión clave de esta misión es elegir
**cómo medir el desempeño en un problema de clases desbalanceadas**: la accuracy sola
engañaría, porque prediciendo siempre la clase mayoritaria se obtiene una cifra
aparentemente alta que no sirve para detectar las quejas reales. La razón para promediar
el F1 y leer la matriz de confusión es que a Olist le interesa **encontrar las reviews
negativas** — el valor operativo de este modelo está en capturar las quejas, no en
acertar la mayoría optimista.

---

## 1. El problema y el target

### 1.1 ¿Qué se predice?

Revisión de cliente sobre su compra. Hay una función de negocio clara: detectar
automáticamente las reviews negativas (clientes insatisfechos) para escalar quejas y
mejorar el servicio, incluso antes de que un humano las lea.

### 1.2 Construcción del target: de `review_score` a sentimiento

No viene una columna "sentimiento". Se derivó del rango de estrellas:

- **Positivo:** 4 o 5 estrellas → satisfecho.
- **Negativo:** 1 o 2 estrellas → queja.
- **Neutral (3 estrellas): se descartó del entrenamiento.** Es la zona ambigua que no
  aporta señal clara a una clasificación binaria. Un 3 puede ser "estuvo bien pero no
  me impresionó" como "tuve un problema menor". Metiéndolo se entrena al modelo para
  adivinar una línea borrosa y se degrada la calidad de la frontera decisoria.

**Filtro importante:** se usaron solo reviews con `review_comment_message` no nulo y con
texto no vacío. Las reviews sin texto son la mayoría del dataset, pero no tienen contenido
para clasificar — no son mala señal, simplemente no son clasificables por un modelo de
texto.

### 1.3 El desbalance de clases

En Olist las reviews tienden a estar polarizadas (4-5 y 1 son más frecuentes que 2-3).
Ese desbalance es real y ya fue consistente en los datos de texto. Si se midiera solo con
accuracy, un modelo inútil que predice "positivo" siempre ya tendría una cifra alta — por
eso se penaliza con métricas sensibles a la clase negativa.

---

## 2. Pipeline de NLP

### 2.1 Limpieza de texto

- **Todo a minúsculas**: cuenta normalizada de la misma palabra escrita distinto.
- **Sacar URLs y saltos de línea.**
- **Sacar dinero** (`r$\d+`) — los valores monetarios no aportan señal de sentimiento
  y son ruido repetitivo.
- **Quedarnos con letras** (incluyendo acentos del portugués): las emojis, números y
  símbolos no aportan a la regla de clasificación.
- **Stopwords en portugués:** se sacan las más comunes (artículos, preposiciones) que no
  transmiten sentimiento por sí solas.
- **Stemming con `RSLPStemmer`** (stemmer estándar para portugués): reduce cada palabra a
  su raíz ("gostei", "gostaram" → raíz común), reduciendo el espacio de features y
  agrupando variantes de la misma idea.

### 2.2 La decisión crítica: mantener las negaciones

Las stopwords se filtraron **pero se conservaron explícitamente las negaciones**
(`nao`, `não`, `nem`, `nunca`, `jamais`, `nenhum`, `nenhuma`, `tampouco`). La razón:
"no me gusto" y "me gusto" comparten el resto de la frase, pero significan lo opuesto. Si
el stemming y la limpieza mataran la negación, las frases negadas quedarían idénticas a
sus positivas y el modelo no tendría forma de distinguirlas.

Un tipo de palabra que no aparece (oversight honesto) son los intensificadores del tipo
"muy", "mucho", "extremadamente" — el stemming los neutraliza igual que a los positivos.
A futuro, un pipeline más sofisticado podría usar emojis o bigramas de negación para
capturar matices que el unigrama pierde.

### 2.3 Representación del texto: TF-IDF

**TF-IDF (Term Frequency - Inverse Document Frequency)** convierte cada review en un
vector numérico donde:
- **TF** cuenta cuántas veces aparece cada término en esa review.
- **IDF** pondera **a la baja** los términos que aparecen en casi todas las reviews
  (no discriminan) y **al alza** los que aparecen en pocas (son más informativos).

La clave: no es solo *"cuán frecuente es esta palabra en este texto"*, es *"cuán frecuente
es acá al tiempo que es rara en el resto del corpus"*. Una palabra que aparece una vez en
un texto pero también en el 95% de los demás aporta poco; una que aparece en 2% de las
reviews es un marcador fuerte.

---

## 3. Modelos comparados

### 3.1 Naive Bayes (ComplementNB)

Es una variante de Naive Bayes diseñada para **clases desbalanceadas**. Naive Bayes
aplica Bayes con la suposición de que las features son independientes (suposición
falsa, pero increíblemente efectiva en texto y muy barata). ComplementNB usa la
información del complemento de cada clase: en vez de calcular qué tan bien explica los
documentos de una clase, calcula qué tan bien explica los documentos de las otras, y
esto ayuda cuando una clase domina mucho el corpus (el caso pesimista de Olist con la
clase positiva dominante).

### 3.2 SVM lineal (LinearSVC)

SVM encuentra el hiperplano que **mejor separa** las dos clases con el mayor margen
posible. En texto, el espacio de features es enorme (miles de términos), algo que hace
inviable un kernel no lineal (por costo computacional), pero muy manejable con un kernel
lineal, que es el estándar de facto para clasificación de texto.

**`class_weight='balanced'`:** vuelve a pesar las clases para que la minoritaria
(negativa) tenga influencia proporcional en el margen — directamente conectado con el
problema de desbalance de la sección 1.3.

---

## 4. Resultados e interpretación

### 4.1 Naive Bayes vs. SVM (resultados reales, ejecución 2026)

| Modelo | Accuracy (test) | F1 (clase negativa) | Recall (negativo) | CV Accuracy |
|---|---|---|---|---|
| ComplementNB | 0.90 | 0.84 | 0.94 | 0.898 |
| LinearSVC | 0.92 | 0.86 | 0.92 | 0.918 |

**Por qué SVM gana:** como clasificador de margen, define la frontera con máxima
separación y maneja mejor los casos cercanos a la línea entre positivo y negativo.
Naive Bayes asume independencia entre palabras — muy simplificadora, aunque barata y
sólida.

### 4.2 La comparación honesta contra la referencia de negocio

Un modelo trivial que siempre dice "positivo" superaría a Naive Bayes en accuracy
(más del 85% de reviews son positivas). Pero el objetivo de negocio del análisis es
detectar clientes insatisfechos. Por eso se evaluaron las confusiones entre la clase
negativa real y la pronosticada, no solo la cifra global: el valor del modelo está en
cuántas quejas genuinas captura (recall sobre la clase negativa), incluso a costa de
alguna alarma falsa.

### 4.3 Palabras que alimentan cada extremo

Al inspeccionar los top términos de cada clase se esperan marcadores claros: palabras de
queja en la clase negativa (devolución, defecto, atrasado…) frente a satisfacción (rápido,
recomendado, impecable…). Además de validación técnica, esto es accionable: Olist podría
sumar alertas tempranas por review que toque términos de alto peso negativo.

---

## 5. Para memorizar y entender profundo

### 5.1 Qué es TF-IDF en una frase memorable

TF-IDF = contás las veces que aparece una palabra en tu review, pero la ponderás por lo
rara que es globalmente en el corpus. Palabras casi universales pierden relevancia;
palabras concentradas ganan peso.

### 5.2 Por qué se usa el F1 (o precision/recall de la clase negativa) y no solo accuracy

- **Accuracy** = aciertos / total. Oculto el desbalance: si el 85% de las reviews son
  positivas, predecir "positivo" siempre da 85% de accuracy y un contexto engañoso.
- **Recall (negativa)** = de las reviews que realmente son negativas, ¿cuántas capturé?
  A Olist le interesa maximizar esto (no perder quejas).
- **Precision (negativa)** = de las reviews que marqué negativas, ¿cuántas eran real?
  Penaliza las alarmas falsas.
- **F1 de la clase negativa:** promedia armónicamente precision y recall de esa clase;
  es la métrica que da una cifra única pero fiel al desbalance.

### 5.3 Por qué Naive Bayes soporta textos enormes así sin más

Cada término se trata como una variable independiente con su propia probabilidad condicional.
No se modela el orden de las palabras ni sus dependencias — por eso es barato, y por eso
mismo su debilidad: una frase como "no es malo" se leería como suma de términos
parcialmente contradictorios. En la práctica, TF-IDF lo hace sorprendentemente difícil
de mejorar en muchísimos problemas de texto.

### 5.4 Kernel lineal vs. no lineal en SVM, y por qué en texto se usa el lineal

La representación TF-IDF puede tener decenas de miles de dimensiones. Los kernels no
lineales (RBF, polinomial) operan transformando a un espacio de dimensionalidad aún mayor
(recomendada explícitamente la letra pequeña de los algoritmos), lo que multiplica el
costo. En textos, el lineal es suficiente: la frontera positivo/negativo es razonablemente
lineal en ese espacio de términos.

### 5.5 Qué es la matriz de confusión y por qué leerla

Es una tabla cruzada entre lo real y lo predicho: recategoriza los 4 casos posibles
(falso positivo, falso negativo, verdadero positivo, verdadero negativo). Leerla muestra
el tipo de error que comete el modelo, no solo cuánto se equivoca — y el tipo de error
importa para negocio (perder una queja es peor que marcar positiva una review neutral).