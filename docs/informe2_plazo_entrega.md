# Misión 2: Predicción de Entregas — Documentación Completa

## Resumen ejecutivo

Se entrenan modelos de árboles (Decision Tree, Random Forest, Gradient Boosting) para
predecir cuántos días tardará una orden en llegar al cliente, usando features
disponibles en el momento del pedido. Gradient Boosting resultó el ganador (MAE ≈ 4.7
días en test, con R² train y test casi idénticos), mientras que Random Forest y el
Decision Tree terminaron desplomados. El hallazgo central de esta misión
es que **elegir un modelo por cómo corre en los datos con los que se entrenó es una
trampa**: el Decision Tree logra R² casi perfecto en training y se desploma en test
(negativo), Random Forest mantiene un R² altísimo en train pero se cae en test
(sobreajuste moderado-severo), y solo Gradient Boosting muestra un balance sano entre
ambos. La diferencia entre leer las métricas y leer la distancia
entre ellas es lo que separa un modelo juguete de una predicción servible.

---

## 1. El problema y el target

### 1.1 ¿Qué se predice y para qué?

`delivery_days` = días entre `order_purchase_timestamp` (cuando se hizo el pedido) y
`order_delivered_customer_date` (cuando el cliente lo recibió). La utilidad de negocio:
si Olist pudiera predecir esto al momento de la compra, podría informar fechas de
entrega más realistas y gestionar proveedores atrasados antes de que el cliente se queje.

### 1.2 Filtrado y construcción del target

- Solo órdenes con `order_status == 'delivered'` (hay entregas sin fecha pero con estado
  "entregado" — esas no se pueden usar, no hay fecha real contra la cual medir).
- Se descartaron valores negativos (fechas invertidas → error de carga) y outliers
  extremos en el extremo superior.

**Decisión defendible sobre `estimated_days`:** hay una discusión legítima sobre si incluir
como feature el tiempo estimado de entrega que Olist le muestra al cliente. Es válido
argumentar que esa estimación está al menos parcialmente correlacionada con el resultado
real porque la empresa la calcula con base en históricos. Lo que NO se usó (y por qué)
es `order_estimated_delivery_date` como fecha absoluta: la distancia en días ya contempla
el viaje, y usar la fecha cruda contaminaría con señales cíclicas duplicadas.

---

## 2. Features

Las features se construyeron **exclusivamente con información disponible al momento del
pedido**, para que la predicción sea usable como sistema preventivo y no como análisis
a posteriori:

| Feature | Fuente | Qué captura |
|---|---|---|
| `precio_total`, `flete_total` | order_items (suma por orden) | Canales de venta y logística |
| `peso_total_g` | products (suma por orden) | Costo/volumen logístico |
| `n_items` | order_items (conteo) | Complejidad del armado |
| `purchase_month`, `purchase_dow` | orders (fechas) | Estacionalidad (cyber monday, liquidaciones) |
| `estimated_days` | orders (fecha estimada - fecha compra) | Promesa del vendedor |
| `installments` | order_payments | Calidad del pedido, perfil de compra |
| `customer_state` (encodificado) | customers | Estado de origen del cliente |

**Por qué `purchase_month` en vez del timestamp crudo:** el timestamp absoluto mezcla
tres señales distintas (mes, día del mes, día de la semana) en un solo número sin
linealidad útil — el día 30 no es "3 veces el día 10" para el negocio. Separar la
estacionalidad anual (mes) de la semanal (día de la semana) les da a los árboles
divisiones que son interpretables y estables.

**LabelEncoder para `customer_state`:** es un encoding ordinal que asigna un número
arbitrario a cada estado. Funciona con árboles porque ellos dividen por umbrales dentro de
una sola variable — para estos modelos, la numeración arbitraria es inocua (solo se corta
en algún punto del rango), aunque sería un error usarla en modelos de distancia como
K-Means/PCA (ver misión 1).

---

## 3. Datos y onda del problema

- **Split:** 80/20 con `random_state=42` (fijo, para reproducción y comparación justa
  entre experimentos).
- **Regresiones corridas:** Decision Tree, Random Forest y Gradient Boosting, todos con
  cross-validation de 5 folds para estimar el error de forma menos optimista que un solo
  split.
- **Qué se midió:** R² (proporción de varianza explicada) y MAE (error promedio absoluto,
  en días — fácil de explicar al negocio).

---

## 4. Resultados y diagnóstico

### 4.1 Tabla de resultados reales (ejecución 2026)

| Modelo | R² Train | R² Test | MAE Train | MAE Test (días) | Gap R² |
|---|---|---|---|---|---|
| Decision Tree | 0.999 | -0.304 | 0.02 | 6.54 | 1.30 |
| Random Forest | 0.907 | 0.348 | 1.78 | 4.75 | 0.56 |
| Gradient Boosting | 0.343 | 0.344 | 4.76 | 4.74 | -0.001 |

(Métricas con seed 42, split 80/20. CV 5-folds de cada modelo: DT 6.60, RF 4.82, GB 4.78.)

### 4.2 El veredicto: Gradient Boosting gana porque generaliza

Gradient Boosting fue el único modelo cuyo R² de test (0.344) es prácticamente igual al de
train (0.343): no memoriza, aprende el patrón estable del negocio. Random Forest mostró la
trampa del R²: en train parece un modelo excelente (0.907) y en test se desploma a 0.348 —
gap de 0.56 puntos, síntoma clásico de sobreajuste. **Por qué el árbol simple rinde tan
mal:** un solo árbol corta el espacio de features hasta dejar cada hoja con pocos o
un único ejemplo (bias bajo, varianza altísima); con R² train de 0.999 literalmente se
aprendió de memoria los datos y en test resultó peor que un modelo trivial (R² negativo).

La lección: **Random Forest sobreajusta más que Gradient Boosting** en este dataset. RF
perfecciona los datos vistos (cada árbol la parcial y el promedio memoriza), mientras que
GB, al construir árboles pequeños y lentos que corrigen residuos, sacrifica R² en train a
cambio de estabilidad — y en test, justo donde importa, gana.

### 4.3 Una pista crucial que NO se exageró: la importancia de `estimated_days`

Que `estimated_days` aparezca entre las features más importantes es una señal valiosa:
la estimación de Olist comparte información con la variable objetivo. A futuro habría que
explorar **cuánto de esa correlación es una profecía autocumplida** (el sistema informa al
cliente una fecha, y el negocio se organiza para honrarla) y cuánto es capacidad
predictiva genuina del vendedor. Antes de meterla como feature definitiva en producción,
vale la pena correr un modelo sin ella y cuantificar cuánto se pierde — esa comparación
define si estamos prediciendo la realidad o replicando la promesa.

### 4.4 Sobre la simpleza de las features numéricas

Se trabajó sobre las tablas de datos disponibles tal cual (la versión de este repo no
mezcló el tiempo de preparación del vendedor ni el peso distinto por tipo de logística).
Con features más ricas y mejor preparadas (por ejemplo, embudo logístico completo), los
MAEs podrían bajar más allá de ~4.7 días.

---

## 5. Para memorizar y entender profundo

### 5.1 Qué es el overfitting — la definición que importa

Un modelo sobreajustado **memoriza los datos de entrenamiento en vez de aprender el
patrón general**. En la práctica se detecta midiendo el comportamiento en training y en
test por separado: si el modelo anduvo bien en training y mal en test, el modelo no
aprendió el patrón del fenómeno, aprendió los ejemplos puntuales que vio. El síntoma
clásico es una **brecha (gap) grande entre métrica de training y de test**, como el
colapso del Decision Tree en esta misión.

**Pregunta para defender:** *"¿Cómo sabés que X no está overfitteado?"* — comparando su
R² de training contra el de test: la distancia entre ambos es la prueba, no la magnitud
absoluta de ninguno de los dos. Un modelo con R² test 0.70 y R² train 0.75 es predecible
y sano; un modelo con R² test 0.60 y R² train 0.98 está memorizando.

### 5.2 Por qué los árboles se caracterizan por tener varianza alta

Un árbol libre desciende hasta que cada hoja contiene pocos o un único ejemplo: la
"regla" que aprende es tan fina y tan dependiente de los casos exactos que vio que no
generaliza. Ese es el trade-off **bias-varianza**: los modelos complejos tienen bias bajo
(pocas suposiciones equivocadas) pero varianza alta (resultados que bailan ante datos
nuevos). El balance entre ambos es el arte del ensamble y de la regularización.

### 5.3 Por qué Random Forest y Gradient Boosting son "menos overfitteados"

- **Random Forest (bagging):** entrena muchos árboles en **submuestras aleatorias con
  reposición** de los datos, y promedia sus predicciones. Cada árbol individual sigue
  sobreajustando, pero sus errores tienden a cancelarse entre sí al promediar. La clave
  de la estabilidad es que **cada árbol ve una vista distinta** de los datos, forzando
  que los errores no se corrijan todos en la misma dirección.
- **Gradient Boosting (boosting):** entrena árboles **secuencialmente**, donde cada árbol
  nuevo aprende a corregir los errores del conjunto anterior. La diferencia con el
  árbol simple es que el proceso es lento (learning rate bajo) y sujeto a regularización.

### 5.4 Qué significa R² y qué significa MAE

- **R²:** cuánta de la varianza de los días de entrega explica el modelo, entre -∞ y 1.
  R²=1 es perfecto (nunca pasa en datos reales).
- **MAE:** en promedio, el modelo se equivoca por X días. Es la métrica más fácil de
  explicarle al negocio y la que muestra el error en la unidad de la pregunta original.

### 5.5 Por qué la validación cruzada mide mejor que un solo train/test

Con un solo split, la suerte decide qué datos caen en test. Con k-fold CV se entrena k
veces, cada fold actúa como test una vez, y se promedian los errores: es una estimación
de error más estable y menos optimista que depender de un solo particionado. El gap
train-test también ayuda a distinguir sesgo (modelo demasiado simple) de varianza (modelo
demasiado flexible).

### 5.6 Para defender el LabelEncoder en árboles

Codificar un state como 0, 1, 2… es ordinal, lo cual es un error conceptual en modelos
basados en distancia (K-Means, PCA, SVM con kernel), pero en árboles la regla de
decisión es "¿x < umbral?" dentro de una sola variable — el orden de los números es
incidental. Es una codificación económica y suficiente para dividir por umbral en este
tipo de modelos.