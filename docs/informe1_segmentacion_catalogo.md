# Misión 1: Recomendaciones Inteligentes — Documentación Completa

## Resumen ejecutivo

Este notebook agrupa productos de Olist en 4 clusters usando K-Means, con PCA para
reducir dimensiones y visualizar el resultado. A diferencia de las misiones 2 y 3, acá
no hay un target que predecir — es aprendizaje **no supervisado**, y el criterio de
éxito es distinto: no se mide con R² o F1, sino con cohesión interna (silhouette score)
e interpretabilidad de negocio. El hallazgo metodológico más importante de esta misión
es que **la elección de escala y transformación de las features es tan decisiva como la
elección del modelo**, algo que no aplica de la misma forma en las misiones supervisadas.

---

## 1. El problema: por qué se clusterizan productos y no clientes

El challenge original permitía elegir entre agrupar "productos o clientes". La decisión
de ir por **productos** se basó en un hallazgo estructural del dataset: los datos de Olist
muestran que la gran mayoría de los clientes compra una sola vez. Esto significa que casi
cualquier feature de comportamiento de cliente (frecuencia de compra, recencia, gasto en
el tiempo) tendría prácticamente cero varianza real entre clientes — casi todos tendrían
"frecuencia = 1", sin nada que un algoritmo de clustering pueda usar para separar grupos
con sentido.

Los productos, en cambio, sí tienen variación real y estable (precio, peso, categoría,
volumen de ventas), y encajan mejor con el ejemplo del propio challenge ("compra una
cafetera, ¿le sugerís café premium o una motosierra?") — un problema de similitud entre
productos, no de comportamiento de clientes.

---

## 2. Pipeline de datos

### 2.1 Agregación de `order_items` por producto

Cada fila de `order_items` es una venta individual. Se agregó por `product_id`:
- `precio_promedio` (promedio de `price`)
- `flete_promedio` (promedio de `freight_value`)
- `veces_vendido` (conteo de ventas — proxy de popularidad)

### 2.2 Merge con `products` y `category_translation`

Se sumó peso, dimensiones y categoría a cada producto. El volumen se calculó
multiplicando las tres dimensiones (`largo × alto × ancho`) — nunca sumar dimensiones
directamente, siempre calcular el volumen por fila primero.

**Nulos encontrados:** menores al 2% en peso, volumen y categoría — proporción chica, se
descartan directo sin investigación adicional.

### 2.3 El problema de escala y asimetría — la decisión técnica central de esta misión

**Por qué esto no aplica igual en la misión de entregas:** los modelos de árboles dividen
el espacio por umbrales, son insensibles a la escala de las variables. **K-Means y PCA sí
miden distancia real entre puntos** — si una feature va de 5 a 5,000 y otra de 0.1 a 40,
la de números más grandes domina la distancia euclidiana solo por magnitud, no porque sea
más importante.

Antes de decidir la transformación se verificó la asimetría de cada feature comparando la
mediana contra el máximo:

| Feature | Mediana | Máximo | Razón máximo/mediana |
|---|---|---|---|
| `precio_promedio` | 79.80 | 6,735.00 | ~84x |
| `flete_promedio` | 16.73 | 409.68 | ~24x |
| `veces_vendido` | 1 | 527 | ~527x |
| `product_weight_g` | 700 | 40,425 | ~58x |
| `volumen_cm3` | 6,859 | 296,208 | ~43x |

Las cinco features tenían cola larga fuerte y sistemática. **Por qué `StandardScaler`
solo no alcanza:** centra y escala, pero **no cambia la forma** de la distribución — si
está sesgada antes, sigue sesgada después. Con esa asimetría, K-Means terminaría tratando
a los superventas y productos carísimos como outliers aislados, forzando a que los
clusters se organicen alrededor de esos extremos en vez de las diferencias reales entre
productos típicos.

**Solución aplicada:** `log1p` (log(x+1), soporta ceros sin error matemático) sobre las 5
features, **antes** de `StandardScaler`. El log corrige la **forma** (asimetría), el
StandardScaler corrige la **escala** — son problemas distintos, se resuelven en orden.

### 2.4 Pesos en cero

Productos con `product_weight_g == 0` — error de carga físicamente imposible, se
descartan. Mismo criterio que en la misión de entregas.

---

## 3. PCA (Análisis de Componentes Principales)

### 3.1 Verificar la varianza explicada antes de reducir a 2D

Reducir a 2 dimensiones para poder graficar es útil, pero si esas 2 dimensiones capturan
poca varianza real, el gráfico resultante sería visualmente prolijo pero engañoso. Se
ajustó PCA con todos los componentes y se revisó la varianza individual y acumulada de
cada uno, **antes** de decidir reducir.

### 3.2 Qué representa cada componente

- **Componente 1:** eje de "tamaño/magnitud del producto" — separa productos
  chicos/baratos de grandes/caros. Tiene sentido porque precio, peso y volumen tienden a
  moverse juntos en un catálogo real.
- **Componente 2:** eje de "popularidad" — independiente del tamaño, separa productos de
  alta rotación de los de baja rotación.

---

## 4. Elección de K (número de clusters)

### 4.1 Dos métricas, una posible tensión entre ellas

- **Elbow method (inercia):** mide qué tan compactos son los clusters — baja
  monótonamente al agregar clusters, así que se busca el punto donde agregar más deja de
  reducir la inercia de forma sustancial.
- **Silhouette score:** mide qué tan bien separados y cohesivos son los clusters.

### 4.2 El razonamiento completo para elegir K

Se calcularon inercia y silhouette para K de 2 a 10, y la reducción porcentual de inercia
entre pasos consecutivos. El salto grande de reducción de inercia se sostiene hasta 3→4 y
después cae a la mitad — el "codo" real está ahí. K=4 resultó el punto de mejor balance:
el codo de inercia se estabiliza justo ahí, y el silhouette todavía se mantiene por encima
de los valores de K≥6.

**Nota honesta sobre el silhouette general (0.2-0.3):** según convención está en zona de
estructura "débil a moderada" — no hay clusters naturalmente separados con fronteras
nítidas, lo cual es esperable porque las features de producto son continuas (no hay
categorías con bordes claros en precio o peso).

---

## 5. Resultado final: K-Means con K=4

### 5.1 Verificación del balance de clusters

Sin cluster minúsculo aislado (el más chico no baja de ~11%) — buena señal de que K=4 no
dejó ningún grupo funcionando como "cajón de outliers".

### 5.2 Interpretación de cada cluster

Los clusters se interpretaron usando la categoría **solo para nombrar** (nunca como input
del modelo), y leyendo los promedios de las features en unidades reales (sin log ni
escalar). Un hallazgo interesante esperable es que productos de la misma categoría
dominante pueden caer en clusters de comportamiento distinto — la categoría nunca entró
como input numérico, así que eso muestra que las features numéricas capturan algo real más
allá de "qué tipo de producto es".

### 5.3 Aclaración sobre "alta rotación"

`veces_vendido` cuenta **ventas totales** del producto, no compras repetidas del mismo
cliente. "Alta rotación" no significa que existan clientes fieles al producto — significa
que **muchos clientes distintos, cada uno comprando una sola vez, eligieron ese producto**.
Es popularidad amplia, no lealtad repetida.

---

## 6. Limitaciones honestas

**Sobre el caso de uso de recomendaciones:** el ejemplo del challenge ("compra una
cafetera, ¿le sugerís café o una motosierra?") requiere capturar **afinidad funcional**
(qué se compra junto, o qué es funcionalmente complementario). Lo que se construyó acá
agrupa por **segmento de mercado** (tamaño, precio, popularidad), que es un problema
distinto. Sirve para decisiones como estrategia de precios o gestión de inventario por
segmento, pero no resuelve directamente "si compraste X, te recomiendo Y" sin una capa
adicional de análisis (por ejemplo, co-ocurrencia de productos dentro de un mismo pedido).

**Otras limitaciones menores:**
- La visualización 2D representa el porcentaje de varianza capturado por 2 componentes,
  no el 100%.
- El silhouette general (0.2-0.3) indica estructura moderada.
- No se probaron otros algoritmos de clustering (DBSCAN, jerárquico).

---

## 7. Para memorizar y entender profundo

### 7.1 Por qué K-Means necesita escalado (y los árboles no)

K-Means asigna cada punto al centroide más cercano midiendo **distancia euclidiana**. Si
las dimensiones tienen escalas muy distintas, la de mayor magnitud numérica domina el
cálculo de distancia sin importar su relevancia real. Los árboles de decisión dividen el
espacio por umbrales independientes en cada dimensión, sin nunca calcular una distancia
combinada entre variables — por eso son inmunes al problema de escala.

**Pregunta para defender:** *"¿Por qué aplicaste log ANTES del StandardScaler y no al
revés?"* — porque resuelven problemas distintos: el log corrige la **forma** de la
distribución (asimetría, cola larga), el StandardScaler corrige la **escala**. Aplicar
solo StandardScaler sobre una distribución muy asimétrica mantiene la asimetría, solo que
centrada.

### 7.2 Qué es PCA en términos simples pero precisos

PCA busca las direcciones (combinaciones lineales de las features originales) donde los
datos tienen **mayor varianza**. La primera componente es la dirección de mayor varianza
posible; la segunda es la siguiente dirección de mayor varianza que sea **perpendicular**
(no correlacionada) a la primera. El resultado es un nuevo sistema de coordenadas donde
las primeras componentes concentran la mayor parte de la información, permitiendo reducir
dimensiones con la menor pérdida posible.

**Por qué los componentes "significan algo":** las componentes de PCA no vienen con
nombre — hay que interpretarlas mirando qué features originales contribuyen más a cada
una o, como se hizo acá, observando el patrón de los datos graficados y comparando contra
la tabla de promedios por cluster.

### 7.3 Elbow method vs. silhouette score

- **Inercia (elbow):** suma de las distancias al cuadrado entre cada punto y el centroide
  de su cluster. Siempre baja al aumentar K, así que se busca el punto donde la curva de
  mejora se aplana ("el codo").
- **Silhouette:** para cada punto, compara qué tan cerca está de los puntos de su propio
  cluster contra qué tan cerca está del cluster más próximo distinto al suyo. Va de -1 a 1.

**Por qué pueden estar en tensión:** la inercia siempre "prefiere" más clusters
(matemáticamente inevitable), mientras que el silhouette puede preferir menos clusters si
eso da fronteras más limpias. El silhouette más alto no siempre es la mejor elección
práctica — hay que combinarlo con criterio de negocio.

### 7.4 Qué es `log1p` y por qué no simplemente `log`

`log1p(x) = log(x + 1)`. La razón técnica es que `log(0)` es matemáticamente indefinido
(no da un número real) — y varias features pueden tener valores en 0. `log1p` evita ese
error manteniendo la propiedad de comprimir la cola larga de valores grandes sin
distorsionar los valores pequeños cercanos a 0.

### 7.5 Por qué esta misión no tiene "R²" ni "F1"

Misiones 2 y 3 son **aprendizaje supervisado**: existe una respuesta correcta conocida
(el tiempo de entrega real, la calificación real) contra la cual comparar. Esta misión es
**aprendizaje no supervisado**: no hay una respuesta correcta de qué cluster debería
tener cada producto. Por eso las métricas de evaluación son distintas: en vez de comparar
contra una verdad conocida, se mide **cohesión interna** (silhouette, inercia) y se valida
la utilidad del resultado con **interpretación de negocio**.

**Pregunta para defender:** *"¿Cómo sabés que los clusters son 'buenos' si no hay una
respuesta correcta?"* — la validación es combinada: matemática (silhouette score, balance
de tamaños entre clusters, verificación de que no hay outliers aislados dominando un
cluster) e interpretativa (los perfiles resultantes tienen coherencia de negocio
explicable, en vez de ser agrupaciones arbitrarias sin lectura posible).