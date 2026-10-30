# Leeuwenhoek: qué hicimos con argmedian h=4 (para entenderlo y contarlo)

Escrito el 2026-10-30 para que Xabier lo entienda y lo cuente con sus palabras. Cada cifra sale de un script del repo (se indica cuál). Los detalles técnicos completos están en `notes/argmedian_4_3.md`.

---

## 1. El contexto en cinco frases

1. **ARC** (Alignment Research Center) es un laboratorio de seguridad de IA. En enero de 2026 publicó **AlgZoo**: un «zoo» de redes neuronales diminutas, de 8 a 1.408 parámetros, entrenadas para tareas sencillas.
2. Su argumento: si no somos capaces de **entender del todo** una red de 32 números, tampoco entenderemos un modelo de lenguaje con miles de millones.
3. Para ARC, «entender» no es contar una historia bonita: es poder **calcular cuánto acierta la red a partir de sus pesos, sin probarla con ejemplos**. A eso lo llaman *estimación mecanística*.
4. Casi toda la gente que trabaja con AlgZoo está en una familia (2nd argmax). Las otras tres (argmedian, median y longest cycle) estaban sin explicar en público [comprobado el 2026-10-30; hay que repetir la búsqueda antes de publicar].
5. **Leeuwenhoek** (por el holandés que vio los primeros microorganismos con un microscopio) es nuestro proyecto: explicar esas familias, empezando por la más pequeña que funciona.

## 2. El espécimen

- **Tarea (argmedian):** recibe tres números (x₁, x₂, x₃) y dice **en qué posición está la mediana**, es decir, el del medio. Ejemplo: con (0,9; −0,4; 0,5) la mediana es 0,5, que está en la posición 3, así que la respuesta correcta es «x₃».
- **Red:** una **RNN** de 4 neuronas y **32 parámetros**. Acierta el **98,46 %** de las veces.
- **Entradas:** números aleatorios con distribución **gaussiana** (normal estándar), como en el entrenamiento.

## 3. Vocabulario (cada término, una vez)

| Término | Qué significa |
|---|---|
| **Parámetro / peso** | Un número de la red que se ajusta al entrenar. Esta red tiene 32. |
| **Neurona** | Una casilla que calcula una suma ponderada de sus entradas y le aplica la ReLU. |
| **ReLU** | Deja pasar un número si es positivo y lo pone a 0 si es negativo. Una neurona está «encendida» si su salida es > 0. |
| **Preactivación** | El valor de la neurona *antes* de la ReLU. Si es > 0, la neurona se enciende. |
| **RNN (red recurrente)** | Lee los números de uno en uno y en cada paso mezcla el número nuevo con lo que recordaba del paso anterior (su **estado oculto**, las 4 neuronas). |
| **Logit** | La puntuación final de cada posición. La red responde la posición con el logit más alto; si empatan, la primera (x₁). |
| **Sesgo (bias)** | Un número que se suma siempre. Esta red no tiene, y eso importa (ver «homogeneidad»). |
| **Homogeneidad** | Si multiplicas la entrada por 2, todo dentro de la red se multiplica por 2 y la respuesta no cambia. Solo importa la **dirección** de (x₁, x₂, x₃), no su tamaño. |
| **Patrón ReLU** | Qué neuronas están encendidas en un paso, escrito con 4 bits: `1100` = la 1 y la 2. |
| **Camino de activación** | La secuencia de patrones en los 3 pasos, por ejemplo `1100-0011-1100`. **Dentro de un camino la red es lineal**: solo suma y multiplica. Hay 70 caminos posibles. |
| **Lineal** | Que se puede escribir como sumas de números por constantes. Dentro de un camino, cada logit es «a·x₁ + b·x₂ + c·x₃». |
| **Casi empate** | Dos de los números muy cercanos. Ahí está casi todo el error. |
| **Gemelo** | Una copia de la red construida *desde la explicación*, no copiando los pesos. Si se comporta igual que la red, la explicación es completa. |
| **La vuelta** | Nuestra regla: una explicación solo cuenta si el gemelo coincide con la red **entrada a entrada**. |
| **Coincidencia** | % de entradas en que gemelo y red dan la misma respuesta (aunque las dos se equivoquen). No es lo mismo que el acierto. |
| **Cono** | Una zona del espacio delimitada por planos que pasan por el origen. Por la homogeneidad, todas las zonas de error son conos. Su probabilidad gaussiana se calcula exacta, con una fórmula de geometría. |
| **Muestrear** | Probar la red con muchos ejemplos aleatorios y contar aciertos. Es lo que ARC quiere *batir*. |
| **Error típico** | El ruido de una cifra muestreada. Con 10⁶ ejemplos, ±0,012 puntos. Una diferencia menor que 2-3 errores típicos es ruido. |
| **Primer / segundo orden** | Un error de primer orden necesita *un* casi empate; uno de segundo orden, *dos* a la vez (es más raro). |

## 4. Lo que hicimos, paso a paso

### Paso 1. Montaje: reproducir la red
Reescribí la red a mano (en numpy) y un test comprueba que da **exactamente** los mismos números que la original de ARC. Así todo lo de después se apoya en algo verificado. Reproducimos los aciertos: h=2 41 %, h=4 98,46 %, h=8 99,95 %.
*Scripts: `src/leeuwenhoek/forward.py`, `tests/test_forward.py`, `scripts/accuracy_table.py`.*

### Paso 2. Primera pista: solo importa la dirección
Como no hay sesgos, la respuesta solo depende de la dirección de la entrada. Con entradas gaussianas, todas las direcciones son igual de probables, así que **el acierto es una fracción del área de una esfera**. Dibujamos esa esfera en un mapa en el que cada píxel tiene la misma probabilidad (proyección de igual área): **el 75 % de los errores están pegados a las fronteras donde dos números empatan**, y el 78 % de las respuestas erróneas son «x₁».
*Scripts: `src/leeuwenhoek/sphere.py`, `scripts/sphere_map.py`.*

### Paso 3. Qué hace: sube o baja
Las 4 neuronas forman **dos mitades espejo**: las neuronas 1-2 se encienden cuando el número **sube** respecto al anterior y las 3-4 cuando **baja** (en el paso 1, «el anterior» es 0). Coincide con la comparación real en el 100 %, 99,92 % y 99,77 % de las entradas en los pasos 1, 2 y 3.
- **Racha** (sube y sube, o baja y baja): la mediana es x₂, siempre. Error: 0,00 %.
- **Giro** (baja y sube: x₂ es el menor; sube y baja: x₂ es el mayor): la red compara **x₁ con x₃** por su tamaño.
*Scripts: `scripts/relu_patterns.py`, `scripts/comparisons.py`, `scripts/paths.py`.*

### Paso 4. Cómo lo recuerda: dos canales
Si miramos cada pareja de neuronas como **suma (s)** y **diferencia (d)**:
- **s = canal de comparación**: s ≈ 8,67 × (número actual − anterior). Su signo decide qué mitad se enciende.
- **d = canal de memoria**: es pequeño, pero permite **reconstruir el número anterior** (para poder comparar el siguiente) y es lo que lee la salida.
La red es **frágil en la memoria**: si redondeas esos pesos a un decimal, el acierto cae al 90 %.
*Script: `scripts/memory_basis.py`.*

### Paso 5. Por qué falla: la pareja se rompe
Cerca de un empate, la comparación (s) se hace más pequeña que la memoria (d) y **la pareja de neuronas se rompe**: queda encendida una neurona de cada mitad (camino «mezclado», patrón `0101`), o ninguna (**estado muerto**). En los dos casos la salida pierde la información y **gana x₁ por defecto**. Por eso hay tantos errores «x₁».
*Scripts: `scripts/error_budget.py`, `scripts/tie_band.py`, `scripts/tie_band_step2.py`.*

### Paso 6. La vuelta: el gemelo
Construimos un gemelo que aplica la regla «sube/baja; en los giros compara x₁ con x₃» y añade las **zonas de error** que sacamos de los pesos (la banda del empate, las fronteras desplazadas y el estado muerto).
- Con 22 zonas (conos) explicadas coincide con la red en el **99,83 %** de las entradas.
- Con 150 conos coincide en **4.000.000 de 4.000.000**. Pero los últimos 128 ya son copiar el cálculo trozo a trozo, no explicarlo. **La explicación de verdad son los 22.**
*Código: `src/leeuwenhoek/mechanisms.py`, `src/leeuwenhoek/twins.py`; script: `scripts/mechanisms_h4.py`.*

### Paso 7. El acierto exacto, sin probar la red
Recorriendo los 70 caminos y calculando el área exacta de cada trozo, sacamos de los pesos el acierto exacto: **98,455038 %**. Es fuerza bruta, no explicación, pero da la cifra exacta contra la que medir.
*Código: `src/leeuwenhoek/exact.py`, `src/leeuwenhoek/cones.py`; script: `scripts/exact_accuracy_table.py`.*

### Paso 8. La estimación mecanística (la métrica de ARC)
Sumando la probabilidad de las zonas de error explicadas, **sin ejecutar la red**, estimamos el acierto. Primero nos quedamos a 0,52 puntos del exacto y al final a **0,149 puntos**. Cuanto más explicas, más cerca te quedas.
*Script: `scripts/estimate_argmedian_4_3.py`.*

### Paso 9. El gemelo desde cero
Escribí las condiciones que debe cumplir la red: «la comparación es exacta», «la memoria guarda lo justo para reconstruir el número anterior» y «la salida aplica la regla de la mediana». Con eso **todos los pesos quedan fijados salvo dos números**: **b** (el tamaño de la memoria) y **e** (cuánto olvida). Leyendo b y e de la red real, el diseño **predice sus demás pesos**, y acierta el 98,52 % (la red, el 98,455 %).
*Código: `src/leeuwenhoek/design.py`; script: `scripts/design_twin_h4.py`.*

### Paso 10. La ley del error
Con el diseño se deduce, con papel y lápiz:

> **error = (√3 / 2π) · (b / (e·a))²** ≈ 0,276 · (b/(e·a))²

donde **a** es el tamaño de la comparación (8,67), **b** el de la memoria (0,74) y **e** cuánto olvida (0,37).
- **Por qué el cuadrado:** en la zona rota, la salida **intercambia** los dos canales (lee la memoria donde debería leer la comparación). Los dos son pequeños en proporción b/a, así que la frontera se desplaza b/a × b/a.
- **Por qué √3/(2π):** sale de la probabilidad gaussiana de caer en esa franja estrecha.
- **La hipótesis que falló:** primero pensé que los errores estarían donde los tres números son casi iguales. Los datos lo descartaron.
- **Comprobación:** contra el cálculo exacto cuadra hasta la cuarta cifra para cualquier e.
- **Consecuencia:** con una memoria más pequeña, **la misma red de 32 parámetros acertaría el 99,999 %**. El entrenamiento se quedó con una memoria unas 10 veces mayor de lo necesario (todavía no sabemos por qué).
*Scripts: `scripts/law_geometry.py`, `scripts/law_pieces.py`.*

### Paso 11. Los 0,064 puntos que faltaban
Para la red real, la ley da 1,481 % y el error real es 1,545 %. Esa diferencia es **la resta de efectos grandes de signo contrario**: la red real tiene la zona de «x₁ por defecto» **más estrecha** que el diseño (−0,38 puntos), pero sus comparaciones **dejan pasar restos** del número x₁ y eso crea errores nuevos (+0,20 en los giros, +0,13 y +0,10 en los empates del paso 2). Además, **sus imperfecciones se compensan entre sí**: cada una sola dispararía el error.
*Scripts: `scripts/gap_h4.py`, `scripts/gap_formula.py`.*

## 5. Lo que salió mal (y por qué lo contamos)

- Puse las fronteras del mapa en el sitio equivocado (60° en vez de 30°). Lo vi porque salían fallos imposibles.
- Una fórmula medía la distancia al centro en vez de a la frontera.
- Al reflejar los caminos invertí el orden de los pasos.
- El mapa de la esfera **infravalora** los errores muy finos: en h=8 daba 8 veces menos error del real.
- La hipótesis de los polos (paso 10) era falsa.

Contarlo da credibilidad: se ve cómo se comprueba cada cosa.

## 6. Lo que falta

- La **contabilidad de sorpresa** de ARC (*surprise accounting*): medir cuántos «bits» de explicación hacen falta.
- Un término de segundo orden de unos 0,1 puntos (paso 11).
- Por qué el entrenamiento deja una memoria tan grande.
- Otros tamaños (h=2, h=8) y otras semillas.

## 7. En una frase

**La red recuerda si cada número sube o baja respecto al anterior; cuando dos números casi empatan, ese recuerdo se rompe y responde x₁; y cuánto falla se puede calcular con una fórmula sacada de sus pesos.**

## 8. Dónde mirar

- **Cuaderno de Colab** (`notebooks/h4.ipynb`): mueves x₁, x₂ y x₃ y ves las lámparas, el plano de respuestas y la lupa.
- **Nota técnica completa:** `notes/argmedian_4_3.md`.
- **Brief del post:** `docs/post-h4-brief.md`.
