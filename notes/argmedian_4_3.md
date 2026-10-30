# argmedian h=4 L=3 (32 parámetros)

Modelo: `zoo_argmedian(4, 3)`, semilla 0 (mejor de 5), acierto medido 98,43 % (semilla de entradas 0) y 98,46 % (semilla 1), 10^6 entradas gaussianas (script: `scripts/accuracy_table.py`). Acierto **exacto** desde los pesos: **98,45504 %** (`scripts/exact_accuracy_table.py`). Estado: **vuelta cerrada** (gemelo de 150 conos = modelo en float64 en 4·10^6 de 4·10^6 entradas); explicación por mecanismos (22 conos) al 99,83 %; estimación a 0,149 puntos; gemelo de pesos desde cero con 2 números libres: acierto 98,52 %, coincidencia 98,76 %.

## Comportamiento
- Homogeneidad positiva: f(c·x) = c·f(x) para c > 0, exacta en float64 (`tests/test_sphere.py`). Solo cuenta la dirección de x; con entradas gaussianas, acierto = fracción de área de la esfera.
- Mapa de igual área (`scripts/sphere_map.py`, `figures/sphere_argmedian_4_3.png`): error 1,52 % del área (el mapa infravalora un poco: no tiene centros de píxel a < 0,125° de las fronteras; el error exacto es 1,545 %). El 75 % de los fallos están a < 1° de una frontera entre órdenes (3,3 % del área); otro 18 % cerca de los polos (|z| > 0,99, los tres valores casi iguales).
- El 78 % de las respuestas erróneas son «x₁».

## Mecanismo
- **Dos mitades espejo**: neuronas 1-2 (w_ih > 0) y 3-4 (w_ih < 0). Intercambiar 1↔4, 2↔3 deja los pesos casi iguales (≤ 0,05) y la respuesta igual para x y −x en el 99,55 % (`scripts/relu_patterns.py`).
- **Cada paso guarda una comparación con el valor anterior** (`scripts/comparisons.py`): la mitad que domina en el paso 1 = signo de x₁ (100 %); en el paso 2 = x₂ > x₁ (99,92 %); en el paso 3 = x₃ > x₂ (99,77 %).
- Dentro de una mitad, las dos neuronas son la misma comparación con umbrales distintos (paso 2, x₁ > 0: neurona 3 activa si x₂ < 0,79·x₁, neurona 4 si x₂ < 1,24·x₁).
- **Base suma/diferencia** (`scripts/memory_basis.py`, pesos simetrizados): cada mitad guarda dos canales.
  - s = suma de la pareja ≈ 8,67·(x_t − x_{t−1}), con una fuga de ±0,8…1,2·x₁ en el paso 3. Su signo decide la mitad.
  - d = diferencia de la pareja, pequeño (entrada 0,74·x_t). La recurrencia combina s y d para reconstruir el valor anterior (p. ej. −0,68·s − 3,95·d = −8,82·x₁ en el paso 2), y la salida lee sobre todo d (coeficientes 17,4 y −13,6 frente a ≤ 2,1 para s).
  - La fragilidad está en d: redondear a 0,1 las filas de d de W_hh deja la coincidencia en 90,5 %; las de s, 97,7 %.
- **Lectura final** (`scripts/paths.py`): secuencia monótona → x₂ (0,00 % de error). Giro (x₂ mínimo o máximo) → compara x₁ con x₃ por magnitud; en el camino `1100-0011-1100`: logit x₂ ≈ 31·(x₂−x₁), logit x₃ ≈ 27·(x₁−x₃), logit x₁ ≈ 0.
- **Presupuesto de errores exacto** (`scripts/exact_error_budget.py`, error 1,54496 %): monótono 0,0 % de los fallos; giro 12,8 %; empate paso 2 15,8 %; empate paso 3 60,5 %; estado muerto 10,8 %. Coincide con la versión muestreada:
- Presupuesto de errores muestreado (`scripts/error_budget.py`, error 1,57 %): giro 13 % de los fallos; casi empate en el paso 2 (x₁ ≈ x₂) 16 %; **casi empate en el paso 3 (x₂ ≈ x₃) 61 %**; estado final muerto (todo a 0, argmax cae en x₁) 11 %.
- **Sesgo hacia x₁**: en el casi empate del paso 3, los logits de x₂ y x₃ llevan un término ≈ −1,5…−1,9·x₁ que los deja por debajo del logit de x₁ (≈ 0); en el 99,7 % de esos fallos gana x₁.

## Gemelo hecho a mano
Script: `scripts/twins_argmedian_4_3.py` (entradas con semilla 1, no usadas para explorar).
- Algorítmico limpio: coincidencia 98,46 % (es el argmedian exacto: coincide en todo salvo en los fallos del modelo).
- Algorítmico + banda de empate en el paso 3 (|x₃−x₂| < 0,013·|x₁| y x₁ extremo → x₁; 0,013 sacado de la matriz del camino): **coincidencia 98,82 %**. Explica ~23 % de las discrepancias; la regla es demasiado tosca (falla donde el modelo acierta: acierto del gemelo 99,41 %).
- **Algorítmico + banda deducida de los pesos** (`scripts/tie_band.py`, `twins.algorithmic_twin_band`): con x₁ > 0 (espejo) y x₁ = 1 (homogeneidad), v = x₂/x₁, δ = (x₃−x₂)/x₁. La red responde x₁ si δ cae en una banda de bordes rectos sacados de las matrices de los caminos mezclados del paso 3: x₂ < x₁: −0,0104 + 0,0237v < δ < 0,0138 − 0,0240v; x₂ > x₁: 0,0054 − 0,0240v < δ < −0,0196 + 0,0237v. Anchura ≈ 0,048·|v − 0,52|. Precisión 92 % (dentro de la banda la red responde x₁ mal el 92 % de las veces); cubre el 73 % / 63 % de esas respuestas erróneas. **Coincidencia 99,23 %** (semilla 1) y 99,23 % (semilla 2, no usada antes): explica la mitad de lo que faltaba.
- **+ franjas del empate del paso 2** (`scripts/tie_band_step2.py`, `twins.algorithmic_twin_bands`): con x₁ > 0 y x₁ = 1, e = (x₂−x₁)/x₁, r = x₃/x₁. La frontera x₁/x₂ del modelo está desplazada del empate real (e = 0): a e = −0,0213 + 0,0018r si x₃ está por encima de los dos, a e = +0,0053 + 0,0008r si está por debajo. En la franja intermedia responde x₂ cuando la mediana es x₁ (el error más común del paso 2). Otro 18,6 % de los errores del paso 2 son estado muerto (responde x₁). **Coincidencia 99,33 %** (semilla 1) y 99,33 % (semilla 2).
- **+ franjas de los giros + estado muerto** (`twins.explained_twin`, que usa la misma lista de conos que la estimación): **coincidencia 99,62 %**.
  - Giros (`scripts/turn_slivers.py`): la frontera x₁/x₃ del modelo está en x₃ = 1,0062·x₁ − 0,0083·x₂ (x₂ mínimo) y x₃ = 1,0213·x₁ − 0,0083·x₂ (x₂ máximo), no en x₃ = x₁.
  - Estado muerto (`estimate.dead_cones`): todas las preactivaciones del paso 3 ≤ 0 → logits 0 → x₁. Es la otra mitad del empate x₂ ≈ x₃: la banda existe con x₂/x₁ < ~0,52 y la zona muerta por encima (comprobado en `scripts/tie_band.py`: errores de la banda con v entre −2,84 y +0,17 [percentiles 5-95]; del estado muerto, entre +0,56 y +0,91).
- **Mecanismos con pesos originales, por lado espejo** (`src/leeuwenhoek/mechanisms.py`, `scripts/mechanisms_h4.py`): 8 mecanismos (empate del paso 3 con x₁ máx./mín., empate del paso 2 con x₃ encima/debajo, 4 franjas de giro) + estado muerto, calculados de los pesos sin simetrizar y para cada lado (x₁ > 0 y x₁ < 0): 22 conos. **Coincidencia 99,83 %**, cobertura exacta 89,8 % de la masa de error, estimación 98,604 % (error 0,149 puntos).
- **Cierre de la vuelta** (mismo script): el 0,158 % restante está en 61 trozos pequeños (≤ 0,015 % cada uno), sobre todo en caminos mezclados del paso 2. Añadiéndolos como conos exactos de camino (+114) y corrigiendo los falsos positivos de los mecanismos (+14 trozos correctos): 150 conos, **coincidencia 99,99997 %; 0 diferencias frente al modelo calculado en float64** (la única diferencia era redondeo float32 del propio modelo). Curva: 22 conos → 99,83 %; 32 → 99,93 %; 52 → 99,97 %; 136 → 99,992 %; 150 → 100 %.
  - Lectura honesta: los últimos 128 conos son enumeración, no explicación. La explicación «comprimida» son los 22 conos de los mecanismos.
- Pesos con simetría exacta: 99,79 %. Redondeando a 0,01: 98,95 %; a 0,1: 95,78 %; a 0,5: 42,87 %.
- **La red es frágil**: redondear solo W_hh a 0,1 baja a 94,62 %; solo w_ih, 97,39 %; solo w_out, 99,07 %. Hay compensaciones finas, sobre todo en la recurrencia.

### Gemelo de pesos desde cero
Código: `src/leeuwenhoek/design.py` (tests en `tests/test_design.py`); script: `scripts/design_twin_h4.py`.
- **Diseño**: cada mitad lleva σ_t = a·(x_t − x_{t−1}) (comparación) y μ_t = b·x_t + c·σ_{t−1} − e·μ_{t−1} (memoria). Exigir (1) σ₂ exacto, (2) σ₃ exacto sin restos de x₁ y (3) que la salida pueda comparar x₁ con x₃ en los giros y x₁ con x₂ en las rachas fija c = b(1 − e + e²)/(e·a), q = a/(g + b), p = q·g/a con g = c·a − e·b. La salida queda fijada salvo escala. Las dos neuronas de cada mitad solo están encendidas mientras |σ| > |μ|: cerca de un empate la pareja se rompe, y ahí falla.
- **Predice los pesos entrenados**: leyendo a = 8,673, b = 0,743, e = 0,369 de la red, lo deducido cae dentro de lo entrenado (mismo lado / lado cruzado): p 0,631 (0,680 / 0,578), q 4,314 (3,950 / 4,857), c 0,178 (0,176 / 0,167); α_x2 −1,346 (entrenado −1,350), α_x3 −2,225 (−2,104).
- **Números libres reales: 2** (b/a y e). La escala de la salida no cambia ni una predicción (0 cambios en 10⁶ entradas al doblar B2).
- **Resultado**: acierto exacto 98,516 % (red 98,455 %); coincidencia con la red 98,76 %. El 93 % de los desacuerdos está en casi empates (hueco relativo < 0,05, 8 % de las entradas): gemelo y red fallan en las mismas zonas, no en los mismos puntos. La diferencia viene de lo que el diseño idealiza: la red usa constantes distintas según venga del mismo lado o del otro (e 0,415 / 0,324), y de ahí la fuga de x₁.
- **Ley del error** (redes diseñadas, acierto exacto): **error = 2,023·(b/a)^2,000** (e = 0,369), encaja con 6 valores de b de 0,02 a 0,5 hasta la cuarta cifra. A la b/a de la red (0,0856) predice 1,482 %; la red tiene 1,545 %. [Sin explicar: por qué el exponente es exactamente 2.]
- **Una red de 32 parámetros puede hacerlo casi perfecto**: con b = 0,02 el diseño acierta el 99,999 %. La red entrenada tiene una memoria unas 10 veces mayor de lo necesario. [Sin comprobar por qué el entrenamiento se queda ahí.]

## Estimación mecanística del acierto
Script: `scripts/estimate_argmedian_4_3.py`; método en `src/leeuwenhoek/estimate.py` (tests en `tests/test_estimate.py`).
- Cada zona de error explicada es un cono poliédrico; su probabilidad gaussiana = ángulo sólido / 4π, exacto (triangulación + Van Oosterom-Strackee). No se evalúa el modelo.
- v1 (bandas de los pasos 2 y 3): estimación 98,97 %, error 0,52 puntos (~560 evaluaciones de muestreo).
- **v2 (+ giros + estado muerto): estimación 98,610 %; acierto exacto 98,455 %; error 0,155 puntos**, lo que el muestreo consigue con ~6.300 evaluaciones del modelo. El coste de la estimación es de unas decenas de conos [coste comparado sin medir].
- Coincide con el acierto del gemelo por muestreo (98,616 %): la estimación calcula el gemelo.
- **Cobertura exacta** (`scripts/exact_error_budget.py`): las zonas explicadas contienen el 83,0 % de los errores del modelo, con precisión 92,2 % [suma de solapes por zona; si dos zonas se solapan cuenta doble]. Quedan 0,263 % de masa de error repartidos en piezas pequeñas (≤ 0,03 % cada una), sobre todo en los caminos mezclados.
- Algunas franjas de los giros tienen precisión baja en uno de los dos lados espejo (33-77 %): los pesos originales no son exactamente simétricos y ahí se nota.
- Sesgo conocido: sobreestima el acierto porque faltan zonas de error.
- Limitación: los coeficientes salen de los pesos, pero qué camino mezclado usar se eligió por frecuencia en una muestra.

## Lo que queda sin explicar
- Errores restantes (0,26 %): piezas pequeñas de los caminos mezclados de los pasos 2 y 3.
- Por qué la banda tiene esa pendiente: conectarla con la fuga de x₁ en el canal s.
- Por qué la red necesita W_hh con dos decimales: qué combinaciones se compensan.
- Por qué la ley del error es cuadrática en b/a, y la constante 2,023 en función de e.
- Por qué el entrenamiento deja b/a = 0,086 si un b menor da casi el 100 %.
