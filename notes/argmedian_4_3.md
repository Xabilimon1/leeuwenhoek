# argmedian h=4 L=3 (32 parámetros)

Modelo: `zoo_argmedian(4, 3)`, semilla 0 (mejor de 5), acierto medido 98,43 % (semilla de entradas 0) y 98,46 % (semilla 1), 10^6 entradas gaussianas (script: `scripts/accuracy_table.py`). Acierto **exacto** desde los pesos: **98,45504 %** (`scripts/exact_accuracy_table.py`). Estado: **mecanismo cualitativo cerrado; gemelo al 99,62 %; estimación a 0,155 puntos del acierto exacto.**

## Comportamiento
- Homogeneidad positiva: f(c·x) = c·f(x) para c > 0, exacta en float64 (`tests/test_sphere.py`). Solo cuenta la dirección de x; con entradas gaussianas, acierto = fracción de área de la esfera.
- Mapa de igual área (`scripts/sphere_map.py`, `figures/sphere_argmedian_4_3.png`): error 1,52 % del área. El 75 % de los fallos están a < 1° de una frontera entre órdenes (3,3 % del área); otro 18 % cerca de los polos (|z| > 0,99, los tres valores casi iguales).
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
  - Estado muerto (`estimate.dead_cones`): todas las preactivaciones del paso 3 ≤ 0 → logits 0 → x₁. Es la otra mitad del empate x₂ ≈ x₃: la banda existe con x₂/x₁ < ~0,52 y la zona muerta por encima [lectura de las desigualdades de los pesos; la frontera exacta entre ambas sin medir].
- Pesos con simetría exacta: 99,79 %. Redondeando a 0,01: 98,95 %; a 0,1: 95,78 %; a 0,5: 42,87 %.
- **La red es frágil**: redondear solo W_hh a 0,1 baja a 94,62 %; solo w_ih, 97,39 %; solo w_out, 99,07 %. Hay compensaciones finas, sobre todo en la recurrencia.

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
- Gemelo de pesos construido desde cero (sin partir de los pesos entrenados).
