# argmedian h=4 L=3 (32 parámetros)

Modelo: `zoo_argmedian(4, 3)`, semilla 0 (mejor de 5), acierto medido 98,43 % (semilla de entradas 0) y 98,46 % (semilla 1), 10^6 entradas gaussianas (script: `scripts/accuracy_table.py`). Estado: **mecanismo cualitativo cerrado; gemelo incompleto; sin estimación.**

## Comportamiento
- Homogeneidad positiva: f(c·x) = c·f(x) para c > 0, exacta en float64 (`tests/test_sphere.py`). Solo cuenta la dirección de x; con entradas gaussianas, acierto = fracción de área de la esfera.
- Mapa de igual área (`scripts/sphere_map.py`, `figures/sphere_argmedian_4_3.png`): error 1,52 % del área. El 75 % de los fallos están a < 1° de una frontera entre órdenes (3,3 % del área); otro 18 % cerca de los polos (|z| > 0,99, los tres valores casi iguales).
- El 78 % de las respuestas erróneas son «x₁».

## Mecanismo
- **Dos mitades espejo**: neuronas 1-2 (w_ih > 0) y 3-4 (w_ih < 0). Intercambiar 1↔4, 2↔3 deja los pesos casi iguales (≤ 0,05) y la respuesta igual para x y −x en el 99,55 % (`scripts/relu_patterns.py`).
- **Cada paso guarda una comparación con el valor anterior** (`scripts/comparisons.py`): la mitad que domina en el paso 1 = signo de x₁ (100 %); en el paso 2 = x₂ > x₁ (99,92 %); en el paso 3 = x₃ > x₂ (99,77 %).
- Dentro de una mitad, las dos neuronas son la misma comparación con umbrales distintos (paso 2, x₁ > 0: neurona 3 activa si x₂ < 0,79·x₁, neurona 4 si x₂ < 1,24·x₁). [Interpretación sin comprobar: esto deja una memoria 2D que permite recuperar x₁ en el paso 3.]
- **Lectura final** (`scripts/paths.py`): secuencia monótona → x₂ (0,00 % de error). Giro (x₂ mínimo o máximo) → compara x₁ con x₃ por magnitud; en el camino `1100-0011-1100`: logit x₂ ≈ 31·(x₂−x₁), logit x₃ ≈ 27·(x₁−x₃), logit x₁ ≈ 0.
- **Presupuesto de errores** (`scripts/error_budget.py`, error 1,57 %): giro 13 % de los fallos; casi empate en el paso 2 (x₁ ≈ x₂) 16 %; **casi empate en el paso 3 (x₂ ≈ x₃) 61 %**; estado final muerto (todo a 0, argmax cae en x₁) 11 %.
- **Sesgo hacia x₁**: en el casi empate del paso 3, los logits de x₂ y x₃ llevan un término ≈ −1,5…−1,9·x₁ que los deja por debajo del logit de x₁ (≈ 0); en el 99,7 % de esos fallos gana x₁.

## Gemelo hecho a mano
Script: `scripts/twins_argmedian_4_3.py` (entradas con semilla 1, no usadas para explorar).
- Algorítmico limpio: coincidencia 98,46 % (es el argmedian exacto: coincide en todo salvo en los fallos del modelo).
- Algorítmico + banda de empate en el paso 3 (|x₃−x₂| < 0,013·|x₁| y x₁ extremo → x₁; 0,013 sacado de la matriz del camino): **coincidencia 98,82 %**. Explica ~23 % de las discrepancias; la regla es demasiado tosca (falla donde el modelo acierta: acierto del gemelo 99,41 %).
- Pesos con simetría exacta: 99,79 %. Redondeando a 0,01: 98,95 %; a 0,1: 95,78 %; a 0,5: 42,87 %.
- **La red es frágil**: redondear solo W_hh a 0,1 baja a 94,62 %; solo w_ih, 97,39 %; solo w_out, 99,07 %. Hay compensaciones finas, sobre todo en la recurrencia.

## Estimación mecanística del acierto
- Pendiente.

## Lo que queda sin explicar
- Forma exacta de las bandas de empate (pasos 2 y 3) y de la zona muerta.
- Por qué la red necesita W_hh con dos decimales: qué combinaciones se compensan.
- Gemelo de pesos construido desde cero (sin partir de los pesos entrenados).
