# argmedian h=4 L=3: las otras semillas (1-4)

Pregunta (de un profesor): ¿el mecanismo de la semilla 0 (`notes/argmedian_4_3.md`) aparece siempre que se entrena desde cero? AlgZoo entrena 5 semillas y publica la mejor (la 0). Aquí se analizan las otras cuatro con las mismas herramientas; no se entrena nada nuevo.

Scripts: `scripts/seeds_h4.py` (todas las cifras; exactas por enumeración de caminos salvo las marcadas como muestreadas: 10^6 entradas, semilla 1; el ajuste del diseño con 200.000), `scripts/sphere_map.py --seed s` (figuras `figures/sphere_argmedian_4_3_seed{1,2,3,4}.png`). Código nuevo: `src/leeuwenhoek/structure.py` (tests en `tests/test_structure.py`, que reproducen las cifras ya publicadas de la semilla 0: 98,455 %, 77 % de fallos «x1», 10,8 % de estado muerto, umbrales 0,79 y 1,24).

**Respuesta corta: no.** Las cuatro comparten con la 0 la comparación «x2 > x1» en el paso 2 y poco más. Ninguna tiene el espejo, ninguna tiene las parejas sube/baja con dos umbrales por mitad, y el diseño de 2 números no describe ni uno de sus fallos. Fallan en regiones abiertas, no en casi empates. Estado: **descriptivo, sin gemelo** (no hay «vuelta» para las semillas 1-4).

## Tabla

| Semilla | Acierto exacto | Fallo responde x1/x2/x3 | Estado muerto (de los fallos) | Fallos en casi empate (hueco rel. < 0,05)* | f(x) = f(−x)* | Espejo (residuo) | Diseño: mejor coincidencia − acierto* |
|---|---|---|---|---|---|---|---|
| 0 | 98,455 % | 77 / 9 / 14 | 10,8 % | 96 % | 99,6 % | 0,004 | +0,67 |
| 1 | 89,283 % | 60 / 17 / 24 | 0 % | 31 % | 84,6 % | 1,03 | +0,11 |
| 2 | 75,901 % | 92 / 8 / 1 | 65,7 % | 13 % | 65,3 % | 1,31 | +2,11 (espurio, ver abajo) |
| 3 | 90,439 % | 36 / 32 / 32 | 17,6 % | 25 % | 84,2 % | 1,41 | +0,06 |
| 4 | 94,516 % | 45 / 12 / 43 | 9,8 % | 47 % | 89,9 % | 1,27 | +0,04 |

\* muestreado. Tasa base del casi empate: 8,3 % de las entradas. «Espejo»: la permutación de neuronas que mejor convierte la red en su propio espejo de signo (0 = exacto; la identidad da 2 = ninguno). «Diseño»: la mejor coincidencia de cualquier red de `design.py` (rejilla de b/a y e) menos el acierto exacto de la semilla; 0 significa que el diseño solo coincide donde la red acierta, es decir, que no explica ningún fallo.

## Lo que comparten las cinco
- **Paso 1 = signo de x1** (100 %): trivial en una red sin sesgos.
- **Paso 2 = «¿x2 > x1?»**: el patrón de activación lo determina en el 96,6 (semilla 0) / 98,4 / 98,2 / 99,9 / 99,3 %. En todas hay al menos una neurona cuyo umbral del paso 2 es x2 ≈ k·x1 con k entre 0,96 y 1,10 para al menos un signo de x1 (semilla 1: 0,99 y 1,10 con x1 > 0; 2: 0,96 con x1 < 0; 3: 1,00 con x1 > 0 y 0,99 con x1 < 0; 4: 0,96 con x1 > 0 y 1,05 con x1 < 0). La 0 es la única que lo hace con una **pareja que encierra el empate** (0,79 y 1,24) en vez de con una neurona puesta en el empate.
- **Paso 3 = «¿x3 > x2?»**, peor: 96,3 / 87,5 / 93,8 / 96,8 / 96,5 %.
- **Estado muerto** (todo a 0 → logits 0 → responde x1) en 4 de 5 (no en la 1); x1 es la respuesta errónea más frecuente en 4 de 5 (en la 4 empata con x3).

## Lo que es solo de la semilla 0
- **El espejo**: ninguna otra tiene permutación de neuronas que la convierta en su espejo (residuo ≥ 1,03 frente a 0,004) ni la simetría de conducta (84,6 / 65,3 / 84,2 / 89,9 % frente a 99,6 %). Todas aciertan distinto según el signo de x1 (semilla 2: 69,3 % con x1 > 0 frente a 82,6 % con x1 < 0; semilla 3: 86,6 % frente a 94,3 %).
- **Los canales suma/diferencia** no se pueden ni definir sin espejo. La estrategia es otra: las semillas 2, 3 y 4 **comparan x3 con x1 directamente con una neurona** (semilla 2, tras 1010-0101: n3 se enciende si x3 > 1,01·x1; semilla 3, tras 1011-0101: n1 si x3 < 1,00·x1; semilla 4, tras 1100-0111: n2 si x3 > 0,84·x1 + 0,27·x2). La 0 nunca compara x3 con x1 en el patrón (sus umbrales del paso 3 son 0,79-1,27·x2 con restos ≤ 0,41·x1): esa comparación la hace la lectura final con el canal de memoria. La 0 comprime a «subió/bajó + memoria»; las otras guardan x1 explícito, o lo pierden.
- **Neuronas lineales**: las semillas 1 y 3 tienen una neurona con w_ih ≈ 0 (0,22 y −0,03) que está encendida casi siempre (99,9 % en el paso 3 de la 1; 99,7 % en el paso 2 de la 3): sin puerta, es un acumulador lineal. En la práctica son redes de 3 neuronas con ReLU más una lineal. La 0 y la 4 no tienen ninguna; la 2 tiene dos neuronas de entrada casi duplicadas (−3,77 y −3,76).
- **El diseño de 2 números** (`design.py`): con las semillas 1, 3 y 4 la mejor coincidencia es su propio acierto (+0,11 / +0,06 / +0,04 puntos) y la red de diseño que la consigue es la mediana casi perfecta (b/a = 0,02; acierto propio 99,99 %): el diseño no describe ni un fallo. Con la 2 salen +2,11 puntos, pero con una red de diseño rota (b/a = 0,10, e = 0,10; acierto propio 69,2 %) que coincide en «responder x1 por defecto»: no es mecanismo compartido. Con la 0: +0,67 (98,76 % con sus propios números b/a = 0,086, e = 0,369; 99,13 % con b/a = 0,080, e = 0,45; por qué la e que mejor coincide no es la leída de los pesos, [sin explicar]).
- **Dónde fallan**: la 0 casi solo en casi empates (96 % de sus fallos con hueco relativo < 0,05; base 8 %). Las otras, en regiones abiertas del mapa (31 / 13 / 25 / 47 %): trozos enteros de un orden, no franjas. En las semillas 1 y 3 los fallos se concentran en las «tapas» de la esfera (|z| > 0,8, los tres valores parecidos y del mismo signo): 55 % y 69 % de sus fallos (base 20 %), y en la 3 el 71 % de esos en la tapa positiva.

## Semilla 1 (89,28 %)
- w_ih = [6,49, −6,21, −8,69, 0,22]. Paso 1: x1 > 0 → 1001; x1 < 0 → 0110. Neurona 4 lineal (ver arriba). Mejor «espejo» 1↔3, residuo 1,03: no hay.
- Paso 2: n2 y n3 son detectores «baja» con dos umbrales (x2 < 0,99·x1 y < 1,10·x1 con x1 > 0; < 0,47 y < 1,23 con x1 < 0), n1 el único «sube» (> 0,62·x1 / > 1,71·x1). Una pareja «baja» sin su pareja «sube».
- Paso 3: el patrón solo determina x3 > x2 en el 87,5 %. Las referencias se alejan de x2: tras 1001-0111, n2 se enciende si x3 < 0,44·x1 + 0,57·x2 (compara con una media) y n3 si x3 < −0,72·x1 + 1,28·x2; tras 0110-1001, n2 si x3 < −5,1·x1 + 0,99·x2. El estado tras el paso 2 tiene valores singulares 14,8 / 3,4 (0110-1001) y 15,6 / 0,75 (1001-0111): cuánta memoria queda depende del lado.
- Fallos: 10,72 %, el 60 % responde x1; ninguno es estado muerto. Por orden: **x1 < x2 < x3 pierde el 23,7 %** (responde x1; trozo mayor 1001-1001-1001, 1,66 % de las entradas) y x3 < x1 < x2 el 15,0 % (responde x3; 1001-1001-0111, 0,94 %). Solo el 31 % en casi empate; el 55 % con |z| > 0,8.
- En prosa: la red detecta si x2 bajó respecto a x1 con dos umbrales y guarda el estado en una neurona lineal; en el paso 3 compara x3 con una mezcla de x1 y x2 en vez de con x2, así que cuando los tres valores se parecen y suben (x1 < x2 < x3, los tres del mismo signo) no distingue el caso y responde x1. Es peor que la 0 porque le falta la mitad «sube» de la pareja y la comparación del paso 3 no está centrada en x2.

## Semilla 2 (75,90 %)
- w_ih = [7,44, −3,77, 7,96, −3,76]: dos neuronas positivas y dos negativas casi iguales (duplicadas, no espejo; residuo 1,31; conducta simétrica solo el 65 %).
- Paso 2 con x1 > 0: n1 y n2 son detectores del **signo de x2** (umbrales −0,02·x1 y −0,00·x1), n3 «sube» con 0,33, n4 «baja» con 1,20; con x1 < 0: n2 «baja» con 0,96 (el único comparador exacto), n3 signo de x2. El patrón determina x2 > 0 en el 99,8 % y x2 > x1 en el 98,2 %.
- Paso 3 tras 1010-0101: n1 y n3 comparan x3 con x1 (x3 > 1,06·x1 − 0,34·x2; x3 > 1,01·x1), n2 con x2 (x3 < 0,96·x2).
- Fallos: 24,10 %, el 92 % responde x1 y el 65,7 % son estado muerto. **Pierde un orden entero: x2 < x3 < x1 (x1 máximo, mediana x3) falla el 100 %** (16,67 % de las entradas; camino 1010-0101-0000, 12,0 %, y otros con final 0000). Con x1 > 0 y x2 < 0, si x2 < x3 < x1 no se enciende nada: n1 y n3 piden x3 > x1, n2 pide x3 < x2 y n4 pide x3 < −1,39·x1 + 2,15·x2 (negativo) → logits 0 → x1. Además, con x1 < 0 y 0,96·x1 < x2 < 0 el estado se borra en el paso 2 (0101-0000-…) y el paso 3 solo ve x3: si x3 > 0 responde x2 (acierta: 6,0 % de las entradas, 0 % de fallo), si x3 < 0 responde x1 (acierta 1 de 3: 5,9 % de las entradas, 66,5 % de fallo).
- Acierto 69,3 % con x1 > 0 frente a 82,6 % con x1 < 0. Solo el 13 % de los fallos en casi empate.
- En prosa: la red mira sobre todo signos (de x1, de x2, de x3 frente a x1) y resuelve bien los órdenes que se deciden con ellos; el caso «x1 es el máximo y x2 < x3» no tiene ninguna neurona que lo represente y cae en el estado muerto, que responde x1. Es la peor porque un sexto de las entradas está perdido por construcción, no por empates.

## Semilla 3 (90,44 %)
- w_ih = [−5,97, 1,28, −4,91, −0,03]. Paso 1: x1 > 0 → 0100 (solo n2, ganancia 1,28: un x1 positivo se guarda débil); x1 < 0 → 1011. Neurona 4 lineal. Residuo de espejo 1,41.
- Paso 2: con x1 > 0, n3 es un comparador exacto (x2 < 1,00·x1) y n1, n2 detectores del signo de x2; con x1 < 0, n1 es el comparador exacto (x2 < 0,99·x1), n2 casi siempre encendida y n3 casi nunca. El patrón determina x2 > x1 en el 99,9 %: la mejor de las cinco.
- Paso 3 tras 1011-0101 (x1 < 0, x2 > x1): n1 si x3 < 1,00·x1 y n3 si x3 < 1,02·x2: **las dos comparaciones, x3 con x1 y x3 con x2, en neuronas separadas**. Tras 0100-1011 y 0100-0111 (x1 > 0): n1 compara con x2 (0,99·x2 y −0,10·x2) y n2, n3 tienen referencias inservibles (−12,5·x1 + 12,1·x2; −4,4·x1 + 7,9·x2). El patrón determina x3 > x2 en el 96,8 %.
- Fallos: 9,56 %, repartidos (respuesta errónea 36 / 32 / 32; por orden entre 3,5 % y 14,2 %); estado muerto 17,6 % (1011-1101-0000: 8,0 % de las entradas, 18,8 % de fallo). El camino 0100-0111-0101 (6,1 %) mezcla a partes iguales los tres órdenes con x2 < x1 y falla el 22 %. Acierto 86,6 % con x1 > 0 frente a 94,3 % con x1 < 0. El 69 % de los fallos con |z| > 0,8 y el 71 % de ellos con z > 0: la tapa positiva del mapa.
- En prosa: con x1 negativo la red funciona casi como un comparador explícito (guarda x1 en una neurona lineal y compara x3 con x1 y con x2 por separado); con x1 positivo solo le queda una neurona débil para recordarlo y las comparaciones del paso 3 se desordenan. Es peor que la 0 porque la mitad x1 > 0 está mal construida y porque un 8 % de las entradas acaba en estado muerto.

## Semilla 4 (94,52 %)
- w_ih = [45,4, 5,95, −6,80, −0,93]: una neurona con ganancia 45. Paso 1: 1100 / 0011, como la 0, pero sin espejo (residuo 1,27; conducta simétrica 89,9 %).
- Paso 2: n1 es el comparador «sube» para los dos signos (x2 > 0,96·x1 con x1 > 0; > 1,05·x1 con x1 < 0); n3 «baja» con umbral flojo (0,63 / 0,25); n2 y n4 casi siempre encendidas con x1 > 0 (n2 también comparador, 1,05, con x1 < 0). Determina x2 > x1 en el 99,3 %.
- Paso 3 tras 1100-0111 (x1 > 0, x2 < 0,96·x1): n1 si x3 > 1,13·x2 y n3 si x3 < 1,16·x2 (una pareja sube/baja sobre x2, como la 0, pero los umbrales no encierran el empate: con x2 < 0 queda una rendija donde no se enciende ninguna); n2 si x3 > 0,84·x1 + 0,27·x2 (compara con ≈ x1). Determina x3 > x2 en el 96,5 %. Valores singulares del estado tras 1100-0111: 172 / 8,3 (la ganancia 45 domina).
- Fallos: 5,48 %; respuesta errónea 45 / 12 / 43; estado muerto 9,8 %; por orden entre 2,6 % y 8,6 %. El 47 % en casi empate (hueco < 0,05) y el 90 % con hueco < 0,2: la más parecida a la 0 en dónde falla, pero con bandas más anchas y manchas en las tapas (61 % con |z| > 0,8).
- En prosa: una neurona enorme hace de comparador «sube» en los dos pasos y otra, floja, de «baja»; la tercera compara x3 con x1 y la cuarta apenas aporta. Se parece a la 0 en que los fallos se pegan a las fronteras, pero sin la pareja simétrica las fronteras están desplazadas unas décimas (0,63, 1,13, 1,16) en vez de centésimas, y de ahí las bandas anchas.

## Lo que esto dice del mecanismo de la semilla 0
- El espejo, las parejas con dos umbrales, los canales suma/diferencia y el diseño de 2 números **no son lo que aprende «el entrenamiento»: son lo que aprendió una semilla de cinco**, y es la mejor por 4 puntos. Los otros cuatro mínimos son redes asimétricas con 3 neuronas útiles y una lineal o duplicada, que pierden regiones enteras.
- Lo que sí parece genérico: una red sin sesgos de 4 neuronas aprende siempre «x2 > x1» en el paso 2 (≥ 96,6 %) y casi siempre «x3 > x2» en el paso 3, y cuando falla por defecto responde x1 (estado muerto).
- La ley del error de la 0 (∝ (b/(e·a))²) presupone el diseño; no se puede aplicar a las otras.

## Avisos
- Las frases «en prosa» interpretan los umbrales impresos por el script; lo comprobado son las cifras (exactas: acierto, reparto de fallos, estado muerto, matrices de camino, umbrales; muestreadas: casi empates, |z|, simetría de conducta, ajuste del diseño).
- Sin gemelo para ninguna semilla: no hay «vuelta». Las descripciones de 1-4 son descriptivas, no explicaciones cerradas.
- Checkpoints parciales sin mirar (pendiente: ¿la semilla 0 pasa por una fase sin espejo antes de encontrarlo?).
- Cinco semillas es una muestra pequeña: «1 de 5 encuentra el espejo» no es una tasa.

## Entrenar aquí (2026-11-19)
Scripts: `scripts/time_training_h4.py` (cronómetro) y `scripts/train_seed_h4.py` (entrenamiento completo con la receta de AlgZoo y comparación con ARC). Puntos del entrenamiento en `data/trained/argmedian_4_3_seed0/` (fuera de git).
- **Una semilla completa (2³⁰ secuencias) tarda 316 s** en el MacBook Air M3 (CPU, 1 hilo; 0,28-0,29 ms por paso, igual con 1 que con 4 hilos). 50 semillas ≈ 4,4 h en serie.
- **La semilla 0 entrenada aquí parte de la misma inicialización exacta que la de ARC** (punto 0 idéntico bit a bit) y **acaba con el mismo mecanismo**: w_ih [3,96, 4,70, −4,71, −3,96] frente a [3,97, 4,70, −4,72, −3,97]; residuo espejo 0,003 (ARC 0,004); acierto exacto 98,600 % (ARC 98,455 %); diferencia máxima de pesos 0,029. No es bit a bit: algo del entrenamiento difiere, seguramente los datos o la aritmética de su GPU [sin verificar].
- Lectura: misma inicialización → mismo mecanismo aunque cambie el resto. Primera pista directa de que **el punto de partida decide el algoritmo** (con los signos de w_ih fijos desde 2²² secuencias en los 5 entrenamientos de ARC, según el contraste de Fable).

## Literatura (búsqueda de Fable, 2026-11-19; informe completo en x-brain `Projects/Leeuwenhoek/Literatura-semillas.md`)
- Universalidad parcial, conocida: Chughtai et al. 2023 (arXiv 2302.03025), Zhong et al. 2023 «The Clock and the Pizza» (2306.17844), Li et al. 2016 «Convergent Learning» (1511.07543), Gurnee et al. 2024 (2401.12181).
- Estrategia por semilla y su relación con el resultado: Juneja et al. 2023 (2205.12411, BERT; lo más cercano), Lubana et al. 2023 (2211.08422), Gross et al. 2024 (2406.11779).
- El inicio decide: Frankle et al. 2020 (1912.05671, 2002.10365), Zhou et al. 2019 (1905.01067, los signos).
- Explicar errores: Zhang et al. 2026 (2609.35686), Geng et al. 2026 (2610.02098). Usar *dónde* falla para separar mecanismos limpios: no encontrado.
- AlgZoo: el README dice «best of 5 random seeds»; Snider (LessWrong, oct. 2026) reentrenó ~8.000 redes de 2nd argmax. Nadie ha estudiado argmedian ni las semillas no publicadas.
- ⚠️ Comentario de «phoenix» (5-8-2026, bajo «Returning to ARC» de Christiano, LessWrong), escrito por IA, repo privado «ATLAS»: afirma «M₄,₃: from-weights accuracy 0.9845503814», que es nuestro acierto exacto de argmedian h=4 con 10 cifras, y que una ley de empates «transfers to argmedian». No verificable; hay que citarlo si publicamos.
- Veredicto: las piezas son conocidas; la combinación en un benchmark real de interpretabilidad (solo la semilla publicada tiene el mecanismo limpio, lo limpio va con el acierto, dónde falla los separa, el inicio decide) parece nueva. Un revisor pedirá: 50-100 semillas, una puntuación numérica de mecanismo, validar el predictor de errores en semillas no vistas, puntos tempranos que predigan el mecanismo final y conectividad entre semillas.

## A lo largo del entrenamiento de ARC (2026-11-19)
Script: `scripts/checkpoints_h4.py` (acierto exacto, signos de w_ih y residuo espejo en cada punto publicado, 2^20 … 2^30).

| Semilla | 2^20 | 2^21 | 2^22 | 2^24 | 2^26 | 2^28 | 2^30 |
|---|---|---|---|---|---|---|---|
| 0: acierto / espejo | 80,46 % / 0,545 | 92,17 % / 0,155 | 95,46 % / 0,075 | 96,81 % / 0,037 | 97,67 % / 0,015 | 98,27 % / 0,005 | 98,455 % / 0,004 |
| 1 | 68,22 / 1,259 | 81,54 / 1,253 | 84,45 / 1,285 | 84,50 / 1,081 | 86,39 / 1,029 | 88,99 / 1,060 | 89,28 / 1,028 |
| 2 | 62,49 / 0,925 | 73,92 / 1,215 | 74,67 / 1,168 | 75,74 / 1,259 | 76,02 / 1,281 | 75,81 / 1,301 | 75,90 / 1,312 |
| 3 | 57,04 / 1,715 | 85,66 / 1,163 | 89,34 / 1,119 | 90,13 / 1,360 | 89,50 / 1,369 | 90,22 / 1,395 | 90,44 / 1,406 |
| 4 | 58,11 / 1,385 | 76,74 / 1,130 | 78,47 / 1,032 | 92,38 / 0,978 | 89,75 / 1,130 | 93,93 / 1,234 | 94,52 / 1,265 |

- **Signos de w_ih fijos desde 2^22 en las 5** (la 3 cambia en 2^21, la 4 en 2^22). Confirma, ya con script, lo que vio Fable.
- **Pero los signos no deciden el mecanismo**: la semilla 4 acaba con el mismo patrón que la 0 (`++--`) y nunca tiene espejo (residuo ≥ 0,98 en todos los puntos).
- **El residuo espejo sí separa pronto**: la 0 baja sin parar (0,545 → 0,155 en 2^21 → 0,004); las otras cuatro se quedan por encima de 0,92 desde 2^20 hasta el final. A 2^21 secuencias (el 0,2 % del entrenamiento) la 0 ya es distinta.
- El acierto, en cambio, no ordena bien pronto: la 4 está por debajo de la 1 y la 3 en 2^22 y acaba por encima.
- Lectura honesta: con 5 semillas es una pista. Si se mantiene con 20-100, el residuo espejo temprano sería un predictor del mecanismo final (lo que un revisor pediría, punto 4).

## Piloto de 20 semillas: criterios fijados antes de ver los datos (2026-11-19)

Pregunta de Javier: ¿aparece el mecanismo de la semilla 0 al entrenar desde cero? Semillas 5-24 entrenadas aquí con la receta de AlgZoo (`scripts/train_seed_h4.py --train-only`, 1 hilo, 2 procesos a la vez). Las 0-4 de ARC van aparte: la 0 es la mejor de 5 (elegida), no una semilla al azar.

Escrito y commiteado mientras entrenaban, sin haber mirado ningún resultado de las semillas 5-24:

- **Mecanismo presente** si y solo si residuo espejo final < 0,1 **y** ≥ 80 % de los fallos en casi empate (hueco relativo < 0,05). Referencia: semilla 0, 0,004 y 96 %; semillas 1-4, 1,03-1,41 y 13-47 %.
- **Predictor temprano**: residuo espejo con 2²¹ secuencias (0,2 % del entrenamiento) < 0,5. Semilla 0: 0,155; semillas 1-4: ≥ 1,13 en ese punto.
- Tasa = semillas con mecanismo / 20, con intervalo exacto al 95 % (Clopper-Pearson). Con 20 semillas el intervalo es ancho: 0 de 20 da 0-17 %; 1 de 20, 0,1-25 %.
- Se informan también los valores continuos (residuo, % en casi empate, acierto exacto) y las semillas que pasan solo una de las dos pruebas, para no esconder casos frontera.
- Script: `scripts/pilot_seeds_h4.py`.

## Piloto de 20 semillas: resultado (2026-11-19/20)

`PYTHONPATH=src python scripts/pilot_seeds_h4.py`. 20 de 20 entrenadas sin errores, 307-323 s cada una. Revisado por un agente Fable (código correcto, criterios aplicados sin cambios; sus correcciones a la interpretación están incorporadas abajo y en el script).

**Lo registrado de antemano:**

- **Mecanismo en 2 de 20 semillas (7 y 9): 10 %, intervalo exacto al 95 % 1-32 %.** Ninguna semilla pasa solo una de las dos pruebas: o las dos (residuo 0,005 y 0,014; 97 % de fallos en casi empate) o ninguna (residuo ≥ 1,03; ≤ 55 %).
- **El predictor temprano falla: 0 de 2.** Residuo con 2²¹: 0,640 (semilla 7) y 1,341 (semilla 9), los dos por encima de 0,5. Lo que falló es el momento, no la dirección (post hoc): la 7 baja 0,88 → 0,64 → 0,296 (2²²) → 0,005; la 9 sube primero (1,15 → 1,34) y baja después, 0,405 en 2²⁴ y 0,014 al final. Probar otro punto exigiría semillas nuevas: elegirlo ahora con estos datos sería ajustar el umbral al resultado.

**Post hoc (mirado después de ver la tabla; no registrado):**

- **Las semillas 7 y 9 son funcionalmente el mecanismo de la semilla 0.** Coinciden con la semilla 0 de ARC en el 99,71 % y el 99,63 % de las entradas; donde la de ARC falla, dan la misma respuesta errónea el 87,9 % y el 85,3 %. Referencia: la semilla 0 reentrenada aquí (misma inicialización) da 99,70 % y 85,5 %. Es decir, están tan cerca de la de ARC como su propia reentrenada. El diseño de 2 números coincide con ellas +0,71 puntos por encima de su acierto (ARC 0: +0,67), lo que significa que acierta parte de sus fallos, no todos (Fable: reproduce un 49-53 % de los fallos de las tres). [muestreado, 1 M entradas]
- **Acierto y mecanismo van juntos en estos datos:** las 3 redes con mecanismo (ARC 0, 7, 9) dan 98,46-98,54 %; la mejor sin él, 96,45 % (semilla 11). Con 3 frente a 22 es una observación, no un resultado.
- **El entrenamiento repite soluciones.** Huella = error exacto por orden verdadero (6 números; no cambia al reordenar o reescalar neuronas), comparada también con la red de signo cambiado (la red con w_ih negado calcula f(−x), igual de válida e igual de probable). Grupos con huella a menos de 0,2 puntos, confirmados por coincidencia directa de respuestas (la mejor de x y −x):
  - 13, 14, 16, 22: 99,87-99,90 %. Fallan un orden entero (x1 < x3 < x2, o su espejo x2 < x3 < x1: 16,67 % de las entradas).
  - 7, 9 y ARC 0: 99,71-99,82 %.
  - 5 y 6: 99,75 %. 12 y 18: 99,35 % (una es el espejo de la otra). 10 y ARC 1: 99,33 % (ídem).
  - Las otras 12 de 25 no tienen pareja (la huella más cercana, a 0,53-11,5 puntos). Con 25 redes no se puede decir que haya un conjunto pequeño de soluciones; sí que algunas se repiten casi exactas.
- **Las semillas 5, 6 y 20 (52-57 %) son las peores** (el azar es 33 %): responden x1 el 70-79 % de las veces; el 37-47 % de sus fallos son estado muerto. En la 5 y la 6 los cuatro pesos de entrada son positivos, así que con las tres entradas negativas el estado se apaga entero; pero eso solo explica el 12,5 % de las entradas.

**Qué dirá un revisor:** n = 20 no distingue «raro» (1 %) de «uno de cada tres»; los umbrales 0,1 / 80 % / 0,05 se calibraron con las 5 semillas conocidas; el entrenamiento local (CPU) no es bit a bit el de ARC (GPU), aunque la semilla 0 reentrenada reproduce el mecanismo; residuo espejo + dónde falla no prueban por sí solos la estructura de comparadores (lo hace la coincidencia funcional con la semilla 0).

**Respuesta a Javier:** el mecanismo aparece, pero no siempre: en 2 de 20 entrenamientos desde cero (10 %, 1-32 %), y cuando aparece es el mismo, casi respuesta por respuesta. El resto cae en otras soluciones, algunas repetidas. Siguiente paso natural: más semillas para estrechar la tasa (100 semillas ≈ 9 h de Mac, o en otra máquina) y, para el predictor temprano, fijar un punto nuevo (por ejemplo 2²² o 2²⁴) y probarlo en semillas que aún no hayamos mirado.

## 100 semillas: criterios fijados antes de entrenar (2026-11-20)

Semillas 25-124, que nadie ha mirado. Se entrenan por tandas con `scripts/train_batch_h4.py` (la misma receta del piloto: AlgZoo, 1 hilo; para por número de semillas o por horas, sigue donde lo dejó, vigila el calor). Las tandas, pausas y cortes no cambian nada de lo de abajo.

Escrito y subido a GitHub antes de entrenar la primera:

- **Mecanismo**: el mismo criterio del piloto, sin tocar (residuo espejo final < 0,1 y ≥ 80 % de fallos en casi empate). Tasa = k / 100 con intervalo exacto al 95 %; se da también la suma con el piloto (k + 2) / 120.
- **Aviso temprano**, calibrado con las 25 redes ya conocidas (`scripts/calibrate_early_h4.py`) y **probado solo en las 100 nuevas**:
  - principal: residuo espejo con 2²⁴ secuencias (1,6 % del entrenamiento) < 0,7. En las 25 conocidas: con mecanismo ≤ 0,405; sin él ≥ 0,978;
  - secundario: con 2²² (0,4 %) < 0,9. En las conocidas: ≤ 0,822 frente a ≥ 1,003 (margen estrecho).
  - Funciona si detecta todas las semillas con mecanismo y da como mucho 5 falsas alarmas.
- **Soluciones repetidas**: dos redes son la misma solución si su huella (error exacto por orden, también contra la red de signo cambiado) difiere menos de 0,2 puntos **y** dan la misma respuesta en ≥ 99 % de las entradas (la mejor de x y −x). Se informa del número de soluciones distintas y del tamaño de los grupos mayores.
- Comando: `PYTHONPATH=src python scripts/pilot_seeds_h4.py --seeds $(seq 25 124) --early 24:0.7 22:0.9`. Cualquier análisis que no esté aquí se marcará como post hoc.

## Inicio en espejo, experimento emparejado: criterios fijados antes de entrenar (2026-11-24)

Pista de un brainstorm (Fable, `scripts/explore_mirror_init/`, entrenamiento corto, post hoc): empezar con la red exactamente en espejo subía el mecanismo de 2/15 a 10/15. Esto la pone a prueba con la receta completa.

- **Diseño**: semillas 25-48. Cada una se entrena dos veces con la receta completa de AlgZoo: inicio normal y el mismo inicio hecho espejo exacto (`models.mirror_init_`: conserva la primera mitad, no saca números aleatorios, así que los datos son idénticos en las dos ramas). Las 24 normales cuentan también para las 100 semillas (mismo archivo, misma receta). Lanzador: `scripts/train_batch_h4.py --first 25 --last 48 --arms normal mirror`.
- **Mecanismo** (en las dos ramas): criterio del piloto (residuo espejo < 0,1 y ≥ 80 % de fallos en casi empate) **y** misma respuesta que la semilla 0 de ARC en ≥ 99 % de las entradas (la mejor de x y −x). En la rama espejo el residuo pequeño viene en parte de fábrica; la coincidencia es la que decide.
- **La idea sigue solo si**: rama espejo ≥ 10/24 **y** McNemar exacto unilateral sobre las parejas p < 0,05. Si no, se abandona y lo de semillas queda en post. Se informa además de las redes espejo que pasan el criterio del piloto sin llegar al 99 % (¿otra solución simétrica?) y de las que se salen de la simetría.
- Análisis: `PYTHONPATH=src python scripts/mirror_init_h4.py`. Lo que no esté aquí se marcará post hoc.
