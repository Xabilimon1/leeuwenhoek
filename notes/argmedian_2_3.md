# argmedian h=2 L=3 (12 parámetros)

Modelo: `zoo_argmedian(2, 3)`, semilla 0 (mejor de 5). Acierto **exacto** desde los pesos: **41,017126 %** (`scripts/exact_accuracy_table.py`; muestreo 10^7: 41,005 ± 0,016 %). Estado: **primera mirada descriptiva, sin explicación.**

## Comportamiento (primera mirada)
- Mapa: `figures/sphere_argmedian_2_3.png` (`scripts/sphere_map.py --hidden 2`).
- Error por orden real (mapa): casi perfecto cuando la mediana es x₁ (x₃<x₁<x₂: 1,4 % de error; x₂<x₁<x₃: 0,4 %); falla casi siempre cuando la mediana es x₃ (100 %) o es x₂ con x₁ máximo (98,7 %); x₁<x₂<x₃: 53 %.
- Casi todas las respuestas erróneas son «x₁».

## Mecanismo (observaciones sin explicar)
- Pesos (`scripts/relu_patterns.py --hidden 2`): w_ih = (2,95, 3,15), los dos positivos (no hay mitades espejo como en h=4). W_hh[0,0] = −76,5: la neurona 1 se apaga a sí misma con mucha fuerza.
- Estado final muerto (patrón `00` en el paso 3) en el **60,0 %** de las entradas; ahí todos los logits son 0 y argmax responde x₁. [Sin comprobar: que esto explique por sí solo la mayor parte del 41 %.]
- 13 caminos de activación (exacto); los 6 primeros cubren el 90,6 % (`scripts/paths.py --hidden 2`).

## Lo que queda sin explicar
- Todo el mecanismo. Hipótesis de partida: la red es «casi siempre x₁» y solo detecta algún caso de x₂.
