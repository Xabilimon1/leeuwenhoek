# argmedian h=8 L=3 (96 parámetros)

Modelo: `zoo_argmedian(8, 3)`, semilla 1 (mejor de 5). Acierto **exacto** desde los pesos: **99,945490 %** (`scripts/exact_accuracy_table.py`; muestreo 10^7: 99,9455 ± 0,0007 %). Estado: **primera mirada descriptiva, sin explicación.**

## Comportamiento (primera mirada)
- Mapa: `figures/sphere_argmedian_8_3.png`. Ojo: el mapa no sirve para medir aquí. El 91 % de los errores está a < 0,125° de una frontera entre órdenes, más fino que un píxel; el mapa da 0,007 % de error frente a 0,0545 % exacto.
- Errores repartidos entre los seis órdenes (0,004-0,011 % cada uno en el mapa), sin sesgo claro hacia x₁ como en h=4.

## Mecanismo (observaciones sin explicar)
- 232 caminos de activación (exacto; el muestreo con 3·10^5 entradas ve 177). Los 8 primeros cubren el 60 % (`scripts/paths.py --hidden 8`).
- En el paso 1 solo aparecen dos patrones complementarios, `10110101` y `01001010`: parece haber otra vez dos mitades (positiva y negativa, 4 + 4 neuronas). [Sin comprobar la simetría espejo.]

## Lo que queda sin explicar
- Todo el mecanismo. Pregunta de partida: ¿es el algoritmo de h=4 (comparar con el anterior; en los giros, x₁ contra x₃) con más precisión en los empates?
