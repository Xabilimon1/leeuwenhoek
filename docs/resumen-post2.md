# Leeuwenhoek, post 2: la ley del error (para entenderlo y contarlo)

Escrito el 2026-11-10 para que Xabier lo entienda y lo cuente con sus palabras. Continúa el post 1 («Taking apart a 32-number network»). Cada cifra sale de un script del repo (columna «script» de la segunda parte); los detalles completos están en `notes/argmedian_4_3.md`.

- **Primera parte: la historia.** Para leer de corrido.
- **Segunda parte: la referencia.** Vocabulario, cifras con su script, lo que falta y figuras posibles.

---

# Primera parte: la historia

## Dónde lo dejamos

En el post 1 desmontamos la red de argmedian de 32 parámetros (encuentra la posición de la mediana de tres números). Vimos que:

- sus 4 neuronas son dos parejas espejo: una se enciende cuando el número sube y la otra cuando baja;
- cada pareja lleva dos canales: **comparación** (¿subió o bajó?) y **memoria** (lo justo para reconstruir el número anterior);
- falla casi solo cuando dos números están casi empatados, y entonces tira hacia x₁.

Terminamos con una sorpresa: escribiendo solo lo que la explicación exige, fabricamos una red cuyos pesos coinciden con los entrenados. Y prometimos una **ley del error**. Este post es esa ley: de dónde sale, cuánto se acerca a la red real y qué nos dice.

## La red fabricada tiene solo 2 números libres

Al fabricar la red desde la explicación pusimos tres condiciones:

1. que la comparación del paso 2 sea exacta;
2. que la del paso 3 también, sin restos de x₁;
3. que la salida pueda aplicar la regla de la mediana.

Con eso, casi todos los pesos quedan fijados. Solo quedan libres **tres** números, y uno de ellos no importa:

- **a**: el tamaño del canal de comparación;
- **b**: el tamaño de la memoria (cuánto del número actual se guarda);
- **e**: cuánto olvida la memoria en cada paso.

Como la red no tiene sesgos, solo importa la proporción **b/a**. Así que los números libres de verdad son **2: b/a y e**. La escala de la salida tampoco cambia nada: al doblarla, 0 cambios en 10⁶ entradas.

Leyendo a, b y e de la red entrenada (a = 8,673, b = 0,743, e = 0,369), las fórmulas predicen el resto de sus pesos. Por ejemplo, un peso de salida deducido vale −1,346 y el entrenado −1,350. La red fabricada acierta el **98,516 %** (la entrenada, el 98,455 %).

## La ley

Para la red fabricada, el error tiene una fórmula exacta:

**error = (√3 / 2π) × (b / (e × a))²  ≈  0,2757 × (b / (e × a))²**

En palabras: el error crece con el **cuadrado** del tamaño relativo de la memoria. Si divides la memoria entre 2, el error se divide entre 4.

### De dónde sale (la intuición)

1. **Dónde están los errores.** Mi primera idea fue que estarían en los «polos», las direcciones donde los tres números se parecen. Falso: los errores forman con la diagonal un ángulo de unos 60°, igual que cualquier entrada al azar. Lo que se encoge al bajar la memoria es el **hueco entre los dos números empatados**.
2. **El mecanismo.** Cerca de un empate de x₂ con x₃, la comparación (que mide x₃ − x₂) se hace casi cero y se queda más pequeña que la memoria. Entonces la pareja de neuronas se rompe: queda encendida una neurona de cada pareja. En ese estado, la salida lee la memoria donde debería leer la comparación, y responde x₁.
3. **El ancho de la ventana.** Eso pasa dentro de una ventana de |x₃ − x₂| cuyo ancho es (b/(e×a))² × |x₁ − v|, donde v es el valor de la pareja empatada. Sale al cuadrado porque la memoria entra dos veces: una por lo que guarda y otra por cuánto pesa en la salida.
4. **La cuenta.** Error = probabilidad de caer en la ventana = (densidad de x₃ − x₂ en el cero) × (ancho medio de la ventana).
   - x₃ − x₂ es una gaussiana de varianza 2, con densidad 1/(2√π) en el cero.
   - x₁ − v es una gaussiana de varianza 3/2, con valor absoluto medio √(3/π).
   - Producto: √3/(2π) = 0,2757.

### Comprobada

- Con la enumeración exacta (el cálculo exacto del acierto por zonas del post 1), el cociente error / (b/(e×a))² da **0,2757 para todos los valores de e** probados, de 0,25 a 0,70.
- Todos los errores de la red fabricada son ese empate de x₂ con x₃, con x₁ en un extremo.
- Aguanta incluso con la memoria grande de la red real: con los a, b y e de la red real, la ley da 1,4814 % y el cálculo exacto del diseño 1,4838 %.

## La red entrenada frente a la ley: 0,064 puntos

Con los a, b y e de la red entrenada, la ley predice un error del **1,481 %**. La red tiene un **1,545 %**. Faltan **0,064 puntos**.

Lo esperable sería que fuese un error pequeño de la red. Pero no: **es la resta de efectos grandes de signo contrario**. La red entrenada no es el diseño ideal:

| Tipo de error | Diseño ideal | Red entrenada | Diferencia |
|---|---|---|---|
| Empate x₂ ≈ x₃, responde x₁ (banda + estado muerto) | 1,475 % | 1,100 % | **−0,376** |
| Giro: frontera x₁/x₃ desplazada | 0 | 0,199 % | **+0,199** |
| Empate x₁ ≈ x₂, responde x₂ | 0 | 0,133 % | **+0,133** |
| Empate x₁ ≈ x₂ y luego el paso 3 decide mal | 0,009 % | 0,112 % | **+0,103** |
| Empate x₂ ≈ x₃, otra respuesta | 0 | 0,002 % | +0,002 |
| La ley frente al diseño exacto (orden superior) | | | +0,002 |
| **Total** | **1,481 % (ley)** | **1,545 %** | **+0,064** |

- La red entrenada **estrecha la ventana del «x₁ por defecto»**: comete 0,376 puntos menos de ese error. A cambio, **comete errores nuevos** que el diseño no tiene (fronteras desplazadas en los giros y en el paso 2): suman 0,435 puntos.
- Tres de esas filas se predicen con una fórmula de primer orden (densidad en el empate × ancho de la ventana, anchos sacados de los pesos): giro 0,198 % (exacto 0,192), paso 2 0,118 % (exacto 0,133), banda + muerto 1,131 % (exacto 1,100).
- **La cuarta no la predecimos**: necesita dos casi empates a la vez (segundo orden). Solo la medimos. Por eso la predicción de primer orden, sumada, da 1,447 % y **falla por 0,090 puntos**, más que la propia ley (0,064): el desglose explica de dónde sale cada error, pero no predice mejor el total.
- **Curioso**: si trasplantamos al diseño cada diferencia de la red por separado, el error se dispara (hasta +2,4 puntos). Todas juntas, solo +0,053. **Los pesos de la red se compensan entre sí.**

## Podría ser casi perfecta

La ley dice que, con menos memoria, la red fallaría menos:

- con b 2,5 veces más pequeña (b = 0,3), el diseño acierta el **99,76 %**;
- con b 37 veces más pequeña (b = 0,02), el **99,999 %**.

O sea, **una red de 32 parámetros puede hacer esta tarea casi perfecta, y el entrenamiento no lo encontró**: se quedó con una memoria mucho mayor de la necesaria. **No sabemos por qué.** Es la pregunta abierta del post. Algunas ideas, sin comprobar:

- el entrenamiento optimiza una pérdida suave (entropía cruzada), no el acierto, y puede que le compense otra cosa;
- con una memoria diminuta, la red sería más frágil ante el ruido.

## Las comprobaciones que sugirió ARC

Le mandamos el post 1 a Jacob Hilton (ARC) y propuso dos pruebas de solidez.

### 1. ¿Cuánto cuesta la estimación? (FLOPs)

Un FLOP es una operación aritmética básica: una suma, una multiplicación, una comparación. La idea de ARC: una explicación vale si permite calcular el acierto **más barato** que muestrear (probar la red con entradas al azar).

- Nuestra estimación cuesta **570.000 FLOPs** y se equivoca en 0,149 puntos.
- Muestrear hasta ese error cuesta **1,0 millones** (6.800 entradas a 147 FLOPs cada una). **Solo somos 1,8 veces más baratos.**
- El cálculo exacto por fuerza bruta cuesta **7,3 millones** (12,8 veces más) y da error 0.
- Lectura honesta: en una red tan pequeña, la explicación apenas ahorra cómputo. Su valor está en entender la red, no en el ahorro. Además, nuestro error es un sesgo fijo: por encima de ~10⁶ FLOPs, muestrear gana.

### 2. ¿Aguanta si toco los pesos? (ruido)

Multiplicamos cada peso por (1 + ε × ruido), recalculamos la estimación desde la explicación con los pesos nuevos y la comparamos con el acierto exacto de la red tocada.

| Ruido ε | Cuánto cae el acierto | Qué parte de esa caída sigue la estimación |
|---|---|---|
| 0,1 % | 0,31 puntos | 85 % |
| 0,3 % | 1,0 puntos | 77 % |
| 1 % | 5,7 puntos | 65 % |

- **Con poco ruido, la explicación generaliza**: sigue la mayor parte de lo que le pasa a la red.
- **Con más ruido aparecen fallos nuevos**: la banda del empate sigue en su sitio, pero la red ya no responde x₁ ahí, sino x₃. La estructura aguanta; el detalle no.
- **La fragilidad está en W_hh**, la matriz que pasa la memoria de un paso al siguiente: con ruido solo en ella, el acierto cae 1,09 puntos; solo en la salida, 0,23; solo en la entrada, 0,17.
- La red **no está exactamente en un óptimo**: con un ruido diminuto (0,01 %), el acierto sube en 13 de 20 redes. Desde el 0,3 %, baja siempre.

## Para cerrar

Ideas que podrías destacar (son tuyas para elegir):

- Una red de 32 números tiene una ley exacta para su error, y cabe en una línea.
- La red real no es el diseño ideal: cambia un error grande por varios pequeños que casi se compensan.
- Podría ser casi perfecta, y el entrenamiento no lo encontró.
- La explicación aguanta el ruido pequeño, pero ahorra poco cómputo: entender no es lo mismo que calcular barato.

---

# Segunda parte: la referencia

## 1. Vocabulario (lo nuevo respecto al post 1)

- **a, b, e**: los tres números libres del diseño. a = tamaño del canal de comparación; b = tamaño de la memoria; e = cuánto olvida la memoria en cada paso.
- **b/a**: tamaño relativo de la memoria. Es lo que importa, porque la red no tiene sesgos.
- **Ley del error**: fórmula que da el error del diseño a partir de b/a y e, sin probar la red.
- **Ventana**: rango de |x₃ − x₂| en el que la red responde x₁ por error.
- **Densidad en el cero**: probabilidad por unidad de ancho de que x₃ − x₂ valga casi 0. Multiplicada por el ancho de la ventana, da la probabilidad de caer dentro.
- **Primer / segundo orden**: primer orden = errores que necesitan un casi empate; segundo orden = dos a la vez (mucho más raros y más difíciles de predecir).
- **Trasplante**: poner en el diseño un peso de la red entrenada para ver cuánto error añade.
- **FLOP**: una operación aritmética básica.
- **Muestrear**: estimar el acierto probando la red con entradas al azar.
- **Sesgo fijo**: error que no baja por mucho que calcules (al contrario que el del muestreo, que baja con más muestras).
- **W_hh**: la matriz recurrente; pasa el estado (la memoria) de un paso al siguiente.
- **Caída capturada**: de lo que baja el acierto con ruido, qué parte predice la estimación recalculada.

## 2. Cifras y de dónde salen

| Afirmación | Cifra | Script |
|---|---|---|
| Números libres del diseño | 2 (b/a y e) | `scripts/design_twin_h4.py` |
| Peso deducido frente a entrenado | −1,346 frente a −1,350 | `scripts/design_twin_h4.py` |
| Acierto del diseño | 98,516 % (red 98,455 %) | `scripts/design_twin_h4.py` |
| Constante de la ley | 0,2757 para e de 0,25 a 0,70 | `scripts/law_geometry.py` |
| Los errores no están en los polos | ángulo mediano ≈ 60° | `scripts/law_geometry.py` |
| Ley con la b real | 1,4814 % frente a 1,4838 % exacto | `scripts/gap_h4.py` |
| Red entrenada frente a ley | 1,545 % frente a 1,481 %: 0,064 puntos | `scripts/gap_h4.py` |
| Desglose de los 0,064 | −0,376 / +0,199 / +0,133 / +0,103 | `scripts/gap_h4.py` |
| Predicción de primer orden | 0,198 / 0,118 / 1,131 % | `scripts/gap_formula.py` |
| Trasplantes | hasta +2,4 solos; +0,053 juntos | `scripts/gap_h4.py` |
| Memoria menor | b = 0,3 → 99,76 %; b = 0,02 → 99,999 % | `scripts/design_twin_h4.py` (b = 0,02: ley) |
| FLOPs | 570.244 frente a 1,0·10⁶; exacto 7,3·10⁶ | `scripts/jacob_flops.py` |
| Ruido | 85 / 77 / 65 % capturado | `scripts/jacob_noise.py` |
| Fragilidad por matriz | W_hh 1,09; salida 0,23; entrada 0,17 | `scripts/jacob_noise.py` |

## 3. Lo que NO tenemos (el post debe decirlo)

- Por qué el entrenamiento deja b/a = 0,086 si una memoria menor daría casi el 100 %.
- La fila «paso 2 y luego paso 3 mal» (0,112 %) no se predice; solo se mide.
- Solo un tamaño (h=4) y una semilla de entrenamiento.
- La ley es para el diseño ideal. Para la red entrenada hay que sumarle el desglose.
- *Surprise accounting* de ARC: sin hacer.

## 4. Figuras posibles (por hacer, decidir cuáles)

1. **La ley**: error frente a b/(e×a) en escala logarítmica, puntos exactos y la recta de la ley, con la red entrenada marcada.
2. **Los 0,064 puntos**: gráfico de cascada, del 1,481 % de la ley al 1,545 % de la red, fila a fila.
3. **La ventana**: zoom de la esfera en un empate, con la ventana del diseño y la de la red.
4. **Ruido**: caída del acierto y parte capturada, frente a ε.
5. **Memoria frente a acierto**: acierto del diseño según b, con la b entrenada marcada («podría estar aquí»).

## 5. Antes de publicar

- Revisar que las cifras del post coinciden con los scripts (volver a ejecutarlos).
- Para LessWrong: el inglés lo escribe Xabier desde el principio, sin traductor ni retoques de Claude.
