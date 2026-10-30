# Brief del post: argmedian h=4 (paso 1 del flujo del blog)

Escrito el 2026-10-30. Paso 2 (tuyo): boceto a mano con tus ideas y frases, sin pulir. Después lo montamos en localhost.

## Antes: revisión del hito

**Qué afirmamos** (cada cifra en `notes/argmedian_4_3.md`, con su script):

| Afirmación | Prueba | Solidez |
|---|---|---|
| La red guarda si cada número sube o baja respecto al anterior | la mitad encendida coincide con la comparación real en 100 / 99,92 / 99,77 % de las entradas | alta |
| Cada mitad lleva dos canales: comparación (s) y memoria (d) | base suma/diferencia; los pesos de salida leen d | alta |
| La vuelta está cerrada | gemelo de 150 conos = red en float64 en 4·10⁶ de 4·10⁶ entradas | alta, pero los últimos 128 conos son enumeración, no explicación |
| La explicación comprimida cubre casi todo | 22 conos: 99,83 % de coincidencia, 89,8 % de la masa de error | alta |
| Un diseño de 2 números libres reproduce la red | predice los demás pesos dentro de su margen; 98,52 % de acierto (red 98,455 %); coincide en 98,76 % | media: falla donde la red, no en los mismos puntos |
| Ley del error deducida: (√3/2π)·(b/(e·a))² | enumeración exacta: 0,2757 para todo e; red entrenada 1,481 % frente a 1,545 % | alta para el diseño; para la red entrenada faltan 0,064 puntos |

**Lo que NO tenemos** (y el post debe decirlo):
- *Surprise accounting* de ARC: sin hacer.
- Los 0,064 puntos entre la ley y la red entrenada: **explicados** como resta de efectos grandes (banda más estrecha −0,376; giros +0,199; paso 2 +0,133; paso 2 y luego paso 3 +0,103). Tres predichos por fórmula; el último (segundo orden, dos casi empates) solo medido. Decidido 2026-10-30: se da por explicado y se escribe.
- Por qué el entrenamiento deja b/a = 0,086 si un b menor daría casi el 100 %.
- Solo un tamaño (h=4) y una semilla.
- **Antes de publicar**: repetir la búsqueda de que nadie ha explicado argmedian (3 búsquedas o más, regla del proyecto). Lo verificado el 2026-10-30 no vale para el día de publicar.

**Decisión tuya**: ¿damos h=4 por cerrado para el post, o primero cerramos los 0,064 puntos?

## El brief

**Audiencia**
- Equipos de interpretabilidad (ARC, Anthropic) y gente de LessWrong / Alignment Forum: tienen que poder comprobarlo todo.
- Lectores técnicos de xabier.me que no conocen AlgZoo: tienen que entenderlo sin saber qué es una RNN por dentro.
- Tú dentro de un año.

**Tesis en una frase**
Una red de 32 parámetros que busca la mediana de tres números se puede explicar entera: guarda si cada número sube o baja, y su 1,5 % de errores sigue una ley que se deduce con papel y lápiz.

**Estructura sugerida**
1. **El bicho.** AlgZoo, el reto de ARC, la red de 32 números y su 98,46 %. Por qué una red tan pequeña importa (ARC: entender del todo modelos de cientos de parámetros como requisito).
2. **Qué hace.** Sube / baja: una mitad de neuronas por cada sentido. Racha → la del medio es x₂; giro → compara x₁ con x₃. Figura: las lámparas (`plots.plot_steps`).
3. **Dónde falla.** Casi empates; la pareja de neuronas se rompe y gana x₁ por defecto. Figura: el plano y la lupa (`plots.overview`, caso «banda»).
4. **La vuelta.** Volver a fabricarla: el gemelo de 2 números libres predice los pesos entrenados. Qué significa «explicar» aquí.
5. **La ley.** La deducción en cuatro líneas, la hipótesis que falló (los polos) y la comprobación exacta. La red real a 0,064 puntos de la fórmula, y por qué: sus imperfecciones se compensan entre sí (cada una sola sube el error hasta +2,4 puntos; juntas, +0,053).
6. **Lo que falta.** Lo de la lista de arriba, sin suavizar. Siguiente: h=8 y el estudio de agentes.

**Tono**
Técnico y honesto; frases cortas; contar lo que se rompió (la hipótesis de los polos, el mapa que mentía en h=8). Mentor sin paternalismo: cada término se explica una vez.

**Evitar**
- «Primera explicación» o «nadie lo había hecho» sin la búsqueda repetida.
- Vender la vuelta de 150 conos como explicación: la explicación son los 22 conos y la ley.
- Hype sobre «entender redes neuronales». Es una red de 32 números.
- El punto medio «·» (regla del blog).

**Longitud**: 1.800-2.500 palabras, unos 10-12 minutos. Versión en inglés y en castellano, como los demás posts.

**Frontmatter propuesto**
```yaml
title: "Taking apart a 32-number network"   # elegido por Xabier 2026-10-30
excerpt: "..."                              # cuando haya boceto
publishedAt: "YYYY-MM-DD"
category: "research"
readingMinutes: 11
icon: "ph:microscope-fill"                  # [sin verificar que exista en Phosphor]
color: "mint"
lang: "en"
translationOf: "<slug en castellano>"
draft: true
keywords: ["mechanistic interpretability", "AlgZoo", "ARC", "RNN", "argmedian"]
```

**Figuras** (todas salen de `scripts/render_plots.py`; habrá que exportarlas al tema del blog): lámparas de un giro; plano + lupa de la banda; gráfica de la ley (error frente a b/(e·a), puntos exactos y curva).

**Tus decisiones pendientes**
1. ~~¿h=4 cerrado o primero los 0,064 puntos?~~ Decidido 2026-10-30: primero los 0,064 puntos.
2. ~~Título.~~ «Taking apart a 32-number network».
3. ¿Post solo en xabier.me o también LessWrong desde el principio?
