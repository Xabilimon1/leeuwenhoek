# Estudio de agentes: resumen de las tres redes

Cuatro agentes (Claude Code con Opus 5.5, Codex CLI, Antigravity CLI y Grok Build), tres repeticiones cada uno, en argmedian L=3 con h=2, 4 y 8. Misma tarea v2 en todas: explicar la red, un gemelo escrito a mano sin copiar la red y una estimación del acierto que salga de la explicación (30 min, hasta 10⁶ entradas del modelo, sin internet). Detalle, avisos y trampas en [`argmedian_4_3-round1/`](argmedian_4_3-round1/README.md) (h=4, 2026-11-08) y [`argmedian-round2/`](argmedian-round2/README.md) (h=2 y h=8, 2026-11-10/16/24). Cifras: medias de los `score.json` de cada carpeta (`agents/score.py`, 10⁶ entradas, semilla 20261108).

| Agente | h=2: gemelo / error est. | h=4: gemelo / error est. | h=8: gemelo / error est. |
|---|---|---|---|
| Claude Code | 99,994 % / 0,006 | 99,37 % / 0,075 | 99,972 % / 0,0045 ¹ |
| Antigravity CLI | 99,985 % / 0,006 | 99,13 % / 0,099 | 99,963 % / 0,0080 |
| Grok Build | 99,995 % / 0,006 | 1 gemelo de 3 (99,54 %) / ninguna | 99,942 % / 0,0054 ¹ |
| Codex CLI | 98,15 % / 1,29 | 98,45 % / 0,182 | 99,942 % / 0,0053 ² |
| Acierto exacto de la red | 41,017 % | 98,455 % | 99,945 % |

Error de la estimación en puntos porcentuales. ¹ Incluye estimaciones que ejecutan la red (bisección) o la recorren trozo a trozo, prohibido por la tarea. ² Anchos de franja escritos sin deducción. Ver los avisos de cada ronda.

**¿Gana alguna estimación al muestreo con el mismo cómputo?** (la métrica de ARC)

- h=2: sí, casi todas (Claude, Grok y Antigravity, 3 de 3 cada uno). Pero con 12 parámetros la explicación completa cabe en unas reglas: explicar y compilar casi coinciden.
- h=4: ninguna de las 9 estimaciones entregadas.
- h=8: ninguna limpia. Las que ganan (claude-8-1, claude-8-2, grok-8-1, grok-8-2) ejecutaron o recorrieron la red. Las cuatro deducidas de los pesos de principio a fin (antigravity-8-1/2/3 y grok-8-3) pierden.

**Lo que se aprende**

- Con más neuronas, el gemelo es fácil (la mediana perfecta ya coincide con h=8 en el 99,942 %) y lo difícil es estimar honestamente las franjas de error.
- Antigravity es el único con las tres ejecuciones de h=8 limpias, y sus gemelos (constantes deducidas de los pesos) están a la altura de los de Claude (constantes medidas sobre el modelo).
- Codex simplifica de más (regla de la mediana en h=4 y h=8); Grok profundiza y no siempre cierra a tiempo (h=4).
- Revisar las transcripciones cambia la lectura: la puntuación sola premia a quien ejecuta la red a escondidas.
