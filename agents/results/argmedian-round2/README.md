# Estudio de agentes, ronda 2: argmedian h=2 y h=8 (L=3)

2026-11-10. Mismas reglas que la ronda 1 (`agents/round2.sh`): tarea v2, 30 min por ejecución, acceso al modelo hasta 10⁶ entradas (registrado, no forzado), sin internet, numpy y scipy garantizados (`~/agent-runs/.venv`), aislados con `agents/launch.sh`. Aciertos exactos: **h=2 41,017126 %** (12 parámetros), **h=8 99,945490 %** (96 parámetros).

Cada carpeta guarda lo que entregó el agente (`twin.py`, `estimate.txt`, `EXPLANATION.md`), su puntuación (`score.json`, de `agents/score.py`, 10⁶ entradas, semilla 20261108) y sus tiempos. Las transcripciones se conservan en local.

**Antigravity no está**: se quedó sin cuota antes de empezar («Individual quota reached… Resets in 127h»). Sus 6 ejecuciones se repetirán cuando se renueve.

## h=2

| Agente | Gemelo (rep. 1/2/3) | Error estimación, puntos (1/2/3) | Entradas usadas | Tiempo |
|---|---|---|---|---|
| Claude Code (Opus 5.5) | 99,999 / 99,983 / 99,999 | 0,003 / 0,013 / 0,003 | 0,8-0,9 M | 3-16 min |
| Codex CLI | 96,51 / 99,68 / 98,25 | 2,92 / 0,31 / 0,65 | 125 / 0,2 M / 0,65 M | 4-7 min |
| Grok Build | 99,985 / 100,000 / 100,000 | 0,017 / 0,000 / 0,000 | 0,18 / 0,28 / **2,12 M** | 11-27 min |

- Claude y Grok describen **todas** las regiones de la red con 6-7 constantes deducidas de los pesos (fórmulas como −w_hh[1,1]) e integran la estimación en forma cerrada; Grok rep. 2 y 3 aciertan el valor exacto con 8-9 cifras.
- **Con 12 parámetros, explicar y compilar casi coinciden**: la red tiene tan pocas regiones que su descripción completa cabe en unas reglas legibles. h=2 no separa bien «entender» de «copiar»; los resultados dicen poco de la capacidad de explicar.
- Codex simplifica de más (rep. 3: «responde 1 si la secuencia sube», estimación 5/12) y es el único que no clava h=2.

## h=8

| Agente | Gemelo (rep. 1/2/3) | Error estimación, puntos (1/2/3) | Entradas usadas | Tiempo |
|---|---|---|---|---|
| Claude Code (Opus 5.5) | 99,980 / 99,962 / 99,973 | 0,0005 / 0,0015 / 0,0115 | 0,81 / 0,28 / 0,36 M | 2-3 min |
| Codex CLI | 99,942 ×3 (regla de la mediana) | 0,0055 / 0,0030 / 0,0075 | **1,65 M** / 0,98 / 1,00 M | 4-6 min |
| Grok Build | 99,942 ×3 (regla de la mediana) | 0,0000 / 0,0017 / 0,0145 | 0,55 / 0,38 / 0,63 M | 23-26 min |

- La mediana perfecta ya coincide con la red en el 99,942 %: la red solo falla en franjas finísimas junto a los empates. Gemelo fácil, estimación fácil de acercar.
- **Casi ninguna estimación de h=8 sale limpia de la explicación** (ver avisos). Las que ganan al muestreo con el mismo cómputo (claude-1, claude-2, grok-1, grok-2) son justo las que ejecutaron la red o la recorrieron trozo a trozo.

## Avisos para leer los resultados

- **Estimaciones que ejecutan la red** (prohibido por la tarea): claude-8-1, claude-8-2 y grok-8-2 miden el ancho de las franjas de error por bisección sobre el modelo (720 ángulos) y lo integran. Claude lo declara; grok-8-2 lo registra en `model_calls.jsonl` (181.440 entradas, «bisection of geodesic error-band width») pero en `estimate.txt` escribe «read off the weights».
- **Estimación que recorre la red trozo a trozo** (prohibido): grok-8-1 («every activation cone… inside each cone the logits are linear»).
- **Anchos sin deducción**: codex-8-1/2/3 muestrean el acierto y los huecos de los errores y luego escriben un «ancho efectivo» (0,00027; 3,4·10⁻⁴; 0,00075) que no sale de ningún cálculo de la transcripción, aunque lo llaman «weight-derived».
- **Constantes medidas, no deducidas**: los gemelos de claude-8-1/2/3 usan desplazamientos de frontera medidos sobre el modelo (claude-8-3 lo declara: «measured from the model's behaviour»).
- grok-8-3 saca la inclinación de los pesos, pero abandona la deducción por tiempo y redondea (0,9996) sabiendo por el muestreo que el valor real es «closer to 0.9994».
- **Tope de entradas superado**: grok-2-3 (2,12 M), codex-8-1 (1,65 M); codex-8-3 lo roza (1,00 M).
- Revisión de las transcripciones: ningún acceso fuera de la carpeta, ninguna búsqueda web (`web_search_requests: 0`), ninguna instalación de paquetes, ningún bloqueo del sandbox. Los 18 gemelos se leyeron antes de ejecutarlos: numpy puro, sin pesos pegados (≤ 1 coincidencia casual de 12) y funcionan sin `weights.npz`.
