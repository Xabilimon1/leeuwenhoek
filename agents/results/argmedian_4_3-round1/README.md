# Estudio de agentes, ronda 1: argmedian h=4 L=3

2026-11-08. Tarea v2 (`agents/TASK.md`): explicar la red, gemelo sin copiar la red y estimación del acierto desde la explicación. 30 min por ejecución, acceso al modelo hasta 10⁶ entradas, sin internet, aislados con `agents/launch.sh`. Acierto exacto de la red: 98,455038 %.

Cada carpeta guarda lo que entregó el agente (`twin.py`, `estimate.txt`, `EXPLANATION.md`), su puntuación (`score.json`, de `agents/score.py`) y sus tiempos. Las transcripciones no están aquí (contienen rutas y listados del disco del Mac donde se ejecutaron); se conservan en local.

| Agente | Gemelo (rep. 1/2/3) | Media | Error estimación, puntos (1/2/3) | Media | Tiempo usado |
|---|---|---|---|---|---|
| Claude Code (Opus 5.5) | 99,24 / 99,25 / 99,61 | 99,37 % | 0,025 / 0,145 / 0,055 | 0,075 | 3-16 min |
| Antigravity CLI | 99,23 / 99,11 / 99,05 | 99,13 % | 0,125 / 0,115 / 0,057 | 0,099 | 12-13 min |
| Codex CLI | 98,45 / 98,45 / 98,45 | 98,45 % | 0,245 / 0,145 / 0,155 | 0,182 | 3-5 min |
| Grok Build (rep. 3/4/5) | 99,54 / — / — | — | — / — / — | — | 30 min, cortado |
| Referencia (nuestra) | 99,83 (22 zonas) | | 0,149 | | días |

- **Ninguna estimación supera al muestreo con el mismo cómputo** (las entradas que gastó cada agente habrían dado 0,011-0,026 puntos de error). Es la métrica de ARC.
- Codex entrega siempre la regla de la mediana, sin modelar errores. Grok entrega 1 gemelo de 3 y ninguna estimación: profundiza y no cierra. Claude, Codex y Antigravity paran mucho antes del límite.
- `grok-argmedian_4_3-1` es el piloto con la tarea v1: gemelo 100 % copiando la red (16 de 32 pesos literales) y estimación integrando su función. Motivó la tarea v2. `grok-argmedian_4_3-2` no cuenta (siguió 40 min por un fallo del corte de tiempo, ya arreglado).

**Avisos para leer los resultados**

- **scipy faltaba** en el Python de los agentes aunque la tarea lo prometía (9 transcripciones dan «No module named 'scipy'»). Antigravity-3 lo instaló con pip a mitad de la ronda B; las ejecuciones posteriores o simultáneas pudieron tenerlo. Condición desigual.
- **Reglas rotas:** tope de entradas superado por claude-1 (1,30 M), codex-3 (1,20 M) y grok-3 (1,11 M). Antigravity-3 instaló scipy desde internet y recorrió el disco con `find /` (el sandbox le bloqueó las notas y los repos).
- **Constantes medidas, no deducidas:** algunos gemelos ajustan constantes con muestras del modelo (p. ej. claude-3, «measured knots»), algo que la tarea pedía derivar de los pesos.
- Claude y Antigravity vieron el acierto muestreado antes de estimar (permitido por el acceso al modelo): sus estimaciones no son del todo independientes del muestreo.
