# Plan de Leeuwenhoek

Escrito el 2026-10-30 a partir de la investigación de ese día (x-brain `Projects/ResearchPath/Brainstorm/2026-10-30-busqueda-proyecto-labs.md`). Es un plan de partida: las decisiones de investigación las toma Xabier.

## 1. La pregunta

¿Hasta qué tamaño se puede **explicar del todo** cada familia de AlgZoo, y dónde se rompe? Y, en paralelo: ¿pueden los agentes de IA (Claude Code, Codex, Gemini CLI y quizá Grok) hacer ese mismo trabajo, y dónde fallan o hacen trampas?

Titular si sale bien: «primeras explicaciones completas de argmedian / median / longest cycle». Titular si sale mal: «aquí es donde deja de poder explicarse, y por qué». Los dos se pueden publicar.

## 2. Lo que ya sabemos (verificado el 2026-10-30)

**AlgZoo** ([repo](https://github.com/alignment-research-center/alg-zoo), commit `4433f50`, licencia MIT-0; [post de ARC](https://www.alignment.org/blog/algzoo-uninterpreted-models-with-fewer-than-1-500-parameters/), Jacob Hilton, 26-01-2026):
- 4 familias: `2nd_argmax`, `argmedian`, `median` (RNN) y `longest_cycle` (transformer). Modelos de 8 a 1408 parámetros.
- Cada modelo: mejor de 5 semillas (`seed=0..4` para las demás) y checkpoints de entrenamiento (`n_seqs` = 0, 2^20 … 2^29; el final es 2^30).
- Entradas: secuencias de números con distribución normal estándar (RNN). El post de ARC pide entender por qué el modelo acierta **con entradas gaussianas**.
- `argmedian`: ocultas {2, 4, 8, 16, 32} × longitud {3, 5, 7, 11}. Arquitectura `DistRNN`: `rnn.weight_ih_l0` (h×1), `rnn.weight_hh_l0` (h×h), `linear.weight` (L×h), sin sesgos, ReLU.
- Medido en el Mac (venv temporal, 2 hilos), argmedian longitud 3:

  | Modelo | Parámetros | Acierto (10^6 ejemplos) |
  |---|---|---|
  | h=2 | 12 | 41,0 % |
  | h=4 | 32 | 98,46 % |
  | h=8 | 96 | 99,95 % |

  10^6 ejemplos tardan < 0,1 s. El cómputo no es un problema.
- `median` (4,3) = `ScalarRNN` de 24 parámetros (salida escalar, se mide con error cuadrático). `longest_cycle` (3,3) = `DistTransformer` de 99 parámetros.
- Tablas completas de acierto por tamaño: imágenes `images/zoo_*.svg` del repo de ARC.

**Qué cuenta como explicación para ARC** (del post):
- Una **estimación mecanística** del acierto (o de la pérdida): deducir el acierto a partir de la estructura del modelo, sin muestrear.
- Métrica 1: **error cuadrático frente a cómputo**, comparado con muestrear al azar con el mismo cómputo. Igualar al muestreo en el modelo de 432 parámetros sería, según ARC, «un gran avance».
- Métrica 2: **surprise accounting**: entendimiento completo = una explicación con tantos bits de sorpresa como bits de optimización usados para elegir el modelo.
- ARC «más o menos» entiende el de 32 parámetros de 2nd argmax; el de 10 parámetros lo explica con unos 40 bits de sorpresa.

**El hueco** (verificado con búsquedas web, revisión de los 8 repos de GitHub sobre AlgZoo y búsqueda de código): nadie ha publicado explicaciones de argmedian, median ni longest cycle. Toda la comunidad trabaja en el modelo de 432 parámetros de 2nd argmax (6 repos en octubre de 2026 y un post de fuerza bruta del 26-10-2026). Puede haber trabajo sin publicar: volver a mirar antes de publicar.

**Agentes de interpretabilidad** (lo más cercano):
- AgenticInterpBench (arXiv 2606.24026, EMNLP 2026 Findings): 84 circuitos semisintéticos, y al agente se le da el circuito ya localizado.
- «Pitfalls in Evaluating Interpretability Agents» (arXiv 2603.20101): los agentes pueden reproducir hallazgos publicados de memoria.
- Nadie ha medido agentes en AlgZoo. Las familias sin explicación publicada evitan el problema de la memoria.

## 3. Método por modelo («la vuelta»)

Para cada modelo, en este orden:
1. **Comportamiento**: acierto, en qué entradas falla, sensibilidad a cada posición.
2. **Mecanismo**: qué calcula cada neurona en cada paso; qué neuronas están activas (regiones lineales de la ReLU); qué lee la capa final.
3. **Gemelo hecho a mano**: construir los pesos desde la explicación (como los `handcrafted` de ARC) y comprobar que da el mismo acierto y que **coincide con el original entrada a entrada**. Esto es la vuelta: si el gemelo no coincide, la explicación está incompleta.
4. **Estimación mecanística del acierto**: calcular el acierto desde la explicación, sin muestrear, y compararlo con el muestreo al mismo cómputo (métrica de ARC).
5. **Nota** en `notes/<familia>_<h>_<L>.md` con la explicación, las cifras y lo que queda sin explicar.

Reglas: cero invención (lo que no se ha comprobado, se dice); cada cifra sale de un script del repo; los tests se ven en rojo antes de darlos por buenos.

## 4. Fases

**Fase 0, montaje** (primera sesión)
- `uv sync`; cargar los modelos de argmedian; script de acierto por muestreo que reproduzca la tabla de arriba.
- Decidir con Xabier la estructura del código (qué va en `src/leeuwenhoek/`).

**Fase 1, argmedian** (decisión de Xabier: empezar aquí)
- h=4, L=3 (32 parámetros, 98,46 %): primer espécimen.
- h=2, L=3 (12 parámetros, 41 %): ¿por qué falla? Un fallo explicado también vale.
- h=8, L=3 (96 parámetros), y después longitud 5.
- Opcional: checkpoints y semillas (¿aprenden todas las semillas el mismo algoritmo?).

**Fase 2, median y longest cycle**: el más pequeño que funcione de cada una.

**Fase 3, estudio de agentes** (en paralelo; **Grok solo hasta el 24-11-2026**)
- Protocolo en `agents/PROTOCOL.md` (borrador). Se diseña antes de ejecutar nada.
- Agentes: Claude Code, Codex CLI (`codex`, instalado) y Gemini CLI (`gemini`, instalado) con las cuentas de Xabier; Grok sin CLI instalado: averiguar cómo usarlo antes del 24-11.
- Cada agente, en un directorio limpio **sin nuestras notas** (contaminación), con la misma tarea, presupuesto y acceso solo a los pesos.
- Puntuación objetiva: el gemelo (¿coincide entrada a entrada?) y la estimación del acierto frente al muestreo. Registrar si el agente muestrea a escondidas o se salta las reglas.

**Fase 4, publicar**
- Post en xabier.me («AI, taken apart») y en LessWrong / Alignment Forum.
- Repo público.
- Avisar a ARC (Jacob Hilton) cuando haya la primera explicación completa. ARC tiene un formulario para investigadores visitantes [vigencia sin verificar].

## 5. Hitos

- 2 semanas: argmedian h=4 L=3 explicado, con gemelo y estimación.
- Antes del 24-11-2026: experimentos con Grok hechos (aunque sea en un modelo).
- 6 semanas: escalera de argmedian hasta donde llegue + primera ronda de agentes.
- 12 semanas: median y longest cycle, post publicado.

## 6. Riesgos

- **Matemáticas**: ReLU a trozos, inducción, teoría de la información. Es el aprendizaje que pide el Pilar 2 de ResearchPath.
- **Que alguien publique antes**: revisar GitHub y LessWrong antes de cada post.
- **Que la escalera se pare pronto**: el titular negativo sigue valiendo si está bien medido.
- **Agentes**: límites de uso de las cuentas; contaminación si ven nuestras notas.
