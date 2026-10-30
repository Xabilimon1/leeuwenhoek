# Protocolo del estudio de agentes (BORRADOR)

Sin ejecutar. Xabier lo revisa y lo aprueba antes de lanzar ningún agente.

## Pregunta

¿Puede un agente de programación explicar un modelo de AlgZoo que nadie ha explicado en público? ¿Dónde falla y dónde hace trampas?

## Agentes

| Agente | Cuenta | Estado (2026-10-30) |
|---|---|---|
| Claude Code | suscripción de Xabier | instalado |
| Codex CLI (`codex`) | ChatGPT | instalado; login por comprobar |
| Gemini CLI (`gemini`) | Google | instalado; login por comprobar |
| Grok | xAI, **solo hasta el 24-11-2026** | sin CLI instalado; buscar cómo usarlo |

## Diseño (por decidir)

- **Directorio limpio por ejecución** (`agents/runs/<agente>-<modelo>-<n>/`, fuera de git): solo los pesos, un script para cargar el modelo y la tarea. Sin `notes/` ni `docs/`.
- **Misma tarea para todos**: explicar el modelo, construir el gemelo y dar una estimación del acierto sin muestrear.
- **Presupuesto** igual para todos (tiempo o número de turnos).
- **Acceso**: ¿pueden ejecutar el modelo? Si pueden, contar los forward passes como cómputo (métrica de ARC). Si no, solo pesos.
- **Puntuación objetiva**: el gemelo coincide entrada a entrada (sí/no, % de coincidencia) y el error de su estimación frente al muestreo al mismo cómputo.
- **Trampas a registrar**: muestrear cuando no se puede, copiar los pesos originales en el «gemelo», buscar en internet explicaciones, declarar éxito sin comprobarlo.
- **Repeticiones**: al menos 3 por agente y modelo (las respuestas varían).

## Pendiente de decidir

- Modelos para el estudio: ¿los que Xabier ya ha explicado (hay referencia) o también los que no?
- Cómo se registran las conversaciones (transcripciones completas para el análisis).
- Límites de uso de cada cuenta.
