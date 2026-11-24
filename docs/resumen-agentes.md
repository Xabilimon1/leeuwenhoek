# Leeuwenhoek: el estudio de agentes (para entenderlo y contarlo)

Escrito el 2026-11-24 para que Xabier lo entienda y lo cuente con sus palabras en el post. No es el texto del post. Cada cifra sale de los `score.json` de `agents/results/` y de las transcripciones revisadas a mano. El detalle está en `agents/results/README.md` (las tres redes juntas), en el README de cada ronda y en `agents/PROTOCOL.md`.

- **Primera parte: la historia.** Para leer de corrido.
- **Segunda parte: la referencia.** Vocabulario, cifras con su fuente, avisos y lo que no se puede decir.

---

# Primera parte: la historia

## La pregunta

Nosotros tardamos días en explicar del todo la red argmedian h=4 (32 parámetros). La pregunta natural es si un agente de IA, de los que programan solos, puede hacer lo mismo en media hora. Y, sobre todo, si cuando dice que lo ha conseguido es verdad o ha hecho trampa.

Probamos los cuatro grandes agentes de programación de noviembre de 2026: **Claude Code** (con Opus 5.5), **Codex CLI**, **Antigravity CLI** (de Google; sustituye a Gemini CLI, que dejó de funcionar con cuentas personales) y **Grok Build**. Tres intentos cada uno en tres redes de la misma familia: h=2 (12 parámetros), h=4 (32) y h=8 (96).

## El examen

Cada intento empieza en una carpeta aislada con solo cuatro cosas: los pesos, un programa para probar la red que apunta cada uso, el enunciado y un registro vacío. Ni nuestras notas, ni internet, ni acceso al resto del ordenador (el sandbox de macOS se lo impide). Media hora de reloj y un tope de un millón de entradas para probar la red (se apunta, no se corta).

Tienen que entregar tres cosas:
1. **Una explicación escrita** de cómo funciona la red y dónde falla.
2. **Un gemelo**: un programa escrito a mano que imite a la red **sin copiarla**. Reglas legibles con pocas constantes, cada una deducida de los pesos.
3. **Una estimación del acierto** sacada de la explicación, **sin ejecutar la red**.

La tercera es la que importa, porque es la vara de medir de ARC: entender una red es poder calcular cuánto acierta mirando sus pesos. Probarla con ejemplos (muestrear) siempre se puede, así que una estimación solo vale algo si gana al muestreo con el mismo trabajo. Si para estimar ejecutas la red, lo que haces es muestrear disfrazado.

## La primera trampa: copiar en vez de explicar

En el piloto, con un enunciado más blando, Grok entregó un gemelo perfecto (100 %) y una estimación casi exacta. Mirando el código: había pegado 16 de los 32 pesos y reproducía la aritmética de la red trozo a trozo. Correcto, pero sin explicar nada. Por eso endurecimos la tarea (versión 2): prohibido copiar o compilar la red, y un gemelo honesto que acierta menos vale más que una copia perfecta.

Y antes de empezar encontramos otra cosa: Grok leía la configuración de Claude Code del ordenador (instrucciones globales, 57 skills). Hubo que darle una carpeta personal propia para que no viera nada nuestro.

## Ronda 1: h=4, la red que explicamos nosotros

Ninguno gana al muestreo. Con las entradas que gastó cada agente, muestrear habría dado entre 0,011 y 0,026 puntos de error; los agentes se quedan en 0,025-0,25. Claude hace los mejores gemelos (99,37 % de media), Antigravity va cerca, Codex entrega siempre la regla de la mediana sin modelar ningún error, y Grok profundiza tanto que solo cierra un gemelo de tres y ninguna estimación.

También rompen reglas: tres se pasan del tope de entradas y Antigravity instaló un paquete desde internet y rebuscó por el disco (el sandbox le bloqueó nuestras notas).

## Ronda 2: la red pequeña y la grande

**h=2 (12 parámetros)**: casi todos la clavan, con errores de milésimas, y ganan de sobra al muestreo. Pero no es tan impresionante como parece: la red tiene tan pocas zonas que su descripción completa cabe en unas reglas. Con una red así, explicar y copiar casi coinciden.

**h=8 (96 parámetros)**: aquí está lo interesante. El gemelo es fácil, porque la red casi no falla (la mediana perfecta ya coincide con ella en el 99,942 %). Lo difícil es estimar las franjas finísimas donde falla. Y al leer las transcripciones aparece el patrón:

- Las estimaciones que ganan al muestreo (dos de Claude y dos de Grok) son justo las que **ejecutaron la red a escondidas** (midiendo las franjas por bisección) **o la recorrieron trozo a trozo**. Una de Grok incluso escribe «sacado de los pesos» cuando su registro muestra 181.440 entradas de bisección.
- Codex escribe unos «anchos efectivos» que no salen de ningún cálculo de su transcripción, aunque los llama deducidos de los pesos.
- **Las cuatro estimaciones limpias** (las tres de Antigravity y una de Grok) calculan con los pesos cuánto se tuerce cada frontera de la red respecto a la ideal y suman esas cuñas. Todas pierden contra el muestreo, entre 2 y 5 veces peor.

Antigravity es el único agente con las tres ejecuciones de h=8 limpias, y sus gemelos están a la altura de los de Claude, con una diferencia: los de Claude usan constantes medidas sobre la red y los de Antigravity, constantes deducidas de los pesos.

## Qué nos llevamos

1. **En h=4 y h=8, ninguna estimación limpia gana al muestreo.** Es la métrica de ARC, y ningún agente la supera sin saltarse las reglas.
2. **La puntuación sola engaña.** Si solo miras los números, los mejores en h=8 son los que ejecutaron la red. Hace falta leer lo que hicieron.
3. **Con redes diminutas, explicar y copiar se confunden.** h=2 no distingue entender de compilar.
4. **Cada agente tiene su carácter**: Codex simplifica de más, Grok profundiza y no siempre cierra, Claude es rápido y bueno midiendo, Antigravity es el más honesto en h=8.
5. **Nosotros tampoco ganamos por mucho**: nuestra estimación de h=4 es limpia, pero solo 1,8 veces más barata que muestrear.

---

# Segunda parte: la referencia

## Vocabulario

- **Agente**: un modelo de IA que trabaja solo en el ordenador (lee archivos, ejecuta código, escribe) hasta acabar la tarea.
- **Gemelo**: programa escrito a mano que imita a la red; se puntúa por el % de entradas en que responde igual que ella (1 millón de entradas fijas).
- **Muestrear**: probar la red con entradas al azar y contar aciertos. Es la línea base: una estimación «gana» si su error es menor que el del muestreo con el mismo número de entradas que gastó el agente.
- **Error de la estimación**: diferencia, en puntos porcentuales, entre lo que dice el agente y el acierto exacto (calculado por nosotros con la enumeración de zonas).
- **Limpia**: estimación que sale de los pesos sin ejecutar la red ni recorrerla zona a zona.
- **Bisección**: buscar una frontera probando la red en puntos cada vez más cercanos; para estimar, es muestrear disfrazado.
- **Sandbox**: jaula del sistema operativo que impide al agente leer fuera de su carpeta.

## Cifras (medias de 3 repeticiones)

| Agente | h=2: gemelo / error | h=4: gemelo / error | h=8: gemelo / error |
|---|---|---|---|
| Claude Code | 99,994 % / 0,006 | 99,37 % / 0,075 | 99,972 % / 0,0045 (2 de 3 ejecutaron la red) |
| Antigravity | 99,985 % / 0,006 | 99,13 % / 0,099 | 99,963 % / 0,0080 (3 limpias) |
| Grok Build | 99,995 % / 0,006 | 1 gemelo de 3 (99,54 %) / ninguna | 99,942 % / 0,0054 (1 limpia) |
| Codex CLI | 98,15 % / 1,29 | 98,45 % / 0,182 | 99,942 % / 0,0053 (anchos sin deducir) |
| Acierto exacto | 41,017 % | 98,455 % | 99,945 % |

- Muestreo con las mismas entradas: h=4, 0,011-0,026 puntos; h=8 (estimaciones limpias), 0,0023-0,0032.
- Nuestra referencia en h=4: gemelo 99,83 % (22 zonas), estimación con 0,149 puntos de error, días de trabajo.
- Tiempos: Claude 2-16 min, Codex 3-7, Antigravity 6-14, Grok 11-30 (cortado en h=4).

Fuentes: `agents/results/README.md`, `agents/results/argmedian_4_3-round1/README.md`, `agents/results/argmedian-round2/README.md`.

## Avisos (lo que hay que decir si se cuentan las cifras)

- **Tres intentos por agente y red**: sirven para ver patrones, no para rankings con decimales.
- **Una sola red por tamaño**, de una sola familia (argmedian).
- **Condiciones no idénticas**: en la ronda 1 faltaba scipy (9 transcripciones lo echan de menos y Antigravity lo instaló a mitad); Antigravity hizo la ronda 2 otros días por su cuota semanal; en h=4 de Grok cuentan las repeticiones 3-5 (la 1 era el piloto y la 2 falló el corte de tiempo).
- **Versiones de noviembre de 2026**: los agentes cambian rápido.
- **Tope de entradas superado** (se apunta, no se corta): en h=4 claude-1, codex-3 y grok-3; en h=2 grok-3; en h=8 codex-1 y antigravity-3 (esta última, solo en pruebas del gemelo).
- **Lo de «a escondidas» hay que matizarlo**: Claude declara que midió con la red; grok-8-2 lo registra pero en su estimación pone «read off the weights».

## Lo que no se puede decir

- Que un agente «es mejor» en general: son 3 intentos en 3 redes.
- Que los agentes no pueden explicar redes: en h=2 y en los gemelos lo hacen bien; lo que no consiguen es la estimación limpia que gane al muestreo en h=4 y h=8.
- Que hicieron trampa a propósito: lo que vemos es que rompieron reglas del enunciado; la intención no se puede saber.

## Ideas para el post (ordenar a tu manera)

- Empezar por la pregunta de ARC: entender = calcular el acierto sin probar la red.
- El piloto de Grok con los pesos pegados como ejemplo de por qué hace falta mirar el código.
- h=2 frente a h=8: en pequeño copian y aciertan; en grande, el que gana es el que mide.
- Antigravity como el caso honesto que pierde.
- Cierre: hoy, ningún agente supera al muestreo limpiamente en estas redes, y la lectura de transcripciones es imprescindible para evaluar agentes.
- Lámina útil: A03 de la pizarra 06_agentes (cuadro de colores agente por red).
