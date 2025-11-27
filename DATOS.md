# Documentacion de Recogida de Datos

**Alumno:** Iván Ezquerro Muñoz y Adrián Laga Lambea

## Formato del CSV

Tu archivo `data/partidas.csv` debe tener **minimo** estas columnas:

| Columna | Descripcion | Ejemplo |
|---------|-------------|---------|
| `numero_ronda` | Numero de la ronda (1, 2, 3...) | 1 |
| `jugada_j1` | Jugada del jugador 1 | piedra |
| `jugada_j2` | Jugada del jugador 2 (oponente) | papel |

### Ejemplo de CSV minimo:

```csv
numero_ronda,jugada_j1,jugada_j2
1,piedra,papel
2,tijera,piedra
3,papel,papel
4,piedra,tijera
...
```

---

## Como recogiste los datos?

Marca con [x] el metodo usado y describe brevemente:

### Metodo de recogida:

- [X] **Programa propio**: Cree un programa para jugar y guardar datos
- [ ] **Manual**: Jugue partidas y apunte los resultados a mano
- [ ] **Aplicacion/Web externa**: Use una app y exporte los datos
- [ ] **Otro**: _________________

### Descripcion del proceso:

```
Se hizo un programa para la recogida de datos en Python,
el cual se puede encontrar en la carpeta src con el nombre "rpsData.py".
El programa funciona de la siguiente manera: 
Este programa permite registrar partidas de Piedra, Papel o Tijera entre dos jugadores (Adrián e Iván) 
y almacenar los resultados en un archivo CSV para su posterior análisis.
Cómo funciona el programa
1. Entrada de datos
Los jugadores introducen sus jugadas mediante el teclado usando las letras:
    p = piedra
    l = papel (del inglés leaf o inspirado en hoja)
    t = tijera
2. Determinación del ganador
    La función determinar_ganador() evalúa las jugadas según las reglas clásicas:
        Piedra vence a Tijera
        Tijera vence a Papel
        Papel vence a Piedra
        Jugadas iguales resultan en empate
3. Estructura de sets
    Cada set consta de 3 partidas
    Después de cada set, el programa pregunta si desean continuar
    Los jugadores pueden registrar múltiples sets en una misma sesión
4. Almacenamiento de datos
    Los resultados se guardan en datos_ppt_ai.csv con la siguiente estructura:
5. Validación de entradas
    El programa verifica que las entradas sean válidas (p/l/t) y solicita reingresar datos en caso de error, asegurando la integridad de los datos registrados.

Ventajas de este método
    Automático: No requiere transcripción manual posterior
    Estructurado: Los datos quedan organizados desde el inicio
    Escalable: Permite registrar tantos sets como se desee
    Trazable: Cada partida queda identificada por set y número
    Reutilizable: El CSV puede analizarse con pandas, Excel o cualquier herramienta de análisis de datos
```
---

## Datos adicionales capturados

Si capturaste datos extra ademas de los basicos, marcalos aqui:

- [ ] `tiempo_reaccion_ms` - Tiempo que tardo el jugador en responder
- [ ] `timestamp` - Fecha/hora de cada jugada
- [X] `sesion` - ID de sesion de juego
- [X] `resultado` - victoria/derrota/empate
- [ ] Otro: _________________

### Descripcion de datos adicionales:

```
Capturamos los set de partida, cada set es de tres partidas. 
Los capturamos para poder ver los resultados
de cada set y tomarnos las partidas mas en serio sin que
sean un montón de partidas sin objetivo claro.
```

---

## Estadisticas del dataset

- **Total de rondas:** 162
- **Numero de sesiones/partidas:** 54
- **Contra cuantas personas diferentes:** 1

### Tipo de IA:

- [X] **IA Especifica**: Entrenada para ganar a UNA persona concreta
  - Nombre/identificador del oponente: Iván
- [ ] **IA General**: Entrenada para ganar a cualquier oponente
