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
(Explica aqui como recogiste los datos. Si usaste un programa,
describe brevemente como funciona. Si fue manual, explica el proceso.)
Este programa permite registrar partidas de Piedra, Papel o Tijera entre dos jugadores (Adrián e Iván) y almacenar los resultados en un archivo CSV para su posterior análisis.
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
- [ ] `sesion` - ID de sesion de juego
- [ ] `resultado` - victoria/derrota/empate
- [ ] Otro: _________________

### Descripcion de datos adicionales:

```
(Si capturaste datos extra, explica aqui por que y como los usas)


```

---

## Estadisticas del dataset

- **Total de rondas:** _____
- **Numero de sesiones/partidas:** _____
- **Contra cuantas personas diferentes:** _____

### Tipo de IA:

- [ ] **IA Especifica**: Entrenada para ganar a UNA persona concreta
  - Nombre/identificador del oponente: _________________
- [ ] **IA General**: Entrenada para ganar a cualquier oponente
