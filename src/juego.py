"""
RPSAI - Juego Interactivo
=========================

Este script permite jugar contra tu IA de forma interactiva.
Es diferente al evaluador: aquí puedes jugar libremente sin presión.

Uso:
    python src/juego.py
"""

import sys
from pathlib import Path

# Agregar el directorio src al path
sys.path.insert(0, str(Path(__file__).parent))

from modelo import JugadorIA, GANA_A
import random

# Mapeo de entrada a jugada
ENTRADA_A_JUGADA = {
    "1": "piedra", "2": "papel", "3": "tijera",
    "p": "piedra", "a": "papel", "t": "tijera",
    "piedra": "piedra", "papel": "papel", "tijera": "tijera"
}

# Emojis para hacer el juego más visual
EMOJIS = {
    "piedra": "🪨",
    "papel": "📄",
    "tijera": "✂️"
}


def limpiar_pantalla():
    """Limpia la pantalla (funciona en Windows, Linux y Mac)."""
    import os
    os.system('cls' if os.name == 'nt' else 'clear')


def mostrar_banner():
    """Muestra el banner del juego."""
    print("=" * 60)
    print("     🤖 PIEDRA, PAPEL O TIJERA - IA 🤖")
    print("=" * 60)
    print()


def leer_jugada_humano() -> str:
    """Lee la jugada del humano con interfaz amigable."""
    print("\n" + "-" * 40)
    print("  [1/p] 🪨 Piedra")
    print("  [2/a] 📄 Papel")
    print("  [3/t] ✂️  Tijera")
    print("  [q] Salir")
    print("-" * 40)

    while True:
        entrada = input("\n👉 Tu jugada: ").lower().strip()

        if entrada == 'q':
            return None

        if entrada in ENTRADA_A_JUGADA:
            return ENTRADA_A_JUGADA[entrada]

        print("❌ Jugada no válida. Intenta de nuevo.")


def obtener_resultado(jugada_humano: str, jugada_ia: str) -> tuple:
    """
    Obtiene el resultado del juego.

    Returns:
        (resultado_texto, puntos_humano, puntos_ia)
        resultado_texto: 'humano', 'ia' o 'empate'
    """
    if jugada_humano == jugada_ia:
        return 'empate', 0, 0
    elif GANA_A[jugada_humano] == jugada_ia:
        return 'humano', 1, 0
    else:
        return 'ia', 0, 1


def mostrar_resultado(jugada_humano: str, jugada_ia: str, resultado: str):
    """Muestra el resultado de la ronda de forma visual."""
    print("\n" + "=" * 40)
    print(f"  Tú:  {EMOJIS[jugada_humano]}  {jugada_humano.upper()}")
    print(f"  IA:  {EMOJIS[jugada_ia]}  {jugada_ia.upper()}")
    print("-" * 40)

    if resultado == 'humano':
        print("  🎉 ¡GANASTE! 🎉")
    elif resultado == 'ia':
        print("  🤖 ¡LA IA GANA! 🤖")
    else:
        print("  🤝 ¡EMPATE! 🤝")

    print("=" * 40)


def mostrar_estadisticas(puntos_humano: int, puntos_ia: int, empates: int, ronda: int):
    """Muestra las estadísticas actuales."""
    total = puntos_humano + puntos_ia + empates

    print(f"\n📊 ESTADÍSTICAS (Ronda {ronda})")
    print("-" * 40)
    print(f"  Tú:      {puntos_humano} victorias")
    print(f"  IA:      {puntos_ia} victorias")
    print(f"  Empates: {empates}")

    if puntos_humano + puntos_ia > 0:
        winrate_humano = puntos_humano / (puntos_humano + puntos_ia) * 100
        winrate_ia = puntos_ia / (puntos_humano + puntos_ia) * 100
        print(f"\n  Tu winrate: {winrate_humano:.1f}%")
        print(f"  IA winrate: {winrate_ia:.1f}%")

    print("-" * 40)


def mostrar_prediccion(prediccion: str, revelar: bool = False):
    """Muestra la predicción de la IA (opcional)."""
    if revelar:
        print(f"\n🔮 La IA predijo que jugarías: {EMOJIS[prediccion]} {prediccion.upper()}")


def jugar():
    """Función principal del juego."""
    limpiar_pantalla()
    mostrar_banner()

    print("¡Bienvenido! Vas a jugar contra una IA entrenada.")
    print("La IA intentará predecir tus movimientos.")
    print()

    # Cargar la IA
    try:
        ia = JugadorIA()
        if ia.modelo is None:
            print("⚠️  ADVERTENCIA: No se encontró modelo entrenado.")
            print("   La IA jugará de forma ALEATORIA.")
            print("   Entrena tu modelo primero: python src/modelo.py")
        else:
            print("✅ Modelo de IA cargado correctamente.")
    except Exception as e:
        print(f"⚠️  Error al cargar la IA: {e}")
        print("   La IA jugará de forma ALEATORIA.")
        ia = JugadorIA()

    # Preguntar modo de juego
    print("\n¿Quieres ver las predicciones de la IA? (s/n)")
    revelar = input("👉 ").lower().strip() == 's'

    input("\n📢 Presiona ENTER para comenzar...")

    # Variables del juego
    puntos_humano = 0
    puntos_ia = 0
    empates = 0
    ronda = 0

    # Bucle principal
    while True:
        ronda += 1

        # Mostrar estadísticas
        if ronda > 1:
            mostrar_estadisticas(puntos_humano, puntos_ia, empates, ronda - 1)

        # La IA hace su predicción y decide jugada
        prediccion_ia = ia.predecir_jugada_oponente()
        jugada_ia = ia.decidir_jugada()

        # El humano juega
        jugada_humano = leer_jugada_humano()

        # Salir del juego
        if jugada_humano is None:
            break

        # Mostrar predicción (si está activado)
        if revelar:
            mostrar_prediccion(prediccion_ia, revelar=True)

        # Determinar resultado
        resultado, pts_h, pts_ia = obtener_resultado(jugada_humano, jugada_ia)

        # Mostrar resultado
        mostrar_resultado(jugada_humano, jugada_ia, resultado)

        # Actualizar puntuación
        puntos_humano += pts_h
        puntos_ia += pts_ia
        if resultado == 'empate':
            empates += 1

        # Registrar en historial de la IA
        ia.registrar_ronda(jugada_humano, jugada_ia)

        # Pausa para ver resultado
        input("\n⏎  Presiona ENTER para siguiente ronda...")

    # Mostrar estadísticas finales
    print("\n" + "=" * 60)
    print("   🏁 JUEGO TERMINADO 🏁")
    print("=" * 60)

    mostrar_estadisticas(puntos_humano, puntos_ia, empates, ronda - 1)

    # Determinar ganador
    print("\n🏆 RESULTADO FINAL:")
    if puntos_humano > puntos_ia:
        print("   ¡GANASTE EL JUEGO! 🎉")
    elif puntos_ia > puntos_humano:
        print("   ¡LA IA GANÓ EL JUEGO! 🤖")
    else:
        print("   ¡EMPATE TÉCNICO! 🤝")

    print("\n¡Gracias por jugar!")
    print("=" * 60)


def main():
    """Punto de entrada."""
    try:
        jugar()
    except KeyboardInterrupt:
        print("\n\n🛑 Juego interrumpido. ¡Hasta pronto!")
    except Exception as e:
        print(f"\n❌ Error inesperado: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()