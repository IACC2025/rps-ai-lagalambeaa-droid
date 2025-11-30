import csv
import os

def determinar_ganador(e1, e2):
    """
    Determina el ganador de Piedra, Papel o Tijera.
    e1 = jugada jugador1
    e2 = jugada jugador2
    """
    if e1 == e2:
        return "empate"
    elif (e1 == "p" and e2 == "t") or \
         (e1 == "t" and e2 == "l") or \
         (e1 == "l" and e2 == "p"):
        return "jugador1"
    else:
        return "jugador2"

def convertir_opcion(c):
    """Convierte p/l/t en piedra/papel/tijera"""
    opciones = {"p": "piedra", "l": "papel", "t": "tijera"}
    return opciones.get(c, None)

def registrar_sets_reales(nombre_archivo="datos_ppt_ai.csv"):
    """
    Registro de partidas entre dos jugadores reales: Adrián e Iván
    """
    jugador1 = "Adrián"
    jugador2 = "Iván"

    nuevo_archivo = not os.path.exists(nombre_archivo)
    with open(nombre_archivo, mode="a", newline="", encoding="utf-8") as archivo:
        writer = csv.writer(archivo)
        if nuevo_archivo:
            writer.writerow(["jugador1", "jugador2", "resultado", "set", "partida",
                             "jugada_jugador1", "jugada_jugador2"])

        set_num = 1
        continuar = True

        print("=== Piedra, Papel o Tijera - Registro de Datos ===")
        print("Usa: p = piedra | l = papel | t = tijera")
        print("Se jugarán sets de 3 partidas cada vez.\n")

        while continuar:
            print(f"\n--- Set {set_num} ---")
            for partida in range(1, 4):
                while True:
                    e1 = input(f"{jugador1} → ").lower()
                    e2 = input(f"{jugador2} → ").lower()
                    if e1 not in ["p", "l", "t"] or e2 not in ["p", "l", "t"]:
                        print("❌ Entrada inválida (usa p/l/t). Vuelve a intentar.")
                        continue
                    break

                resultado = determinar_ganador(e1, e2)
                writer.writerow([jugador1, jugador2, resultado, set_num, partida,
                                 convertir_opcion(e1), convertir_opcion(e2)])
                print(f"✅ Partida {partida} guardada → {resultado}")

            seguir = input("\n¿Jugar otro set de 3 partidas? (s/n): ").lower()
            if seguir != "s":
                continuar = False
            else:
                set_num += 1

    print(f"\n📁 Todos los datos guardados en '{nombre_archivo}'. ¡Gracias por jugar!")

# Punto de entrada del script
if __name__ == "__main__":
    registrar_sets_reales()
