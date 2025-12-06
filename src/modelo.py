"""
RPSAI - Modelo de IA para Piedra, Papel o Tijera (Versión Ensamble v5.0)
========================================================================
ARQUITECTURA DE VOTO PONDERADO:
Combina predicciones de 3 fuentes distintas para tomar la decisión más robusta.

1. Modelo ML (Gradient Boosting): Peso dinámico basado en confianza.
2. Cadenas de Markov (Live): Detecta secuencias exactas en la partida actual.
3. Anti-Spam / Frecuencia: Penaliza jugadas sobre-utilizadas por el rival.

MEJORA CLAVE: Reducción de empates mediante predicción probabilística combinada.
"""

import os
import pickle
import warnings
from pathlib import Path
from collections import defaultdict

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier

warnings.filterwarnings("ignore", message="X does not have valid feature names")
warnings.filterwarnings("ignore", category=FutureWarning)

# =============================================================================
# CONFIGURACIÓN
# =============================================================================

try:
    BASE_DIR = Path(__file__).parent.parent
except NameError:
    BASE_DIR = Path.cwd()

RUTA_DATOS = BASE_DIR / "data" / "partidas.csv"
RUTA_MODELO = BASE_DIR / "models" / "modelo_entrenado.pkl"

JUGADA_A_NUM = {"piedra": 0, "papel": 1, "tijera": 2}
NUM_A_JUGADA = {0: "piedra", 1: "papel", 2: "tijera"}
SENTINEL_VALUE = -1

GANA_A = {"piedra": "tijera", "papel": "piedra", "tijera": "papel"}
PIERDE_CONTRA = {"piedra": "papel", "papel": "tijera", "tijera": "piedra"}

# =============================================================================
# 1. PREPARACIÓN DE DATOS
# =============================================================================

def cargar_datos(ruta_csv: Path = None) -> pd.DataFrame:
    if ruta_csv is None: ruta_csv = RUTA_DATOS
    if not ruta_csv.exists():
        # Si no existe, creamos un dummy para que no falle y la IA aprenda en vivo
        print("⚠️ No hay CSV. Se creará un DataFrame vacío.")
        return pd.DataFrame(columns=['jugada_jugador1', 'jugada_jugador2'])

    df = pd.read_csv(ruta_csv)
    cols = df.columns
    if 'jugada_jugador' in cols:
        df = df.rename(columns={'jugada_jugador': 'jugada_jugador1', 'jugada_oponente': 'jugada_jugador2'})
    elif len(cols) >= 2 and 'jugada_jugador1' not in cols:
        df.columns = ['jugada_jugador1', 'jugada_jugador2'] + list(cols[2:])
    return df

def preparar_datos(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) == 0: return df
    df = df.copy().rename(columns={'jugada_jugador1': 'jugada_j1', 'jugada_jugador2': 'jugada_j2'})
    validas = set(JUGADA_A_NUM.keys())
    df = df[df['jugada_j1'].isin(validas) & df['jugada_j2'].isin(validas)]

    df['jugada_j1_num'] = df['jugada_j1'].map(JUGADA_A_NUM)
    df['jugada_j2_num'] = df['jugada_j2'].map(JUGADA_A_NUM)
    df['proxima_jugada_j2'] = df['jugada_j2_num'].shift(-1)

    def calc_res(row):
        j1, j2 = row['jugada_j1'], row['jugada_j2']
        if j1 == j2: return 0
        return 1 if GANA_A[j1] == j2 else -1

    df['resultado'] = df.apply(calc_res, axis=1)
    return df.dropna(subset=['proxima_jugada_j2'])

# =============================================================================
# 2. FEATURE ENGINEERING
# =============================================================================

def crear_features(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) < 5: return df
    df = df.copy()

    # Lags
    df['j2_lag1'] = df['jugada_j2_num'].shift(1)
    df['j2_lag2'] = df['jugada_j2_num'].shift(2)
    df['j1_lag1'] = df['jugada_j1_num'].shift(1)
    df['j1_lag2'] = df['jugada_j1_num'].shift(2)

    # Secuencias complejas
    df['seq_j2_2'] = df['jugada_j2_num'].shift(1)*10 + df['jugada_j2_num'].shift(2)
    df['seq_mix_2'] = df['jugada_j2_num'].shift(1)*10 + df['jugada_j1_num'].shift(1) # Tu jugada + Mi jugada

    # Frecuencias Rolling
    for n in [0, 1, 2]:
        df[f'roll_freq_{n}'] = (df['jugada_j2_num'] == n).rolling(10, min_periods=1).mean().shift(1)

    # Lógica WSLS (Win-Stay, Lose-Shift)
    # Si ganó la anterior (res=-1 para la IA), ¿repitió jugada?
    df['wsls_trend'] = np.where(
        (df['resultado'].shift(2) == -1) & (df['jugada_j2_num'].shift(1) == df['jugada_j2_num'].shift(2)),
        1, 0
    )

    return df

def seleccionar_features(df: pd.DataFrame):
    if len(df) < 10: return None, None
    cols = [
        'j2_lag1', 'j2_lag2', 'j1_lag1', 'j1_lag2',
        'seq_j2_2', 'seq_mix_2',
        'roll_freq_0', 'roll_freq_1', 'roll_freq_2',
        'wsls_trend'
    ]
    return df[cols].fillna(SENTINEL_VALUE), df['proxima_jugada_j2'].astype(int)

# =============================================================================
# 3. ENTRENAMIENTO
# =============================================================================

def entrenar_modelo(X, y):
    if X is None or len(X) < 10:
        print("⚠️ Pocos datos para entrenar ML. Se usará solo lógica en vivo.")
        return None

    # Gradient Boosting robusto
    modelo = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )
    modelo.fit(X, y)
    return modelo

# =============================================================================
# 4. CEREBRO HÍBRIDO (ENSAMBLE)
# =============================================================================

class JugadorIA:
    def __init__(self, ruta_modelo=None):
        self.historial = []

        # --- CEREBRO MARKOV (Live) ---
        # Almacena transiciones de patrones: { "Patron_Previo": {Jugada_Siguiente: Conteo} }
        self.markov_chain = defaultdict(lambda: defaultdict(float))
        self.decay_factor = 0.9 # Los patrones viejos valen menos

        # --- CEREBRO ML ---
        self.feature_cols = [
            'j2_lag1', 'j2_lag2', 'j1_lag1', 'j1_lag2',
            'seq_j2_2', 'seq_mix_2',
            'roll_freq_0', 'roll_freq_1', 'roll_freq_2',
            'wsls_trend'
        ]

        if ruta_modelo is None: ruta_modelo = RUTA_MODELO
        try:
            with open(ruta_modelo, "rb") as f:
                self.modelo = pickle.load(f)
            print("✅ Cerebro ML cargado.")
        except:
            self.modelo = None
            print("⚠️ Jugando sin Cerebro ML (usando solo Markov + Estadística).")

    def registrar_ronda(self, j1, j2):
        # 1. Actualizar Cadenas de Markov (Aprender en Vivo)
        if len(self.historial) >= 2:
            # Definimos varios contextos (N-Gramas)

            # Contexto A: Tus últimas 2 jugadas
            last_j2 = JUGADA_A_NUM[self.historial[-1][1]]
            prev_j2 = JUGADA_A_NUM[self.historial[-2][1]]
            key_A = f"J2_{prev_j2}_{last_j2}"

            # Contexto B: Tu última jugada + Mi última jugada
            last_j1 = JUGADA_A_NUM[self.historial[-1][0]]
            key_B = f"MIX_{last_j1}_{last_j2}"

            # La jugada que acabas de hacer ahora
            curr_j2 = JUGADA_A_NUM[j2]

            # Actualizamos contadores con Decay (olvido gradual)
            for k in self.markov_chain:
                for move in self.markov_chain[k]:
                    self.markov_chain[k][move] *= self.decay_factor

            self.markov_chain[key_A][curr_j2] += 1.0
            self.markov_chain[key_B][curr_j2] += 1.0

        self.historial.append((j1, j2))

    def _get_markov_probs(self):
        """Devuelve probabilidades basadas en patrones recientes."""
        probs = np.zeros(3)
        if len(self.historial) < 2: return probs # [0,0,0]

        last_j2 = JUGADA_A_NUM[self.historial[-1][1]]
        prev_j2 = JUGADA_A_NUM[self.historial[-2][1]]
        last_j1 = JUGADA_A_NUM[self.historial[-1][0]]

        key_A = f"J2_{prev_j2}_{last_j2}"
        key_B = f"MIX_{last_j1}_{last_j2}"

        # Sumamos la evidencia de ambos contextos
        total_evidence = defaultdict(float)

        for k in [key_A, key_B]:
            if k in self.markov_chain:
                for move, score in self.markov_chain[k].items():
                    total_evidence[move] += score

        # Normalizar a probabilidades
        total_score = sum(total_evidence.values())
        if total_score > 0:
            for move, score in total_evidence.items():
                probs[move] = score / total_score

        return probs

    def _get_ml_probs(self):
        """Devuelve probabilidades del modelo Gradient Boosting."""
        if self.modelo is None or len(self.historial) < 3:
            return np.array([0.33, 0.33, 0.33])

        try:
            # Reconstruir vector de features al vuelo
            j1s = [JUGADA_A_NUM[x[0]] for x in self.historial]
            j2s = [JUGADA_A_NUM[x[1]] for x in self.historial]

            f = {}
            f['j2_lag1'] = j2s[-1]
            f['j2_lag2'] = j2s[-2]
            f['j1_lag1'] = j1s[-1]
            f['j1_lag2'] = j1s[-2]
            f['seq_j2_2'] = j2s[-1]*10 + j2s[-2]
            f['seq_mix_2'] = j2s[-1]*10 + j1s[-1]

            last_10 = j2s[-11:-1] if len(j2s)>1 else []
            L = len(last_10) if last_10 else 1
            f['roll_freq_0'] = last_10.count(0)/L
            f['roll_freq_1'] = last_10.count(1)/L
            f['roll_freq_2'] = last_10.count(2)/L

            # WSLS
            # Resultado hace 2 turnos (-1 es que ganó J2)
            if len(self.historial) >= 2:
                # Calcular resultado ronda anterior
                r_prev = -1 # Default
                if j1s[-2] == j2s[-2]: r_prev = 0
                elif GANA_A[NUM_A_JUGADA[j1s[-2]]] == NUM_A_JUGADA[j2s[-2]]: r_prev = 1
                else: r_prev = -1

                f['wsls_trend'] = 1 if (r_prev == -1 and j2s[-1] == j2s[-2]) else 0
            else:
                f['wsls_trend'] = 0

            feat_vector = np.array([f.get(c, SENTINEL_VALUE) for c in self.feature_cols]).reshape(1, -1)
            return self.modelo.predict_proba(feat_vector)[0]
        except:
            return np.array([0.33, 0.33, 0.33])

    def decidir_jugada(self):
        # 1. Anti-Spam Hardcoded (Seguridad Máxima)
        # Si repite 3 veces, asumimos la 4ta.
        if len(self.historial) >= 3:
            ultimas_3 = [x[1] for x in self.historial[-3:]]
            if ultimas_3[0] == ultimas_3[1] == ultimas_3[2]:
                pred_spam = JUGADA_A_NUM[ultimas_3[0]]
                return PIERDE_CONTRA[NUM_A_JUGADA[pred_spam]]

        # 2. OBTENER PROBABILIDADES DE CEREBROS
        probs_markov = self._get_markov_probs() # Cerebro Rápido
        probs_ml = self._get_ml_probs()         # Cerebro Lento

        # 3. PONDERACIÓN (VOTACIÓN)
        # En partidas cortas, Markov vale más (0.6) que el ML (0.4)
        # porque el ML necesita más historia para ser útil.
        peso_markov = 0.65
        peso_ml = 0.35

        # Si Markov no tiene datos (inicio partida), confiamos más en ML o Random
        if np.sum(probs_markov) == 0:
            final_probs = probs_ml
        else:
            final_probs = (probs_markov * peso_markov) + (probs_ml * peso_ml)

        # 4. DECISIÓN FINAL CON ANTI-ESPEJO
        # No solo tomamos el máximo, analizamos el riesgo de empate.
        pred_idx = np.argmax(final_probs)

        # Heurística: Si la probabilidad de predicción es baja (<45%),
        # y esa predicción es igual a mi última jugada, hay riesgo de "Loop de Empate".
        # Intentamos cambiar a la segunda mejor opción.
        if len(self.historial) > 0:
            mi_ultima = JUGADA_A_NUM[self.historial[-1][0]] # Lo que YO jugué
            op_ultima = JUGADA_A_NUM[self.historial[-1][1]] # Lo que TÚ jugaste

            # Si predigo que vas a repetir tu última jugada, y la certeza es baja...
            if pred_idx == op_ultima and final_probs[pred_idx] < 0.5:
                # A veces la gente cambia tras un empate/derrota.
                # Subimos un poco el peso de la opción que gana a tu última jugada (Counter-Shift)
                shift_idx = (op_ultima + 1) % 3 # Ciclo 0->1->2->0
                final_probs[shift_idx] += 0.15
                pred_idx = np.argmax(final_probs)

        return PIERDE_CONTRA[NUM_A_JUGADA[pred_idx]]

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*60)
    print("   RPSAI v5.0 - ENSAMBLE PONDERADO (Markov + GB + Anti-Spam)")
    print("="*60)

    try:
        df = cargar_datos()
        df = preparar_datos(df)

        modelo = None
        if len(df) > 10:
            print("\n⚙️ Entrenando Cerebro ML Base...")
            df = crear_features(df)
            X, y = seleccionar_features(df)
            if X is not None:
                modelo = entrenar_modelo(X, y)
        else:
            print("\n⚠️ Sin historial previo suficiente. La IA aprenderá 100% en vivo.")

        # Guardar
        RUTA_MODELO.parent.mkdir(parents=True, exist_ok=True)
        with open(RUTA_MODELO, "wb") as f:
            pickle.dump(modelo, f)

        print("\n✅ ¡IA Lista! La arquitectura v5.0 está activa.")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()