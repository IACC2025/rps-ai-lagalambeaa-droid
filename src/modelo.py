"""
RPSAI - Modelo de IA para Piedra, Papel o Tijera (Versión Híbrida v4.0)
=======================================================================
ARQUITECTURA:
1. Nivel 0 (Reflejo): Anti-Spam agresivo (si repites 2 veces, te caza).
2. Nivel 1 (Memoria Viva): Búsqueda de N-Gramas en la sesión actual.
3. Nivel 2 (Intuición): Modelo Gradient Boosting pre-entrenado.

Esta versión APRENDE mientras juega.
"""

import os
import pickle
import warnings
from pathlib import Path
from collections import defaultdict

import pandas as pd
import numpy as np

# Modelos
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
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
# PARTE 1: PREPARACIÓN DE DATOS
# =============================================================================

def cargar_datos(ruta_csv: Path = None) -> pd.DataFrame:
    if ruta_csv is None: ruta_csv = RUTA_DATOS
    if not ruta_csv.exists():
        raise FileNotFoundError(f"Falta archivo: {ruta_csv}")

    df = pd.read_csv(ruta_csv)
    # Normalizar columnas
    cols = df.columns
    if 'jugada_jugador' in cols:
        df = df.rename(columns={'jugada_jugador': 'jugada_jugador1', 'jugada_oponente': 'jugada_jugador2'})
    elif len(cols) >= 2 and 'jugada_jugador1' not in cols:
        df.columns = ['jugada_jugador1', 'jugada_jugador2'] + list(cols[2:])

    return df

def preparar_datos(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().rename(columns={'jugada_jugador1': 'jugada_j1', 'jugada_jugador2': 'jugada_j2'})
    validas = set(JUGADA_A_NUM.keys())
    df = df[df['jugada_j1'].isin(validas) & df['jugada_j2'].isin(validas)]

    df['jugada_j1_num'] = df['jugada_j1'].map(JUGADA_A_NUM)
    df['jugada_j2_num'] = df['jugada_j2'].map(JUGADA_A_NUM)
    df['proxima_jugada_j2'] = df['jugada_j2_num'].shift(-1)

    # Resultado (1: Gana J1, -1: Gana J2)
    def calc_res(row):
        j1, j2 = row['jugada_j1'], row['jugada_j2']
        if j1 == j2: return 0
        return 1 if GANA_A[j1] == j2 else -1

    df['resultado'] = df.apply(calc_res, axis=1)
    return df.dropna(subset=['proxima_jugada_j2'])


# =============================================================================
# PARTE 2: FEATURE ENGINEERING
# =============================================================================

def crear_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1. Lags (Historial inmediato)
    df['j2_lag1'] = df['jugada_j2_num'].shift(1)
    df['j2_lag2'] = df['jugada_j2_num'].shift(2)
    df['j1_lag1'] = df['jugada_j1_num'].shift(1)

    # 2. Secuencias
    df['seq_j2_2'] = df['jugada_j2_num'].shift(1)*10 + df['jugada_j2_num'].shift(2)

    # 3. Frecuencias (Window 10)
    for n in [0, 1, 2]:
        df[f'roll_freq_{n}'] = (df['jugada_j2_num'] == n).rolling(10, min_periods=1).mean().shift(1)

    # 4. Patrones de Respuesta
    # ¿Qué jugó J2 cuando ganó/perdió la última vez?
    df['j2_tras_ganar'] = np.where(df['resultado'].shift(2) == -1, df['jugada_j2_num'].shift(1), -1)
    df['j2_tras_perder'] = np.where(df['resultado'].shift(2) == 1, df['jugada_j2_num'].shift(1), -1)

    # 5. CWP (Counter Winner Previous)
    # ¿Juega lo que le ganó a su jugada anterior?
    def get_counter(val):
        if pd.isna(val) or val == -1: return -1
        return JUGADA_A_NUM[PIERDE_CONTRA[NUM_A_JUGADA[int(val)]]]

    df['cwp_signal'] = df['j2_lag1'].apply(get_counter)

    return df

def seleccionar_features(df: pd.DataFrame):
    cols = [
        'j2_lag1', 'j2_lag2', 'j1_lag1',
        'seq_j2_2',
        'roll_freq_0', 'roll_freq_1', 'roll_freq_2',
        'j2_tras_ganar', 'j2_tras_perder',
        'cwp_signal'
    ]
    return df[cols].fillna(SENTINEL_VALUE), df['proxima_jugada_j2'].astype(int)


# =============================================================================
# PARTE 3: ENTRENAMIENTO (MODELO BASE)
# =============================================================================

def entrenar_modelo(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Usamos Gradient Boosting (mejor generalización que Random Forest en RPS)
    modelo = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42)
    modelo.fit(X_train, y_train)

    acc = accuracy_score(y_test, modelo.predict(X_test))
    print(f"\n📊 Modelo Base (Gradient Boosting) Accuracy: {acc:.2%}")
    return modelo


# =============================================================================
# PARTE 4: CEREBRO HÍBRIDO (CLASE JUGADOR IA)
# =============================================================================

class JugadorIA:
    def __init__(self, ruta_modelo=None):
        self.historial = [] # [(j1, j2), ...]

        # --- MEMORIA VIVA (Live Learning) ---
        # Almacena secuencias: {(prev_j2, prev_j1): {next_j2: count}}
        self.memoria_ngramas = defaultdict(lambda: defaultdict(int))

        # Modelo Estático
        self.feature_cols = [
            'j2_lag1', 'j2_lag2', 'j1_lag1',
            'seq_j2_2',
            'roll_freq_0', 'roll_freq_1', 'roll_freq_2',
            'j2_tras_ganar', 'j2_tras_perder',
            'cwp_signal'
        ]

        if ruta_modelo is None: ruta_modelo = RUTA_MODELO
        try:
            with open(ruta_modelo, "rb") as f:
                self.modelo = pickle.load(f)
            print("✅ Modelo cargado.")
        except:
            self.modelo = None
            print("⚠️ Jugando sin modelo base (solo aprendizaje en vivo).")

    def registrar_ronda(self, j1, j2):
        # Antes de guardar, aprendemos de lo que ACABA de pasar
        if len(self.historial) >= 1:
            prev_j1_num = JUGADA_A_NUM[self.historial[-1][0]]
            prev_j2_num = JUGADA_A_NUM[self.historial[-1][1]]
            curr_j2_num = JUGADA_A_NUM[j2]

            # Key: Estado Anterior (Tú jugada anterior, Mi jugada anterior)
            key = (prev_j2_num, prev_j1_num)
            self.memoria_ngramas[key][curr_j2_num] += 1

        self.historial.append((j1, j2))

    def _get_live_prediction(self):
        """Busca patrones en la memoria viva de esta partida."""
        if len(self.historial) < 1: return None

        # Estado actual
        last_j1 = JUGADA_A_NUM[self.historial[-1][0]]
        last_j2 = JUGADA_A_NUM[self.historial[-1][1]]
        key = (last_j2, last_j1)

        # ¿Hemos visto esta situación antes en esta partida?
        if key in self.memoria_ngramas:
            counts = self.memoria_ngramas[key]
            # Devuelve la jugada que más veces ha hecho el oponente en esta situación
            predicted_move = max(counts, key=counts.get)
            confidence = counts[predicted_move]

            # Solo confiamos si lo ha hecho al menos una vez (se puede subir el umbral)
            if confidence >= 1:
                return predicted_move
        return None

    def _get_features(self):
        N = len(self.historial)
        if N < 3: return np.full((1, len(self.feature_cols)), SENTINEL_VALUE)

        j1s = [JUGADA_A_NUM[x[0]] for x in self.historial]
        j2s = [JUGADA_A_NUM[x[1]] for x in self.historial]

        # Helper resultados
        res = []
        for a, b in zip(j1s, j2s):
            if a == b: res.append(0)
            elif GANA_A[NUM_A_JUGADA[a]] == NUM_A_JUGADA[b]: res.append(1)
            else: res.append(-1)

        f = {}
        f['j2_lag1'] = j2s[-1]
        f['j2_lag2'] = j2s[-2]
        f['j1_lag1'] = j1s[-1]
        f['seq_j2_2'] = j2s[-1]*10 + j2s[-2]

        last_10 = j2s[-11:-1] if N > 1 else []
        L = len(last_10) if last_10 else 1
        f['roll_freq_0'] = last_10.count(0)/L
        f['roll_freq_1'] = last_10.count(1)/L
        f['roll_freq_2'] = last_10.count(2)/L

        f['j2_tras_ganar'] = j2s[-2] if res[-2] == -1 else -1
        f['j2_tras_perder'] = j2s[-2] if res[-2] == 1 else -1
        f['cwp_signal'] = JUGADA_A_NUM[PIERDE_CONTRA[NUM_A_JUGADA[j2s[-1]]]]

        return np.array([f.get(c, -1) for c in self.feature_cols]).reshape(1, -1)

    def decidir_jugada(self):
        # ---------------------------------------------------------
        # NIVEL 0: ANTI-SPAM AGRESIVO (Reflejo)
        # ---------------------------------------------------------
        # Si las últimas 2 jugadas son iguales, asumimos que repetirá.
        if len(self.historial) >= 2:
            h = self.historial
            if h[-1][1] == h[-2][1]: # J2 repitió jugada
                pred = JUGADA_A_NUM[h[-1][1]]
                # print(f"DEBUG: Spam detectado ({NUM_A_JUGADA[pred]})")
                return PIERDE_CONTRA[NUM_A_JUGADA[pred]]

        # ---------------------------------------------------------
        # NIVEL 1: MEMORIA VIVA (Lo que está pasando AHORA)
        # ---------------------------------------------------------
        live_pred = self._get_live_prediction()
        if live_pred is not None:
            # print(f"DEBUG: Patrón vivo detectado ({NUM_A_JUGADA[live_pred]})")
            return PIERDE_CONTRA[NUM_A_JUGADA[live_pred]]

        # ---------------------------------------------------------
        # NIVEL 2: MODELO ML (Intuición General)
        # ---------------------------------------------------------
        if self.modelo and len(self.historial) >= 3:
            try:
                feats = self._get_features()
                probs = self.modelo.predict_proba(feats)[0]
                pred_idx = np.argmax(probs)

                # Diversificación ligera (si top 2 están cerca, azar)
                sorted_idx = np.argsort(probs)[::-1]
                if probs[sorted_idx[1]] > 0.8 * probs[sorted_idx[0]]:
                    pred_idx = np.random.choice(sorted_idx[:2])

                return PIERDE_CONTRA[NUM_A_JUGADA[pred_idx]]
            except:
                pass

        # Fallback: Aleatorio
        return np.random.choice(["piedra", "papel", "tijera"])


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*60)
    print("   RPSAI v4.0 - Entrenamiento Híbrido (Live Learning)")
    print("="*60)

    try:
        # 1. Cargar y Entrenar
        df = cargar_datos()
        df = preparar_datos(df)
        df = crear_features(df)
        X, y = seleccionar_features(df)

        # Verificar clases
        if len(y.unique()) < 3:
            print("⚠️ Advertencia: El dataset no tiene ejemplos de las 3 jugadas.")

        modelo = entrenar_modelo(X, y)

        # 2. Guardar
        RUTA_MODELO.parent.mkdir(parents=True, exist_ok=True)
        with open(RUTA_MODELO, "wb") as f:
            pickle.dump(modelo, f)

        print("\n✅ ¡IA Actualizada a v4.0!")
        print("   Estrategia: Anti-Spam (2 rep) -> Patrón Vivo -> Modelo ML")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()