"""
RPSAI - Modelo de IA para Piedra, Papel o Tijera
=================================================
Implementación completa del modelo de predicción
"""

import os
import pickle
import warnings
from pathlib import Path

import pandas as pd
import numpy as np

warnings.filterwarnings("ignore", message="X does not have valid feature names")

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Configuración de rutas
RUTA_PROYECTO = Path(__file__).parent.parent
RUTA_DATOS = RUTA_PROYECTO / "data" / "datos_ppt_ai.csv"
RUTA_MODELO = RUTA_PROYECTO / "models" / "modelo_entrenado.pkl"

# Mapeo de jugadas a números
JUGADA_A_NUM = {"piedra": 0, "papel": 1, "tijera": 2}
NUM_A_JUGADA = {0: "piedra", 1: "papel", 2: "tijera"}

# Qué jugada gana a cuál
GANA_A = {"piedra": "tijera", "papel": "piedra", "tijera": "papel"}
PIERDE_CONTRA = {"piedra": "papel", "papel": "tijera", "tijera": "piedra"}


# =============================================================================
# PARTE 1: EXTRACCIÓN DE DATOS
# =============================================================================

def cargar_datos(ruta_csv: str = None) -> pd.DataFrame:
    """Carga los datos del CSV de partidas."""
    if ruta_csv is None:
        ruta_csv = RUTA_DATOS

    if not os.path.exists(ruta_csv):
        raise FileNotFoundError(f"No se encontró el archivo: {ruta_csv}")

    df = pd.read_csv(ruta_csv)

    # Verificar columnas necesarias (adaptado a tu CSV)
    columnas_necesarias = ["jugada_jugador1", "jugada_jugador2"]
    columnas_faltantes = [col for col in columnas_necesarias if col not in df.columns]

    if columnas_faltantes:
        raise ValueError(f"Faltan columnas necesarias: {columnas_faltantes}")

    print(f"✅ Datos cargados: {len(df)} partidas")
    return df


def preparar_datos(df: pd.DataFrame) -> pd.DataFrame:
    """Prepara los datos para el modelo."""
    df = df.copy()

    # Renombrar columnas para mantener consistencia
    df = df.rename(columns={
        'jugada_jugador1': 'jugada_j1',
        'jugada_jugador2': 'jugada_j2'
    })

    # Convertir jugadas a números
    df['jugada_j1_num'] = df['jugada_j1'].map(JUGADA_A_NUM)
    df['jugada_j2_num'] = df['jugada_j2'].map(JUGADA_A_NUM)

    # Crear columna de próxima jugada del oponente (target)
    df['proxima_jugada_j2'] = df['jugada_j2_num'].shift(-1)

    # Crear número de ronda si no existe
    if 'numero_ronda' not in df.columns:
        df['numero_ronda'] = range(1, len(df) + 1)

    # Calcular resultado de cada ronda
    def calcular_resultado(row):
        j1, j2 = row['jugada_j1'], row['jugada_j2']
        if j1 == j2:
            return 0  # Empate
        elif GANA_A[j1] == j2:
            return 1  # Gana j1
        else:
            return -1  # Gana j2

    df['resultado'] = df.apply(calcular_resultado, axis=1)

    # Eliminar la última fila (no tiene próxima jugada)
    df = df.dropna(subset=['proxima_jugada_j2'])

    print(f"✅ Datos preparados: {len(df)} filas para entrenar")
    return df


# =============================================================================
# PARTE 2: FEATURE ENGINEERING
# =============================================================================

def crear_features(df: pd.DataFrame) -> pd.DataFrame:
    """Crea features avanzadas para el modelo."""
    df = df.copy()

    # ------------------------------------------
    # Feature 1: Frecuencia de cada jugada de j2
    # ------------------------------------------
    df['freq_j2_piedra'] = (df['jugada_j2_num'] == 0).expanding().mean()
    df['freq_j2_papel'] = (df['jugada_j2_num'] == 1).expanding().mean()
    df['freq_j2_tijera'] = (df['jugada_j2_num'] == 2).expanding().mean()

    # ------------------------------------------
    # Feature 2: Últimas jugadas (lag features)
    # ------------------------------------------
    df['jugada_j2_lag1'] = df['jugada_j2_num'].shift(1)
    df['jugada_j2_lag2'] = df['jugada_j2_num'].shift(2)
    df['jugada_j2_lag3'] = df['jugada_j2_num'].shift(3)

    df['jugada_j1_lag1'] = df['jugada_j1_num'].shift(1)
    df['jugada_j1_lag2'] = df['jugada_j1_num'].shift(2)

    # ------------------------------------------
    # Feature 3: Resultado anterior
    # ------------------------------------------
    df['resultado_anterior'] = df['resultado'].shift(1)

    # ------------------------------------------
    # Feature 4: Racha actual
    # ------------------------------------------
    # Cuenta victorias/derrotas consecutivas de j2
    df['racha_j2'] = 0
    racha = 0
    for i in range(1, len(df)):
        if df.iloc[i-1]['resultado'] == -1:  # j2 ganó
            racha = racha + 1 if racha >= 0 else 1
        elif df.iloc[i-1]['resultado'] == 1:  # j2 perdió
            racha = racha - 1 if racha <= 0 else -1
        else:  # empate
            racha = 0
        df.iloc[i, df.columns.get_loc('racha_j2')] = racha

    # ------------------------------------------
    # Feature 5: Patrón después de ganar/perder
    # ------------------------------------------
    # ¿j2 repite jugada después de ganar?
    df['repite_tras_ganar'] = ((df['resultado'].shift(1) == -1) &
                                (df['jugada_j2_num'] == df['jugada_j2_num'].shift(1))).astype(int)

    # ¿j2 cambia jugada después de perder?
    df['cambia_tras_perder'] = ((df['resultado'].shift(1) == 1) &
                                 (df['jugada_j2_num'] != df['jugada_j2_num'].shift(1))).astype(int)

    # ------------------------------------------
    # Feature 6: Fase del juego
    # ------------------------------------------
    total_rondas = len(df)
    df['fase_juego'] = df['numero_ronda'] / total_rondas  # 0 = inicio, 1 = final

    # ------------------------------------------
    # Feature 7: Diversidad de jugadas recientes
    # ------------------------------------------
    # Cuenta cuántas jugadas diferentes hizo en las últimas 5 rondas
    def calcular_diversidad(serie, ventana=5):
        diversidad = []
        for i in range(len(serie)):
            if i < ventana:
                diversidad.append(len(set(serie[:i+1])))
            else:
                diversidad.append(len(set(serie[i-ventana+1:i+1])))
        return diversidad

    df['diversidad_j2'] = calcular_diversidad(df['jugada_j2_num'].tolist())

    print(f"✅ Features creadas: {len([col for col in df.columns if 'freq_' in col or 'lag' in col or 'racha' in col])} features")

    return df


def seleccionar_features(df: pd.DataFrame) -> tuple:
    """Selecciona las features para entrenar y el target."""

    # Definir columnas de features
    feature_cols = [
        'jugada_j2_lag1', 'jugada_j2_lag2', 'jugada_j2_lag3',
        'jugada_j1_lag1', 'jugada_j1_lag2',
        'freq_j2_piedra', 'freq_j2_papel', 'freq_j2_tijera',
        'resultado_anterior', 'racha_j2',
        'repite_tras_ganar', 'cambia_tras_perder',
        'fase_juego', 'diversidad_j2'
    ]

    # Eliminar filas con valores nulos en las features
    df_clean = df.dropna(subset=feature_cols + ['proxima_jugada_j2'])

    X = df_clean[feature_cols]
    y = df_clean['proxima_jugada_j2'].astype(int)

    print(f"✅ Features seleccionadas: {len(feature_cols)}")
    print(f"✅ Datos limpios: {len(X)} muestras")

    return X, y


# =============================================================================
# PARTE 3: ENTRENAMIENTO Y FUNCIONAMIENTO
# =============================================================================

def entrenar_modelo(X, y, test_size: float = 0.2):
    """Entrena y evalúa múltiples modelos."""

    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    print(f"\n📊 Train: {len(X_train)} | Test: {len(X_test)}")
    print("="*60)

    # Definir modelos a entrenar
    modelos = {
        'KNN (k=5)': KNeighborsClassifier(n_neighbors=5),
        'KNN (k=7)': KNeighborsClassifier(n_neighbors=7),
        'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    }

    resultados = {}

    # Entrenar y evaluar cada modelo
    for nombre, modelo in modelos.items():
        print(f"\n🤖 Entrenando {nombre}...")

        # Entrenar
        modelo.fit(X_train, y_train)

        # Predecir
        y_pred_train = modelo.predict(X_train)
        y_pred_test = modelo.predict(X_test)

        # Calcular accuracy
        acc_train = accuracy_score(y_train, y_pred_train)
        acc_test = accuracy_score(y_test, y_pred_test)

        resultados[nombre] = {
            'modelo': modelo,
            'acc_train': acc_train,
            'acc_test': acc_test
        }

        print(f"   Train Accuracy: {acc_train:.2%}")
        print(f"   Test Accuracy:  {acc_test:.2%}")

        # Mostrar reporte detallado
        print(f"\n   Classification Report:")
        print(classification_report(y_test, y_pred_test,
                                     target_names=['piedra', 'papel', 'tijera'],
                                     zero_division=0))

    # Seleccionar el mejor modelo (mayor accuracy en test)
    mejor_nombre = max(resultados, key=lambda k: resultados[k]['acc_test'])
    mejor_modelo = resultados[mejor_nombre]['modelo']

    print("="*60)
    print(f"🏆 MEJOR MODELO: {mejor_nombre}")
    print(f"   Accuracy: {resultados[mejor_nombre]['acc_test']:.2%}")
    print("="*60)

    return mejor_modelo


def guardar_modelo(modelo, ruta: str = None):
    """Guarda el modelo entrenado."""
    if ruta is None:
        ruta = RUTA_MODELO

    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    with open(ruta, "wb") as f:
        pickle.dump(modelo, f)
    print(f"💾 Modelo guardado en: {ruta}")


def cargar_modelo(ruta: str = None):
    """Carga un modelo previamente entrenado."""
    if ruta is None:
        ruta = RUTA_MODELO

    if not os.path.exists(ruta):
        raise FileNotFoundError(f"No se encontró el modelo en: {ruta}")

    with open(ruta, "rb") as f:
        return pickle.load(f)


# =============================================================================
# PARTE 4: PREDICCIÓN Y JUEGO
# =============================================================================

class JugadorIA:
    """Jugador de IA que predice y juega contra oponentes."""

    def __init__(self, ruta_modelo: str = None):
        """Inicializa el jugador IA."""
        self.modelo = None
        self.historial = []

        try:
            self.modelo = cargar_modelo(ruta_modelo)
            print("✅ Modelo cargado correctamente")
        except FileNotFoundError:
            print("⚠️ Modelo no encontrado. Entrena primero o jugaré aleatorio.")

    def registrar_ronda(self, jugada_j1: str, jugada_j2: str):
        """Registra una ronda jugada."""
        self.historial.append((jugada_j1, jugada_j2))

    def obtener_features_actuales(self) -> np.ndarray:
        """Genera features basadas en el historial actual."""
        if len(self.historial) < 3:
            # No hay suficiente historial, retornar valores por defecto
            return np.zeros(14)

        # Convertir historial a números
        j1_nums = [JUGADA_A_NUM[j1] for j1, j2 in self.historial]
        j2_nums = [JUGADA_A_NUM[j2] for j1, j2 in self.historial]

        # Calcular resultados
        resultados = []
        for j1, j2 in self.historial:
            if j1 == j2:
                resultados.append(0)
            elif GANA_A[j1] == j2:
                resultados.append(1)
            else:
                resultados.append(-1)

        # Crear features (deben coincidir con el orden de entrenamiento)
        features = []

        # Lags de j2
        features.append(j2_nums[-1] if len(j2_nums) >= 1 else 0)
        features.append(j2_nums[-2] if len(j2_nums) >= 2 else 0)
        features.append(j2_nums[-3] if len(j2_nums) >= 3 else 0)

        # Lags de j1
        features.append(j1_nums[-1] if len(j1_nums) >= 1 else 0)
        features.append(j1_nums[-2] if len(j1_nums) >= 2 else 0)

        # Frecuencias
        features.append(j2_nums.count(0) / len(j2_nums))  # piedra
        features.append(j2_nums.count(1) / len(j2_nums))  # papel
        features.append(j2_nums.count(2) / len(j2_nums))  # tijera

        # Resultado anterior
        features.append(resultados[-1] if resultados else 0)

        # Racha
        racha = 0
        for r in reversed(resultados):
            if r == -1:
                racha = racha + 1 if racha >= 0 else 1
            elif r == 1:
                racha = racha - 1 if racha <= 0 else -1
            else:
                break
        features.append(racha)

        # Patrones
        features.append(0)  # repite_tras_ganar (simplificado)
        features.append(0)  # cambia_tras_perder (simplificado)

        # Fase del juego (simplificado a 0.5)
        features.append(0.5)

        # Diversidad
        ultimas_5 = j2_nums[-5:] if len(j2_nums) >= 5 else j2_nums
        features.append(len(set(ultimas_5)))

        return np.array(features).reshape(1, -1)

    def predecir_jugada_oponente(self) -> str:
        """Predice la próxima jugada del oponente."""
        if self.modelo is None:
            return np.random.choice(["piedra", "papel", "tijera"])

        if len(self.historial) < 3:
            # No hay suficiente historial, jugar aleatorio
            return np.random.choice(["piedra", "papel", "tijera"])

        features = self.obtener_features_actuales()
        prediccion = self.modelo.predict(features)[0]
        return NUM_A_JUGADA[int(prediccion)]

    def decidir_jugada(self) -> str:
        """Decide qué jugada hacer para ganar."""
        prediccion_oponente = self.predecir_jugada_oponente()
        # Jugar lo que le gana a la predicción
        return PIERDE_CONTRA[prediccion_oponente]


# =============================================================================
# FUNCIÓN PRINCIPAL
# =============================================================================

def main():
    """Función principal para entrenar el modelo."""
    print("="*60)
    print("   RPSAI - Entrenamiento del Modelo")
    print("="*60)

    try:
        # 1. Cargar datos
        print("\n📁 Cargando datos...")
        df = cargar_datos()

        # 2. Preparar datos
        print("\n🔧 Preparando datos...")
        df = preparar_datos(df)

        # 3. Crear features
        print("\n🧪 Creando features...")
        df = crear_features(df)

        # 4. Seleccionar features
        print("\n🎯 Seleccionando features...")
        X, y = seleccionar_features(df)

        # 5. Entrenar modelo
        print("\n🚀 Entrenando modelos...")
        mejor_modelo = entrenar_modelo(X, y)

        # 6. Guardar modelo
        print("\n💾 Guardando modelo...")
        guardar_modelo(mejor_modelo)

        print("\n✅ ¡ENTRENAMIENTO COMPLETADO!")
        print("\nAhora puedes usar el modelo ejecutando el juego.")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()