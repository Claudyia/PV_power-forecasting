# src/preprocessing.py

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "processed_data"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def load_merged_raw(path: Path | None = None) -> pd.DataFrame:
    """
    Carica il dataset merged_raw creato in Fase 1.
    Prova prima il parquet, poi il csv.
    """
    if path is not None:
        return _load_any(path)

    parquet_path = PROCESSED_DIR / "merged_raw.parquet"
    csv_path = PROCESSED_DIR / "merged_raw.csv"

    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    elif csv_path.exists():
        df = pd.read_csv(csv_path, parse_dates=["timestamp"])
        return df.set_index("timestamp").sort_index()
    else:
        raise FileNotFoundError(
            "Non trovo né merged_raw.parquet né merged_raw.csv in processed_data/. "
            "Assicurati di aver eseguito la Fase 1."
        )


def _load_any(path: Path) -> pd.DataFrame:
    """
    Funzione di supporto: carica parquet o csv a seconda dell'estensione.
    """
    ext = path.suffix.lower()
    if ext == ".parquet":
        return pd.read_parquet(path)
    elif ext in {".csv", ".txt"}:
        df = pd.read_csv(path, parse_dates=["timestamp"])
        return df.set_index("timestamp").sort_index()
    else:
        raise ValueError(f"Estensione file non supportata: {ext}")


# ---------------------------
# 1) Feature temporali
# ---------------------------

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggiunge colonne basate sul timestamp:
    - hour: ora del giorno (0-23)
    - dayofweek: giorno settimana (0=Monday)
    - month: mese (1-12)
    - dayofyear: giorno dell'anno (1-366)
    - is_weekend: 0/1
    """
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("L'indice del DataFrame deve essere un DatetimeIndex per usare add_time_features.")

    df["hour"] = df.index.hour
    df["dayofweek"] = df.index.dayofweek
    df["month"] = df.index.month
    df["dayofyear"] = df.index.dayofyear
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

    return df


# ---------------------------
# 2) Gestione dei NaN
# ---------------------------

def handle_missing_values(
    df: pd.DataFrame,
    target_col: str = "pv_power",
    max_na_ratio: float = 0.3,
) -> pd.DataFrame:
    """
    Pulisce i NaN in modo robusto:

    - rimuove colonne con una quota di NaN > max_na_ratio
    - per le colonne numeriche:
        - interpola nel tempo (interpolate(method='time'))
        - fa forward-fill + back-fill per eventuali NaN residui
    - per la colonna target:
        - drop delle righe dove pv_power è NaN (sono inutilizzabili come target)

    Restituisce un nuovo DataFrame pulito.
    """
    df = df.copy()

    # 1) Droppiamo colonne troppo vuote
    na_ratio = df.isna().mean()
    cols_to_drop = na_ratio[na_ratio > max_na_ratio].index.tolist()

    if cols_to_drop:
        print("Drop colonne con troppi NaN (> {:.0%}):".format(max_na_ratio), cols_to_drop)
        df = df.drop(columns=cols_to_drop)

    # 2) Colonne numeriche: interpolazione + ffill + bfill
    num_cols: List[str] = df.select_dtypes(include=[np.number]).columns.tolist()

    df[num_cols] = df[num_cols].interpolate(method="time", limit_direction="both")

    df[num_cols] = df[num_cols].ffill().bfill()

    # 3) Colonna target: drop righe ancora senza target
    if target_col in df.columns:
        before = len(df)
        df = df.dropna(subset=[target_col])
        after = len(df)
        if after < before:
            print(f"Drop di {before - after} righe senza target '{target_col}'.")
    else:
        print(f"ATTENZIONE: colonna target '{target_col}' non trovata nel DataFrame.")

    return df


# ---------------------------
# 3) Pipeline di preprocessing "base"
# ---------------------------

def preprocess_merged_raw(
    target_col: str = "pv_power",
    max_na_ratio: float = 0.3,
    save: bool = True,
) -> pd.DataFrame:
    """
    Esegue la pipeline base di preprocessing:

    1. Carica merged_raw
    2. Aggiunge feature temporali
    3. Gestisce NaN
    4. (Opzionale) salva merged_clean in processed_data/

    Restituisce il DataFrame pulito.
    """
    df = load_merged_raw()

    print("Loaded merged_raw with shape:", df.shape)

    # Assicuriamoci che il timestamp sia index
    if "timestamp" in df.columns and not isinstance(df.index, pd.DatetimeIndex):
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp").sort_index()

    # 1) Feature temporali
    df = add_time_features(df)

    # 2) Gestione NaN
    df = handle_missing_values(df, target_col=target_col, max_na_ratio=max_na_ratio)

    print("Shape dopo cleaning:", df.shape)

    if save:
        clean_parquet = PROCESSED_DIR / "merged_clean.parquet"
        clean_csv = PROCESSED_DIR / "merged_clean.csv"

        df.to_parquet(clean_parquet)
        df.to_csv(clean_csv)

        print("Salvato merged_clean.parquet e merged_clean.csv in", PROCESSED_DIR)

    return df


if __name__ == "__main__":
    # piccolo test manuale
    df_clean = preprocess_merged_raw()
    print(df_clean.head())