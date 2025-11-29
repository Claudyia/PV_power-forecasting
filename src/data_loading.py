##legge i file excel, gestisce i datatime index,sistema il time alignment, gestione missing values
# src/data_loading.py

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd


# Cartelle di riferimento (partendo da questo file dentro src/)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "raw_data"


def load_pv_data(
    path: Path | None = None,
    rename_power_col: str = "pv_power",
    rename_time_col: str = "timestamp",
) -> pd.DataFrame:
    """
    Carica il dataset fotovoltaico da Excel e lo riporta in un formato pulito:

    - timestamp in una colonna 'timestamp'
    - potenza PV in una colonna 'pv_power'
    - index = timestamp ordinato

    Gestisce la struttura specifica di pv_dataset.xlsx:
    colonne = ['Max kWp', 82.41]
    dove 'Max kWp' è il timestamp e 82.41 è il valore di potenza (kWp installati).
    """

    if path is None:
        path = RAW_DATA_DIR / "pv_dataset.xlsx"

    df = pd.read_excel(path)

    cols = list(df.columns)

    # Caso specifico del file che hai: ['Max kWp', 82.41]
    if len(cols) == 2 and isinstance(cols[1], (int, float)):
        time_col = cols[0]
        power_col = cols[1]

        # memorizzo la potenza di picco dell'impianto (non indispensabile ma può tornare utile)
        max_kwp = float(power_col)

        df = df.rename(
            columns={
                time_col: rename_time_col,
                power_col: rename_power_col,
            }
        )
        df["max_kwp"] = max_kwp
    else:
        # fallback generico: si aspetta già colonne chiamate timestamp / pv_power
        if rename_time_col not in df.columns:
            raise ValueError(
                f"Colonna temporale non riconosciuta nel PV dataset. Colonne trovate: {df.columns}"
            )
        if rename_power_col not in df.columns:
            raise ValueError(
                f"Colonna di potenza non riconosciuta nel PV dataset. Colonne trovate: {df.columns}"
            )

    # parsing timestamp & ordinamento
    df[rename_time_col] = pd.to_datetime(df[rename_time_col])
    df = df.sort_values(rename_time_col).set_index(rename_time_col)

    return df


def load_weather_data(
    path: Path | None = None,
    rename_time_col: str = "timestamp",
) -> pd.DataFrame:
    """
    Carica il dataset meteorologico da Excel e lo pulisce minimamente.

    - dt_iso → timestamp (stringa con timezone +10:00)
    - taglia la parte timezone e crea datetime naive
    - index = timestamp ordinato
    """

    if path is None:
        path = RAW_DATA_DIR / "wx_dataset.xlsx"

    df = pd.read_excel(path)

    if "dt_iso" not in df.columns:
        raise ValueError(
            f"Colonna 'dt_iso' non trovata nel weather dataset. Colonne: {df.columns}"
        )

    # dt_iso è tipo "2010-07-01 00:00:00+10:00"
    # Prendiamo solo la parte "YYYY-MM-DD HH:MM:SS" per allinearci al PV
    ts = pd.to_datetime(df["dt_iso"].astype(str).str.slice(0, 19))

    df = df.drop(columns=["dt_iso"])
    df[rename_time_col] = ts

    df = df.sort_values(rename_time_col).set_index(rename_time_col)

    return df


def merge_pv_weather(
    pv_df: pd.DataFrame,
    wx_df: pd.DataFrame,
    how: str = "inner",
    drop_constant_geo: bool = True,
) -> pd.DataFrame:
    """
    Unisce i dati PV e meteo su base temporale.

    - join sul timestamp con 'how' (default: inner → solo timestamps comuni)
    - opzionalmente rimuove colonne geografiche costanti (lat/lon).
    """

    df = pv_df.join(wx_df, how=how, rsuffix="_wx")

    # in wx_dataset ci sono lat/lon costanti → li possiamo eventualmente droppare
    if drop_constant_geo:
        for col in ["lat", "lon"]:
            if col in df.columns:
                if df[col].nunique(dropna=True) <= 1:
                    df = df.drop(columns=[col])

    return df


def load_all() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Convenience function:
    - carica pv
    - carica meteo
    - restituisce (pv, meteo, merged)
    """
    pv_df = load_pv_data()
    wx_df = load_weather_data()
    merged = merge_pv_weather(pv_df, wx_df)
    return pv_df, wx_df, merged


if __name__ == "__main__":
    # piccolo test manuale
    pv_df, wx_df, merged_df = load_all()
    print("PV shape:", pv_df.shape)
    print("WX shape:", wx_df.shape)
    print("Merged shape:", merged_df.shape)
    print("Merged columns:", merged_df.columns.tolist())