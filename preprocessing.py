from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "raw_data"
OUTPUT_DIR = BASE_DIR / "processed"
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# 1. LOAD PV (floor a ora intera, timeline affidabile)
# ============================================================

def load_pv():
    pv_excel = DATA_DIR / "pv_dataset.xlsx"

    pv1 = pd.read_excel(pv_excel, sheet_name=0)
    pv2 = pd.read_excel(pv_excel, sheet_name=1)
    pv = pd.concat([pv1, pv2], ignore_index=True)

    timestamp_col = "Max kWp"
    value_col = [c for c in pv.columns if c != timestamp_col][0]

    pv = pv.rename(columns={
        timestamp_col: "pv_date",
        value_col: "Production_KWh"
    })

    pv["pv_date"] = pd.to_datetime(pv["pv_date"], format="mixed", errors="coerce")
    pv = pv.sort_values("pv_date").reset_index(drop=True)

    return pv[["pv_date", "Production_KWh"]]

# ============================================================
# 2. LOAD WX (NON toccare i timestamp, li sistemiamo dopo)
# ============================================================

from dateutil import parser

def load_wx():
    wx_excel = DATA_DIR / "wx_dataset.xlsx"

    wx1 = pd.read_excel(wx_excel, sheet_name=0)
    wx2 = pd.read_excel(wx_excel, sheet_name=1)
    wx = pd.concat([wx1, wx2], ignore_index=True)

    # Uniformiamo da subito i nomi che useremo più avanti
    wx = wx.rename(columns={
        "dt_iso": "wx_date",
        "temp": "temperature"
    })

    # Parsing robusto riga per riga
    def safe_parse(x):
        try:
            dt = parser.parse(str(x))   # legge anche +10, +11, ecc.
            if dt.tzinfo is not None:   # se ha timezone, lo togliamo
                dt = dt.replace(tzinfo=None)
            return dt
        except Exception:
            return pd.NaT

    wx["wx_date"] = wx["wx_date"].apply(safe_parse)

    # Forza il tipo a datetime64[ns] 
    wx["wx_date"] = pd.to_datetime(wx["wx_date"], errors="coerce")

    # Pulisce e ordina
    wx = wx.dropna(subset=["wx_date"])
    wx = wx.sort_values("wx_date").reset_index(drop=True)

    return wx

# ============================================================
# 3. FIX TIMELINE WX – RICOSTRUZIONE ORARIA USANDO PV
# ============================================================

def fix_daily_wx_timeline(wx: pd.DataFrame, start_date) -> pd.DataFrame:
    """
    FORZA BRUTA:
    - ignora completamente i valori originali di wx_date
    - costruisce una sequenza oraria continua a partire da start_date
    - assegna questa sequenza a wx["wx_date"]

    Risultato:
      start_date
      start_date + 1h
      start_date + 2h
      ...
      sempre +1h, rollover automatico giorni (00..23, poi 00..)
    """
    wx = wx.sort_values("wx_date").reset_index(drop=True)

    # Punto di partenza: lo forziamo a essere un Timestamp arrotondato all'ora
    start = pd.to_datetime(start_date, errors="coerce")
    start = start.replace(minute=0, second=0, microsecond=0)

    n = len(wx)

    # sequenza oraria continua
    new_dates = pd.date_range(start=start, periods=n, freq="h")

    wx = wx.copy()
    wx["wx_date"] = new_dates

    return wx



# ============================================================
# 4. WEATHER CATEGORY
# ============================================================

def categorize_weather(wx):
    def map_weather(desc):
        d = str(desc).lower()

        if "broken" in d or "overcast" in d:
            return "cloudy"
        if "scattered" in d or "few" in d:
            return "partly_cloudy"
        if "light rain" in d or "moderate rain" in d:
            return "rain"
        if "clear sky" in d or d.strip() == "clear":
            return "clear"
        return "other"

    wx["weather_category"] = wx["weather_description"].apply(map_weather)
    return wx


# ============================================================
# 5. ONE-HOT ENCODING
# ============================================================

def one_hot_encode_weather(wx):
    dummies = pd.get_dummies(wx["weather_category"], prefix="weather")

    expected = [
        "weather_cloudy",
        "weather_partly_cloudy",
        "weather_rain",
        "weather_clear",
        "weather_other"
    ]
    for col in expected:
        if col not in dummies:
            dummies[col] = 0

    return pd.concat([wx, dummies[expected]], axis=1)


# ============================================================
# 6. DROP UNUSED WX COLUMNS
# ============================================================

def drop_unused_columns(wx):
    cols_to_drop = [
        "lat", "lon", "dew_point", "pressure",
        "weather_description"
    ]
    return wx.drop(columns=[c for c in cols_to_drop if c in wx.columns])


# ============================================================
# 7. TIME FEATURES (basate su PV)
# ============================================================

def add_time_features(df, date_col):
    dt = pd.to_datetime(df[date_col], errors="coerce")

    df["month"]       = dt.dt.month
    df["day"]         = dt.dt.day
    df["hour"]        = dt.dt.hour
    df["day_of_week"] = dt.dt.dayofweek
    df["is_weekend"]  = df["day_of_week"].isin([5, 6]).astype(int)
    df["day_of_year"] = dt.dt.dayofyear
    df["hour_sin"]   = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"]   = np.cos(2 * np.pi * df["hour"] / 24)
    df["doy_sin"]   = np.sin(2 * np.pi * df["day_of_year"] / 366)
    df["doy_cos"]   = np.cos(2 * np.pi * df["day_of_year"] / 366)   

    return df


def add_lag_features(df, columns, lags): 
    """" funzione per aggiungere feature di lag  per la potenza per 1h, 12 h, 24h ecc """
    df = df.copy()
    for col in columns:
        if col not in df.columns:
            continue
        for lag in lags:
            df[f"{col}_lag_{lag}h"] = df[col].shift(lag)
    return df 


def add_rolling_features(df, columns, windows):
    """	roll_mean_3h
	•	roll_max_3h
	•	roll_min_3h
	•	roll_std_6h 
	•	roll_mean_24h """
    df = df.copy()
    for col in columns:
        if col not in df.columns:
            continue
        for window in windows:
            df[f"{col}_roll_mean_{window}h"] = df[col].rolling(window=window).mean()
            df[f"{col}_roll_max_{window}h"]  = df[col].rolling(window=window).max()
            df[f"{col}_roll_min_{window}h"]  = df[col].rolling(window=window).min()
            df[f"{col}_roll_std_{window}h"]  = df[col].rolling(window=window).std()
    return df


def add_weather_combinations(df):
    """ ✔ Combinazioni fisiche utili
	•	Effettiva radiazione utile:
        GHI * (1 - CloudCover)
            •	Rapporto fisico:
        GHI / Temperature
            •	Aria secca/umida:
        RH * Temperature
            •	Wind chill sulle celle:
        WindSpeed * Temperature"""
    df = df.copy()
    ghi_col = None
    for candidate in ["GHI", "Ghi", "ghi"]:
        if candidate in df.columns:
            ghi_col = candidate
            break

    if ghi_col and "weather_cloudy" in df.columns:
        df["effective_radiation"] = df[ghi_col] * (1 - df["weather_cloudy"])
    if ghi_col and "temperature" in df.columns:
        df["ghi_temp_ratio"] = df[ghi_col] / df["temperature"].replace(0, pd.NA)
    if "humidity" in df.columns and "temperature" in df.columns:
        df["rh_temp_product"] = df["humidity"] * df["temperature"]
    if "wind_speed" in df.columns and "temperature" in df.columns:
        df["wind_chill_effect"] = df["wind_speed"] * df["temperature"]
    return df


def format_pv_date_for_display(df, date_col, fmt="%d/%m/%y %H:%M"):
    """
    Render pv_date exactly as requested (dd/mm/yy HH:MM) for the exported file.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce").dt.strftime(fmt)
    return df


# ============================================================
# 8. MERGE PER INDICE (side-by-side)
# ============================================================

#crea un nuovo dataset prendendo le colonne che ci servono da pv e wx e le unisce in un unico dataframe
def nuovo_dataset(pv, wx):
    df = pd.DataFrame()

    df["pv_date"] = pv["pv_date"]
    df["Production_KWh"] = pv["Production_KWh"]

    df["temperature"] = wx["temperature"]
    df["humidity"] = wx["humidity"]

    for extra_col in ["Ghi", "Dhi", "Dni", "wind_speed", "clouds_all"]:
        if extra_col in wx.columns:
            df[extra_col] = wx[extra_col]

    # One-hot
    df["weather_cloudy"]         = wx["weather_cloudy"]
    df["weather_partly_cloudy"]  = wx["weather_partly_cloudy"]
    df["weather_rain"]           = wx["weather_rain"]
    df["weather_clear"]          = wx["weather_clear"]
    df["weather_other"]          = wx["weather_other"]

    # Time features da PV
    df["hour_sin"]   = pv["hour_sin"]
    df["hour_cos"]   = pv["hour_cos"]
    df["doy_sin"]   = pv["doy_sin"]
    df["doy_cos"]   = pv["doy_cos"]

    return df











# ============================================================
## ============================================================
# 9. BUILD FINAL DATASET
# ============================================================
def build_merged_dataset():

    pv = load_pv()
    wx = load_wx()

    # WX FE
    wx = categorize_weather(wx)
    wx = one_hot_encode_weather(wx)
    wx = drop_unused_columns(wx)

    # Fissa timeline WX
    wx = fix_daily_wx_timeline(wx, start_date=pv["pv_date"].iloc[0])

    # ⭐ PRIMA aggiungi time features a PV
    pv = add_time_features(pv, date_col="pv_date")

    # ⭐ Poi crea il dataset
    merged = nuovo_dataset(pv, wx)

    lag_columns = ["Production_KWh"]
    lag_hours = [1, 12, 24]
    rolling_columns = ["Production_KWh"]
    rolling_windows = [3, 6, 24]

    merged = add_lag_features(merged, columns=lag_columns, lags=lag_hours)
    merged = add_rolling_features(merged, columns=rolling_columns, windows=rolling_windows)
    merged = add_weather_combinations(merged)

    history = max(max(lag_hours), max(rolling_windows))
    merged = merged.iloc[history:].reset_index(drop=True)

    # Formatta pv_date per Excel
    merged = format_pv_date_for_display(merged, date_col="pv_date")

    out = OUTPUT_DIR / "merged_dataset_final.xlsx"
    merged.to_excel(out, index=False)

    print("✓ merged_dataset_final.xlsx salvato.")
    return merged
# ============================================================

if __name__ == "__main__":
    build_merged_dataset()
