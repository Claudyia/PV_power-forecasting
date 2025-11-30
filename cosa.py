# prende il database merged_clean_unified e gestisce gli outliers con interpolazione

from pathlib import Path
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
PROCESSED_DIR = PROJECT_ROOT / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

MERGED_CLEAN_UNIFIED = PROCESSED_DIR / "merged_dataset_final.xlsx"
OUTPUT_INTERPOLATED = PROCESSED_DIR / "merged_dataset_final_interpolated.xlsx"

def find_outliers(df, column="Production_KWh", ghi_column="Ghi",
                  q_factor=1.20, ghi_ratio=0.25):
    """
    Trova gli outlier senza modificare il dataframe originale.
    Ritorna:
        - df_outliers: le righe anomale
        - mask: maschera booleana
    """

    prod = df[column]
    has_ghi = ghi_column in df.columns
    ghi = df[ghi_column] if has_ghi else None

    # Soglia statistica (1.20 * Q75)
    q75 = prod.quantile(0.75)
    threshold_q = q75 * q_factor

    # Soglia fisica basata su GHI
    mask = prod > threshold_q
    if has_ghi:
        threshold_ghi = ghi * ghi_ratio
        mask = mask | (prod > threshold_ghi)

    df_outliers = df[mask]
    cols_to_show = [column]
    if has_ghi:
        cols_to_show.append(ghi_column)

    print("\n=== OUTLIERS TROVATI ===")
    print(f"Totale: {df_outliers.shape[0]}")
    print(df_outliers[cols_to_show].head())

    return df_outliers, mask



def interpolate_point_by_point(df, mask, column="Production_KWh"):
    """
    Interpola ogni outlier SINGOLARMENTE.
    Nessun rischio di toccare i valori vicini.
    """

    df_fixed = df.copy()

    indices = np.where(mask)[0]

    for i in indices:
        prev_val = df_fixed[column].iloc[i-1] if i > 0 else None
        next_val = df_fixed[column].iloc[i+1] if i < len(df_fixed)-1 else None

        if prev_val is not None and next_val is not None:
            df_fixed.at[i, column] = (prev_val + next_val) / 2
        elif prev_val is not None:
            df_fixed.at[i, column] = prev_val
        elif next_val is not None:
            df_fixed.at[i, column] = next_val

    return df_fixed



def main():

    print(f"Carico file: {MERGED_CLEAN_UNIFIED}")
    df = pd.read_excel(MERGED_CLEAN_UNIFIED)

    # 1) Trova outlier SENZA modificare nulla
    outliers, mask = find_outliers(df)

    # 2) Interpola SOLO gli outlier singoli
    df_clean = interpolate_point_by_point(df, mask)

    df_clean= df_clean.drop(["weather_cloudy","Production_KWh_roll_max","Production_KWh_roll_min"], axis=1, errors='ignore')

    # 3) Salva il nuovo file (pulito)
    df_clean.to_excel(OUTPUT_INTERPOLATED, index=False)
    print(f"\nFile pulito salvato in: {OUTPUT_INTERPOLATED}")


if __name__ == "__main__":
    main()
