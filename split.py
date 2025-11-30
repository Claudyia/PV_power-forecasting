import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def split_and_scale(df, target_col="Production_KWh",
                    time_cols=["hour_sin","hour_cos","doy_sin","doy_cos"],
                    train_ratio=0.6, val_ratio=0.2):
    
    # -------------------------------
    # 1. SPLIT TEMPORALE
    # -------------------------------
    n = len(df)
    train_size = int(n * train_ratio)
    val_size = int(n * val_ratio)

    train_df = df.iloc[:train_size]
    val_df   = df.iloc[train_size : train_size + val_size]
    test_df  = df.iloc[train_size + val_size :]

    print("✓ Temporal split completed")
    print(f"Train: {len(train_df)} ({train_df.index.min()} → {train_df.index.max()})")
    print(f"Val:   {len(val_df)}   ({val_df.index.min()} → {val_df.index.max()})")
    print(f"Test:  {len(test_df)}  ({test_df.index.min()} → {test_df.index.max()})")

    # -------------------------------
    # 2. FEATURES DA SCALARE
    # -------------------------------
    feature_cols = [c for c in df.columns if c not in {target_col, "pv_date"}]

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    scale_cols   = [c for c in feature_cols if c in numeric_cols and c not in time_cols]

    # -------------------------------
    # 3. SCALING SENZA LEAKAGE
    # -------------------------------
    feature_scaler = StandardScaler()
    target_scaler  = StandardScaler()

    # Fit SOLO sul train
    train_scaled_block = feature_scaler.fit_transform(train_df[scale_cols])
    val_scaled_block   = feature_scaler.transform(val_df[scale_cols])
    test_scaled_block  = feature_scaler.transform(test_df[scale_cols])

    y_train_scaled = target_scaler.fit_transform(train_df[[target_col]])
    y_val_scaled   = target_scaler.transform(val_df[[target_col]])
    y_test_scaled  = target_scaler.transform(test_df[[target_col]])

    # -------------------------------
    # 4. RICOSTRUZIONE SENZA SCALARE TIME FEATURES
    # -------------------------------
    def rebuild(df_orig, scaled_block, y_scaled):
        df_out = df_orig.copy()
        df_out[scale_cols] = scaled_block
        df_out[target_col] = y_scaled
        return df_out

    train_scaled = rebuild(train_df, train_scaled_block, y_train_scaled)
    val_scaled   = rebuild(val_df, val_scaled_block, y_val_scaled)
    test_scaled  = rebuild(test_df, test_scaled_block, y_test_scaled)

    print("✓ Scaling completed (time features untouched)")
    
    return train_scaled, val_scaled, test_scaled, feature_scaler, target_scaler


if __name__ == "__main__":
    df = pd.read_excel("data/merged_dataset_final_interpolated.xlsx", index_col=0, parse_dates=True)

    train_df, val_df, test_df, feature_scaler, target_scaler = split_and_scale(df)

    
