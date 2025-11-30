from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from split import split_and_scale

class GRUDirectDataset(Dataset):
    def __init__(self, df,
                 input_length=168,
                 forecast_horizon=24,
                 target_col="Production_KWh",
                 encoder_cols=None,
                 future_time_cols=None):

        self.df = df
        self.input_length = input_length
        self.forecast_horizon = forecast_horizon
        self.target_col = target_col
        self.encoder_cols = encoder_cols
        self.future_time_cols = future_time_cols

    def __len__(self):
        return len(self.df) - self.input_length - self.forecast_horizon

    def __getitem__(self, idx):

        # -------------------------
        # 1. PAST INPUTS (encoder)
        # -------------------------
        x_enc = self.df.iloc[
            idx : idx + self.input_length
        ][self.encoder_cols].values.astype("float32")

        # -------------------------
        # 2. FUTURE TIME FEATURES
        # -------------------------
        x_future_time = self.df.iloc[
            idx + self.input_length : idx + self.input_length + self.forecast_horizon
        ][self.future_time_cols].values.astype("float32")

        # -------------------------
        # 3. TARGET (24 future steps)
        # -------------------------
        y = self.df.iloc[
            idx + self.input_length : idx + self.input_length + self.forecast_horizon
        ][self.target_col].values.astype("float32")

        return (
            torch.tensor(x_enc),          # [168, num_features]
            torch.tensor(x_future_time),  # [24, 4]
            torch.tensor(y)               # [24]
        )


def create_dataloader(df, encoder_cols, future_time_cols,
                      input_length=168, forecast_horizon=24,
                      batch_size=32, shuffle=True,
                      target_col="Production_KWh"):

    dataset = GRUDirectDataset(
        df=df,
        input_length=input_length,
        forecast_horizon=forecast_horizon,
        target_col=target_col,
        encoder_cols=encoder_cols,
        future_time_cols=future_time_cols
    )

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


if __name__ == "__main__":
    TARGET_COL = "Production_KWh"
    FUTURE_TIME_COLS = ["hour_sin", "hour_cos", "doy_sin", "doy_cos"]

    data_path = Path("processed/merged_dataset_final_interpolated.xlsx")
    if not data_path.exists():
        raise FileNotFoundError(f"{data_path} non trovato. Genera prima il dataset.")

    df_full = pd.read_excel(data_path)

    train_scaled, val_scaled, test_scaled, _, _ = split_and_scale(df_full, target_col=TARGET_COL)

    encoder_cols = [c for c in train_scaled.columns if c not in [TARGET_COL, "pv_date"]]

    train_dl = create_dataloader(train_scaled, encoder_cols, FUTURE_TIME_COLS,
                                 input_length=168, forecast_horizon=24,
                                 batch_size=32, shuffle=True)

    val_dl = create_dataloader(val_scaled, encoder_cols, FUTURE_TIME_COLS,
                               input_length=168, forecast_horizon=24,
                               batch_size=32, shuffle=False)

    test_dl = create_dataloader(test_scaled, encoder_cols, FUTURE_TIME_COLS,
                                input_length=168, forecast_horizon=24,
                                batch_size=1, shuffle=False)

    print("Train batches:", len(train_dl))
    print("Val batches:", len(val_dl))
    print("Test batches:", len(test_dl))
