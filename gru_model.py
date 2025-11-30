import torch
import torch.nn as nn
import pandas as pd
import numpy as np

from split import split_and_scale
from dataloader import create_dataloader 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ======================================================
# 1. MODELLO GRU DIRECT MULTISTEP
# ======================================================
class GRUDirectModel(nn.Module):
    def __init__(self, 
                 enc_in_dim,
                 future_time_dim=4,
                 hidden_dim=128,
                 num_layers=2,
                 forecast_horizon=24):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.forecast_horizon = forecast_horizon

        # ENCODER GRU
        self.encoder = nn.GRU(
            input_size=enc_in_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        # MLP finale (hidden + future_time)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim + future_time_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x_enc, x_future_time):
        # Encoder: hidden state
        _, h = self.encoder(x_enc)
        h_last = h[-1]  # [B, hidden_dim]

        # Ripetiamo hidden 24 volte
        h_rep = h_last.unsqueeze(1).repeat(1, self.forecast_horizon, 1)

        # Concatenazione hidden + future_time_features
        fused = torch.cat([h_rep, x_future_time], dim=-1)

        # MLP per ogni timestep
        y_hat = self.mlp(fused).squeeze(-1)
        return y_hat


# ======================================================
# 2. FUNZIONI TRAIN/VAL
# ======================================================
criterion = nn.MSELoss()

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for x_enc, x_future, y in loader:
        x_enc, x_future, y = x_enc.to(device), x_future.to(device), y.to(device)

        optimizer.zero_grad()
        y_hat = model(x_enc, x_future)
        loss = criterion(y_hat, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def eval_one_epoch(model, loader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x_enc, x_future, y in loader:
            x_enc, x_future, y = x_enc.to(device), x_future.to(device), y.to(device)
            y_hat = model(x_enc, x_future)
            loss = criterion(y_hat, y)
            total_loss += loss.item()
    return total_loss / len(loader)



# ======================================================
# 3. MAIN
# ======================================================
if __name__ == "__main__":
    
    # 1) Carica dataset
    df = pd.read_excel("data/merged_dataset_final_interpolated.xlsx",
                       index_col=0, parse_dates=True)

    # 2) Split e scaling
    train_scaled, val_scaled, test_scaled, feature_scaler, target_scaler = split_and_scale(df)

    # 3) Colonne
    TARGET_COL = "Production_KWh"
    future_time_cols = ["hour_sin", "hour_cos", "doy_sin", "doy_cos"]
    encoder_cols = [c for c in train_scaled.columns if c != TARGET_COL]

    # 4) Dataloaders
    train_dl = create_dataloader(train_scaled, encoder_cols, future_time_cols,
                                 batch_size=32)
    val_dl   = create_dataloader(val_scaled, encoder_cols, future_time_cols,
                                 batch_size=32)
    test_dl  = create_dataloader(test_scaled, encoder_cols, future_time_cols,
                                 batch_size=1, shuffle=False)

    # 5) Modello
    model = GRUDirectModel(
        enc_in_dim=len(encoder_cols),
        future_time_dim=len(future_time_cols),
        hidden_dim=128,
        num_layers=2,
        forecast_horizon=24
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 6) Training
    for epoch in range(1, 21):
        train_loss = train_one_epoch(model, train_dl, optimizer, device)
        val_loss = eval_one_epoch(model, val_dl, device)
        print(f"Epoch {epoch:02d} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

    
    # 7) INFERENCE SUL TEST
    model.eval()

    all_predictions = []

    with torch.no_grad():
        for x_enc, x_future, y in test_dl:
            x_enc = x_enc.to(device)
            x_future = x_future.to(device)

            # prediction scalata
            y_hat_scaled = model(x_enc, x_future).cpu().numpy()[0]  # shape [24]

            all_predictions.append(y_hat_scaled)

    # Prendiamo SOLO l’ultima finestra (come richiesto dal progetto)
    last_pred_scaled = np.array(all_predictions[-1]).reshape(-1, 1)  # [24, 1]

    # 8) INVERSE SCALING
    last_pred = target_scaler.inverse_transform(last_pred_scaled).flatten()  # [24]

    # 9) Costruzione timestamp futuri (ultimo timestamp + 24 ore)
    last_timestamp = test_scaled.index[-1]
    future_times = pd.date_range(
        start=last_timestamp + pd.Timedelta(hours=1),
        periods=24,
        freq="H"
    )

    # 10) Costruzione DataFrame finale
    df_pred = pd.DataFrame({
        "timestamp": future_times,
        "value": last_pred
    })

    # 11) Salvataggio CSV
    df_pred.to_csv("predictions.csv", index=False)
    print("✓ File salvato: predictions.csv")