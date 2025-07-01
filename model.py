import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class NeuralNet(nn.Module):
    def __init__(self, input_dim):
        super(NeuralNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 32),
            nn.ReLU(),
            
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.model(x)


class Trainer:
    def __init__(self, input_dim, learning_rate=0.005, batch_size=2048, epochs=2, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = NeuralNet(input_dim).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.batch_size = batch_size
        self.epochs = epochs

    def train(self, X_train, y_train):
        X_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_tensor = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32)
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        print(f"[INFO] Training on device: {self.device}")
        print(f"[INFO] X_tensor shape: {X_tensor.shape}")
        print(f"[INFO] y_tensor shape: {y_tensor.shape}")

        for epoch in range(self.epochs):
            self.model.train()
            running_loss = 0.0
            print(f"\n[INFO] Epoch {epoch + 1}/{self.epochs}")
            for batch_idx, (X_batch, y_batch) in enumerate(loader):
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

                if epoch == 0 and batch_idx == 0:
                    print(f"[DEBUG] First batch X shape: {X_batch.shape}")
                    print(f"[DEBUG] First batch y shape: {y_batch.shape}")

                self.optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                loss.backward()

                total_norm = 0
                for p in self.model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5

                #print(f"Batch {batch_idx + 1} of {epoch + 1}: Loss = {loss.item():.6f}, Grad Norm = {total_norm:.4f}")

                self.optimizer.step()
                running_loss += loss.item()

            print(f" Epoch {epoch + 1} completed. Total Loss: {running_loss:.4f}")

        print("Training completed.")

    def evaluate(self, X_test, y_test, batch_size=1024):
        self.model.eval()

        # Ensure data is created on CPU, not GPU
        X_test_tensor = torch.from_numpy(X_test).float().cpu()
        y_test_tensor = torch.from_numpy(y_test.reshape(-1, 1)).float().cpu()

        dataset = TensorDataset(X_test_tensor, y_test_tensor)
        loader = DataLoader(dataset, batch_size=batch_size)

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                preds = self.model(X_batch)

                all_preds.append(preds.cpu())
                all_targets.append(y_batch.cpu())

        y_pred_np = torch.cat(all_preds).numpy()
        y_test_np = torch.cat(all_targets).numpy()

        rmse = mean_squared_error(y_test_np, y_pred_np) ** 0.5
        mae = mean_absolute_error(y_test_np, y_pred_np)
        r2 = r2_score(y_test_np, y_pred_np)

        return {"MSE": rmse, "MAE": mae, "R² Score": r2}

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            if isinstance(X, np.ndarray):
                X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            elif isinstance(X, torch.Tensor):
                X_tensor = X.to(self.device).float()
            else:
                raise ValueError("Input must be a numpy array or torch tensor")

            preds = self.model(X_tensor).cpu().numpy()

        return preds

