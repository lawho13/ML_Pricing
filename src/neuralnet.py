import numpy as np
import torch
import torch.nn as nn


class NeuralNetModel:
    def __init__(self, hidden=32, epochs=100, 
                 lr=1e-3, l1= 1e-4, batch_size = 10000,
                 patience = 5):
        # l1 (1e-5, 1e-3)
        # lr {0.001, 0.01}
        
        self.hidden = hidden
        self.epochs = epochs
        self.lr = lr
        self.l1 = l1
        self.batch_size = batch_size
        self.patience = patience
        self.model = None

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        X = torch.tensor(np.asarray(X_train), dtype=torch.float32)
        y = torch.tensor(np.asarray(y_train), dtype=torch.float32)
        if X_val is not None:
            X_val = torch.tensor(np.asarray(X_val), dtype=torch.float32)
            y_val = torch.tensor(np.asarray(y_val), dtype=torch.float32)

        self.model = nn.Sequential(
            nn.Linear(X.shape[1], self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, 1),
        )

        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        mse_loss = nn.MSELoss()

        best_val = float('inf')
        patience_count = 0

        n = X_train.shape[0]

        for _ in range(self.epochs):
            
            # SGD
            perm = torch.randperm(n)
            X_train = X_train[perm]
            y_train = y_train[perm]

            # evaluate gradient from small random subsets (accuracy down for faster optimization)
            for i in range(0, n, self.batch_size):
                X_train_batch = X_train[i:i+self.batch_size]
                y_train_batch = y_train[i:i+self.batch_size]

                opt.zero_grad()

                self.model(X_train_batch).squeeze()

                pred = self.model(X_train_batch).squeeze()
                mse = mse_loss(pred, y_train_batch)

                # Paper specifies L1 Regularization 

                l1_norm = sum(p.abs().sum() for p in self.model.parameters())
                loss = mse +self.l1*l1_norm

                loss.backward()
                opt.step()

            if X_val is not None:
                with torch.no_grad():
                    val_pred = self.model(X_val).squeeze()
                    val_loss = mse_loss(val_pred, y_val).item()

                if val_loss < best_val:
                    best_val = val_loss
                    patience_count = 0
                    best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                else:
                    patience_count += 1
                    if patience_count >= self.patience:
                        self.model.load_state_dict(best_state)
                        break

    def predict(self, X):
        X = torch.tensor(np.asarray(X), dtype=torch.float32)
        with torch.no_grad():
            return self.model(X).squeeze().numpy()
