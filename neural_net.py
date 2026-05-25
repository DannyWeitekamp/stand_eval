import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm  # for progress bar


class NeuralClassifier(nn.Module):
    def __init__(self, hidden_dim=100, num_hidden_layers=3, n_classes=2, verbose=False):
        """
        Fully connected feedforward neural network for binary classification.

        Args:
            input_dim (int): Number of input features.
            hidden_dim (int): Number of neurons in each hidden layer.
            num_hidden_layers (int): Number of hidden layers.
        """
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
        self.n_classes = n_classes
        self.classes_ = list(range(n_classes))
        self.verbose = verbose

    def make_model(self, input_dim):
        super(NeuralClassifier, self).__init__()

        layers = []
        layers.append(nn.Linear(input_dim, self.hidden_dim))
        layers.append(nn.ReLU())

        for _ in range(self.num_hidden_layers - 1):
            layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(self.hidden_dim, self.n_classes))
        layers.append(nn.LogSoftmax(dim=1)) 

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        if(isinstance(x, np.ndarray)):
            x  = torch.from_numpy(x.astype(np.float32))

        return self.model(x)


    def predict(self, x):
        log_probs = self.forward(x).exp()
        pred = torch.argmax(log_probs, dim=1)
        return pred.detach().numpy()

    def predict_proba(self, x):
        y_prob = self.forward(x).exp()
        return y_prob.detach().numpy()

    def fit(
        self,
        X_train,
        y_train,
        epochs=100,
        batch_size=100,
        lr=5e-3,
        X_val=None,
        y_val=None,
        verbose=False,
        device=None,
    ):
        """
        Trains the model on given training data.

        Args:
            X_train (torch.Tensor): Input features (N, input_dim)
            y_train (torch.Tensor): Binary labels (N, 1)
            epochs (int): Number of training epochs.
            batch_size (int): Size of each training batch.
            lr (float): Learning rate.
            X_val (torch.Tensor, optional): Validation features.
            y_val (torch.Tensor, optional): Validation labels.
            verbose (bool): Whether to print progress.
            device (str or torch.device): Device to use ("cuda" or "cpu").
        """

        self.make_model(X_train.shape[1])

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device)

        self.to(device)

        
        if(isinstance(X_train, np.ndarray)):
            X_train  = torch.from_numpy(X_train.astype(np.float32))
            y_train  = torch.from_numpy(y_train.astype(np.int64))


        X_train, y_train = X_train.to(device), y_train.to(device)

        dataset = TensorDataset(X_train, y_train)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        criterion = nn.NLLLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        for epoch in range(epochs):
            self.train()
            epoch_loss = 0.0
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = self(batch_X)
                # print(outputs, batch_y)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * batch_X.size(0)

            avg_loss = epoch_loss / len(dataset)

            if verbose or self.verbose:
                msg = f"Epoch [{epoch + 1}/{epochs}] - Loss: {avg_loss:.4f}"

                if X_val is not None and y_val is not None:
                    self.eval()
                    with torch.no_grad():
                        X_val_d, y_val_d = X_val.to(device), y_val.to(device)
                        val_outputs = self(X_val_d)
                        val_loss = criterion(val_outputs, y_val_d)
                        msg += f" - Val Loss: {val_loss.item():.4f}"
                print(msg)


# Example usage
if __name__ == "__main__":
    # Dummy data
    X = torch.randn(500, 20)
    y = torch.randint(0, 2, (500,))  # Class indices: 0 to 3
    # y = (torch.rand(500, 1) > 0.5).float()

    # Split into train and val
    X_train, X_val = X[:400], X[400:]
    y_train, y_val = y[:400], y[400:]

    model = NeuralClassifier(input_dim=20, hidden_dim=64, num_hidden_layers=2)
    model.fit(X_train, y_train, epochs=10, lr=1e-3, X_val=X_val, y_val=y_val)

    print(model.predict(X_train))
