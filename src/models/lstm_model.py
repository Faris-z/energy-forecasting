"""PyTorch LSTM model for sequence-to-one energy forecasting.

Implements a multi-layer bidirectional-optional LSTM with dropout,
gradient clipping, early stopping, and GPU support.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class TimeSeriesDataset:
    """Sliding window dataset for LSTM sequence modelling.

    Parameters
    ----------
    data : np.ndarray
        2-D array of shape (T, F) where T is timesteps and F is features.
    targets : np.ndarray
        1-D array of target values aligned with ``data``.
    seq_len : int
        Number of historical timesteps per input window.
    """

    def __init__(
        self,
        data: np.ndarray,
        targets: np.ndarray,
        seq_len: int,
    ) -> None:
        self.data = data
        self.targets = targets
        self.seq_len = seq_len

    def __len__(self) -> int:
        """Return number of valid windows."""
        return max(0, len(self.data) - self.seq_len)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        """Return (sequence, target) tensors for index ``idx``."""
        import torch

        x = self.data[idx : idx + self.seq_len]
        y = self.targets[idx + self.seq_len]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


class LSTMNet:
    """Thin wrapper: defines the actual nn.Module."""

    @staticmethod
    def build(
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
    ) -> Any:
        """Build and return an ``nn.Module`` LSTM regressor.

        Parameters
        ----------
        input_size : int
            Number of input features per timestep.
        hidden_size : int
            Number of hidden units per LSTM layer.
        num_layers : int
            Number of stacked LSTM layers.
        dropout : float
            Dropout probability between LSTM layers.
        bidirectional : bool
            Whether to use bidirectional LSTM.

        Returns
        -------
        nn.Module
            PyTorch LSTM regression module.
        """
        import torch.nn as nn

        directions = 2 if bidirectional else 1

        class _LSTMRegressor(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=dropout if num_layers > 1 else 0.0,
                    batch_first=True,
                    bidirectional=bidirectional,
                )
                self.fc = nn.Linear(hidden_size * directions, 1)
                self.dropout = nn.Dropout(dropout)

            def forward(self, x: Any) -> Any:
                out, _ = self.lstm(x)
                out = self.dropout(out[:, -1, :])
                return self.fc(out).squeeze(-1)

        return _LSTMRegressor()


class LSTMForecaster:
    """PyTorch LSTM forecaster with early stopping and GPU support.

    Parameters
    ----------
    hidden_size : int
        LSTM hidden units.
    num_layers : int
        Number of stacked LSTM layers.
    dropout : float
        Dropout probability.
    sequence_length : int
        Historical window length (hours).
    batch_size : int
        Mini-batch size.
    num_epochs : int
        Maximum training epochs.
    learning_rate : float
        Adam optimizer learning rate.
    weight_decay : float
        L2 regularisation coefficient.
    patience : int
        Early-stopping patience (epochs without improvement).
    clip_grad_norm : float
        Gradient clipping max-norm.
    bidirectional : bool
        Whether to use bidirectional LSTM.
    """

    def __init__(
        self,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        sequence_length: int = 168,
        batch_size: int = 64,
        num_epochs: int = 50,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4,
        patience: int = 10,
        clip_grad_norm: float = 1.0,
        bidirectional: bool = False,
    ) -> None:
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.patience = patience
        self.clip_grad_norm = clip_grad_norm
        self.bidirectional = bidirectional

        self._model: Optional[Any] = None
        self._scaler_X: Optional[Any] = None
        self._scaler_y: Optional[Any] = None
        self._input_size: int = 0
        self._device: Optional[Any] = None

    def _get_device(self) -> Any:
        """Determine the best available compute device.

        Returns
        -------
        torch.device
            CUDA if available, otherwise CPU.
        """
        import torch

        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _scale(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """Fit scalers on training data and transform all splits.

        Parameters
        ----------
        X_train : np.ndarray
            Training feature matrix.
        y_train : np.ndarray
            Training targets.
        X_val : Optional[np.ndarray]
            Validation feature matrix.
        y_val : Optional[np.ndarray]
            Validation targets.

        Returns
        -------
        Tuple of scaled arrays.
        """
        from sklearn.preprocessing import StandardScaler

        self._scaler_X = StandardScaler()
        self._scaler_y = StandardScaler()

        X_train_s = self._scaler_X.fit_transform(X_train)
        y_train_s = self._scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()

        X_val_s = self._scaler_X.transform(X_val) if X_val is not None else None
        y_val_s = (
            self._scaler_y.transform(y_val.reshape(-1, 1)).ravel() if y_val is not None else None
        )

        return X_train_s, y_train_s, X_val_s, y_val_s

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "LSTMForecaster":
        """Train the LSTM model.

        Parameters
        ----------
        X_train : np.ndarray
            Training feature matrix of shape (T, F).
        y_train : np.ndarray
            Training target array of length T.
        X_val : Optional[np.ndarray]
            Validation features for early stopping.
        y_val : Optional[np.ndarray]
            Validation targets for early stopping.

        Returns
        -------
        LSTMForecaster
            Fitted instance (self).
        """
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader

        self._device = self._get_device()
        logger.info("Training LSTM on device: %s", self._device)

        X_train_s, y_train_s, X_val_s, y_val_s = self._scale(X_train, y_train, X_val, y_val)

        self._input_size = X_train_s.shape[1]
        self._model = LSTMNet.build(
            input_size=self._input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            bidirectional=self.bidirectional,
        ).to(self._device)

        train_ds = TimeSeriesDataset(X_train_s, y_train_s, self.sequence_length)
        train_dl = DataLoader(train_ds, batch_size=self.batch_size, shuffle=False)

        val_dl = None
        if X_val_s is not None and y_val_s is not None:
            val_ds = TimeSeriesDataset(X_val_s, y_val_s, self.sequence_length)
            val_dl = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)

        optimizer = torch.optim.Adam(
            self._model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        criterion = nn.MSELoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3
        )

        best_val_loss = float("inf")
        patience_counter = 0
        best_state: Optional[Dict] = None

        for epoch in range(1, self.num_epochs + 1):
            self._model.train()
            train_losses: List[float] = []

            for X_batch, y_batch in train_dl:
                X_batch = X_batch.to(self._device)
                y_batch = y_batch.to(self._device)

                optimizer.zero_grad()
                preds = self._model(X_batch)
                loss = criterion(preds, y_batch)
                loss.backward()
                nn.utils.clip_grad_norm_(self._model.parameters(), self.clip_grad_norm)
                optimizer.step()
                train_losses.append(loss.item())

            avg_train_loss = np.mean(train_losses)

            if val_dl is not None:
                self._model.eval()
                val_losses: List[float] = []
                with torch.no_grad():
                    for X_batch, y_batch in val_dl:
                        X_batch = X_batch.to(self._device)
                        y_batch = y_batch.to(self._device)
                        preds = self._model(X_batch)
                        val_losses.append(criterion(preds, y_batch).item())

                avg_val_loss = np.mean(val_losses)
                scheduler.step(avg_val_loss)

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    best_state = {k: v.cpu().clone() for k, v in self._model.state_dict().items()}
                else:
                    patience_counter += 1

                if epoch % 5 == 0 or epoch == 1:
                    logger.info(
                        "Epoch %3d/%d — Train Loss: %.4f  Val Loss: %.4f",
                        epoch,
                        self.num_epochs,
                        avg_train_loss,
                        avg_val_loss,
                    )

                if patience_counter >= self.patience:
                    logger.info("Early stopping triggered at epoch %d.", epoch)
                    break
            else:
                if epoch % 5 == 0 or epoch == 1:
                    logger.info(
                        "Epoch %3d/%d — Train Loss: %.4f",
                        epoch,
                        self.num_epochs,
                        avg_train_loss,
                    )

        if best_state is not None:
            self._model.load_state_dict(best_state)

        logger.info("LSTM training complete.")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions from feature matrix.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (N, F) where N >= sequence_length.
            Predictions are generated for each valid window.

        Returns
        -------
        np.ndarray
            Predicted values in original scale.

        Raises
        ------
        RuntimeError
            If the model has not been fitted.
        """
        import torch
        from torch.utils.data import DataLoader

        if self._model is None or self._scaler_X is None:
            raise RuntimeError("Model not fitted.")

        X_s = self._scaler_X.transform(X)
        dummy_y = np.zeros(len(X_s))
        ds = TimeSeriesDataset(X_s, dummy_y, self.sequence_length)

        if len(ds) == 0:
            logger.warning(
                "Prediction input (%d rows) shorter than sequence_length (%d). "
                "Returning zeros.",
                len(X),
                self.sequence_length,
            )
            return np.zeros(len(X))

        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=False)

        self._model.eval()
        preds: List[np.ndarray] = []

        with torch.no_grad():
            for X_batch, _ in dl:
                X_batch = X_batch.to(self._device)
                out = self._model(X_batch).cpu().numpy()
                preds.append(out)

        preds_scaled = np.concatenate(preds)
        preds_orig = self._scaler_y.inverse_transform(preds_scaled.reshape(-1, 1)).ravel()

        pad = np.full(self.sequence_length, preds_orig[0])
        return np.concatenate([pad, preds_orig])

    def save(self, path: str) -> None:
        """Save model weights and scalers to disk.

        Parameters
        ----------
        path : str
            Directory path to save artefacts.
        """
        import torch
        import joblib

        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)

        if self._model is not None:
            torch.save(self._model.state_dict(), save_dir / "lstm_weights.pt")
        if self._scaler_X is not None:
            joblib.dump(self._scaler_X, save_dir / "scaler_X.pkl")
        if self._scaler_y is not None:
            joblib.dump(self._scaler_y, save_dir / "scaler_y.pkl")

        logger.info("LSTM model saved to %s", save_dir)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "LSTMForecaster":
        """Instantiate from project configuration.

        Parameters
        ----------
        config : Dict[str, Any]
            Project configuration dictionary.

        Returns
        -------
        LSTMForecaster
            New instance.
        """
        return cls(**config["lstm"])
