"""Weighted ensemble that combines predictions from all base models.

Applies a static weight vector to the individual model predictions
and produces a final blended forecast. Weights are read from config.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class WeightedEnsemble:
    """Combine multiple forecaster predictions via a weighted average.

    Parameters
    ----------
    model_names : List[str]
        Ordered list of model names (must match weight dict keys).
    weights : Dict[str, float]
        Mapping from model name to non-negative weight. Weights are
        normalised to sum to 1 internally.
    """

    def __init__(
        self,
        model_names: List[str],
        weights: Dict[str, float],
    ) -> None:
        self.model_names = model_names
        raw_weights = np.array([weights.get(name, 0.0) for name in model_names])
        total = raw_weights.sum()
        if total <= 0:
            raise ValueError("At least one model weight must be positive.")
        self.weights = raw_weights / total
        logger.info(
            "Ensemble weights (normalised): %s",
            dict(zip(model_names, self.weights.round(4))),
        )

    def predict(
        self,
        predictions: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """Blend individual model predictions into a single forecast.

        Parameters
        ----------
        predictions : Dict[str, np.ndarray]
            Mapping from model name to 1-D prediction array. All arrays
            must have the same length.

        Returns
        -------
        np.ndarray
            Weighted-average forecast array.

        Raises
        ------
        ValueError
            If prediction arrays have inconsistent lengths.
        """
        arrays: List[np.ndarray] = []
        for name, weight in zip(self.model_names, self.weights):
            if name not in predictions:
                logger.warning("Model '%s' missing from predictions — using zeros.", name)
                n = next(iter(predictions.values())).shape[0]
                arrays.append(np.zeros(n) * weight)
            else:
                arrays.append(predictions[name] * weight)

        lengths = {len(a) for a in arrays}
        if len(lengths) > 1:
            raise ValueError(f"Prediction arrays have inconsistent lengths: {lengths}")

        blended = np.sum(arrays, axis=0)
        logger.info(
            "Ensemble prediction: mean=%.4f  std=%.4f  min=%.4f  max=%.4f",
            blended.mean(),
            blended.std(),
            blended.min(),
            blended.max(),
        )
        return blended

    @classmethod
    def from_config(
        cls,
        config: Dict[str, Any],
        model_names: Optional[List[str]] = None,
    ) -> "WeightedEnsemble":
        """Instantiate WeightedEnsemble from project configuration.

        Parameters
        ----------
        config : Dict[str, Any]
            Project configuration dictionary.
        model_names : Optional[List[str]]
            Override the default model order. If None, uses all keys
            present in ``config['ensemble']['weights']``.

        Returns
        -------
        WeightedEnsemble
            New instance.
        """
        ens_cfg = config["ensemble"]
        weights = ens_cfg["weights"]
        if model_names is None:
            model_names = list(weights.keys())
        return cls(model_names=model_names, weights=weights)
