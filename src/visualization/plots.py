"""Visualization utilities for the energy forecasting project.

All plots are saved automatically to ``reports/figures/`` in PNG format.
No plots are shown interactively (``plt.show()`` is never called) so
the module works in headless/container environments.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")  # headless backend — must precede pyplot import
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _save(fig: Any, name: str, config: Dict[str, Any]) -> Path:
    """Save a matplotlib figure and close it.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save.
    name : str
        Base filename without extension.
    config : Dict[str, Any]
        Project configuration dictionary.

    Returns
    -------
    Path
        Absolute path of the saved file.
    """
    figures_dir = Path(config["paths"]["figures_dir"])
    figures_dir.mkdir(parents=True, exist_ok=True)
    fmt = config["visualization"]["figure_format"]
    dpi = config["visualization"]["figure_dpi"]
    out_path = figures_dir / f"{name}.{fmt}"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved figure → %s", out_path)
    return out_path


def plot_forecast_vs_actual(
    y_true: np.ndarray,
    predictions: Dict[str, np.ndarray],
    index: pd.DatetimeIndex,
    config: Dict[str, Any],
    title: str = "Forecast vs Actual — 24h Horizon",
) -> Path:
    """Plot actual vs predicted values for all models.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth target values.
    predictions : Dict[str, np.ndarray]
        Mapping from model name to prediction array.
    index : pd.DatetimeIndex
        Datetime index for the x-axis.
    config : Dict[str, Any]
        Project configuration dictionary.
    title : str
        Plot title.

    Returns
    -------
    Path
        Path to the saved figure.
    """
    style = config["visualization"]["style"]
    try:
        plt.style.use(style)
    except Exception:
        plt.style.use("ggplot")

    figsize = config["visualization"]["figsize_large"]
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(index, y_true, label="Actual", color="black", linewidth=2, zorder=5)

    colors = plt.cm.tab10(np.linspace(0, 1, len(predictions)))
    for (name, preds), color in zip(predictions.items(), colors):
        length = min(len(index), len(preds))
        ax.plot(
            index[:length],
            preds[:length],
            label=name.replace("_", " ").title(),
            color=color,
            linewidth=1.2,
            alpha=0.85,
            linestyle="--",
        )

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Datetime")
    ax.set_ylabel("Global Active Power (kW)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    return _save(fig, "forecast_vs_actual", config)


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    index: pd.DatetimeIndex,
    config: Dict[str, Any],
) -> Path:
    """Plot residuals (errors) over time and their histogram.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth values.
    y_pred : np.ndarray
        Predicted values.
    model_name : str
        Name of the model (used in title and filename).
    index : pd.DatetimeIndex
        Datetime index.
    config : Dict[str, Any]
        Project configuration dictionary.

    Returns
    -------
    Path
        Path to the saved figure.
    """
    residuals = y_true - y_pred[: len(y_true)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(index[: len(residuals)], residuals, color="#e74c3c", linewidth=0.8)
    axes[0].axhline(0, color="black", linestyle="--", linewidth=1)
    axes[0].set_title(f"{model_name} — Residuals over Time")
    axes[0].set_xlabel("Datetime")
    axes[0].set_ylabel("Residual (kW)")
    axes[0].grid(True, alpha=0.4)

    axes[1].hist(residuals, bins=50, color="#3498db", edgecolor="white", alpha=0.8)
    axes[1].axvline(0, color="black", linestyle="--")
    axes[1].set_title(f"{model_name} — Residual Distribution")
    axes[1].set_xlabel("Residual (kW)")
    axes[1].set_ylabel("Count")
    axes[1].grid(True, alpha=0.4)

    fig.tight_layout()
    safe_name = model_name.lower().replace(" ", "_")
    return _save(fig, f"residuals_{safe_name}", config)


def plot_feature_importance(
    feature_names: List[str],
    importances: np.ndarray,
    model_name: str,
    config: Dict[str, Any],
    top_n: int = 25,
) -> Path:
    """Plot top-N feature importances as a horizontal bar chart.

    Parameters
    ----------
    feature_names : List[str]
        Names of the features.
    importances : np.ndarray
        Importance scores (higher = more important).
    model_name : str
        Model name for title and filename.
    config : Dict[str, Any]
        Project configuration dictionary.
    top_n : int
        Number of top features to display.

    Returns
    -------
    Path
        Path to the saved figure.
    """
    idx = np.argsort(importances)[-top_n:]
    top_features = [feature_names[i] for i in idx]
    top_importances = importances[idx]

    fig, ax = plt.subplots(figsize=(10, max(6, top_n // 2)))
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(top_features)))
    ax.barh(top_features, top_importances, color=colors)
    ax.set_title(f"{model_name} — Top {top_n} Feature Importances", fontweight="bold")
    ax.set_xlabel("Importance Score")
    ax.grid(True, axis="x", alpha=0.4)
    fig.tight_layout()

    safe_name = model_name.lower().replace(" ", "_")
    return _save(fig, f"feature_importance_{safe_name}", config)


def plot_shap_summary(
    model: Any,
    X: np.ndarray,
    feature_names: List[str],
    model_name: str,
    config: Dict[str, Any],
    max_samples: int = 500,
) -> Optional[Path]:
    """Generate a SHAP summary (beeswarm) plot.

    Parameters
    ----------
    model : Any
        Fitted tree model with a ``predict`` method compatible with SHAP.
    X : np.ndarray
        Feature matrix to explain.
    feature_names : List[str]
        Feature names for axis labels.
    model_name : str
        Model name for title and filename.
    config : Dict[str, Any]
        Project configuration dictionary.
    max_samples : int
        Maximum number of rows to pass to SHAP (for speed).

    Returns
    -------
    Optional[Path]
        Path to saved figure, or None if SHAP is unavailable.
    """
    try:
        import shap
    except ImportError:
        logger.warning("shap not installed — skipping SHAP plot.")
        return None

    X_sample = X[:max_samples]
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)

        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(
            shap_values,
            X_sample,
            feature_names=feature_names,
            show=False,
            plot_type="dot",
        )
        safe_name = model_name.lower().replace(" ", "_")
        return _save(plt.gcf(), f"shap_summary_{safe_name}", config)
    except Exception as exc:
        logger.warning("SHAP plot failed for %s: %s", model_name, exc)
        return None


def plot_seasonal_decomposition(
    series: pd.Series,
    config: Dict[str, Any],
    period: int = 24,
) -> Path:
    """Plot additive seasonal decomposition (trend, seasonality, residual).

    Parameters
    ----------
    series : pd.Series
        Hourly target time series.
    config : Dict[str, Any]
        Project configuration dictionary.
    period : int
        Seasonal period in hours (24 for daily).

    Returns
    -------
    Path
        Path to the saved figure.
    """
    from statsmodels.tsa.seasonal import seasonal_decompose

    result = seasonal_decompose(
        series.dropna(), model="additive", period=period, extrapolate_trend="freq"
    )

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    components = [
        (series, "Observed"),
        (result.trend, "Trend"),
        (result.seasonal, "Seasonal"),
        (result.resid, "Residual"),
    ]

    for ax, (data, label) in zip(axes, components):
        ax.plot(data, linewidth=0.8, color="#2c3e50")
        ax.set_ylabel(label, fontsize=10)
        ax.grid(True, alpha=0.3)

    axes[0].set_title(
        "Seasonal Decomposition — Global Active Power (hourly)",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    return _save(fig, "seasonal_decomposition", config)


def plot_correlation_heatmap(
    df: pd.DataFrame,
    config: Dict[str, Any],
    top_n: int = 20,
) -> Path:
    """Plot a Pearson correlation heatmap of the top-N features.

    Parameters
    ----------
    df : pd.DataFrame
        Feature-engineered DataFrame.
    config : Dict[str, Any]
        Project configuration dictionary.
    top_n : int
        Number of columns with highest variance to include.

    Returns
    -------
    Path
        Path to the saved figure.
    """
    try:
        import seaborn as sns
    except ImportError:
        logger.warning("seaborn not installed — using matplotlib heatmap.")
        sns = None

    variances = df.var().sort_values(ascending=False)
    top_cols = variances.head(top_n).index.tolist()
    corr = df[top_cols].corr()

    fig, ax = plt.subplots(figsize=(14, 12))

    if sns is not None:
        sns.heatmap(
            corr,
            ax=ax,
            annot=True,
            fmt=".2f",
            cmap="RdBu_r",
            center=0,
            linewidths=0.5,
            annot_kws={"size": 7},
        )
    else:
        im = ax.imshow(corr.values, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
        plt.colorbar(im, ax=ax)
        ax.set_xticks(range(len(top_cols)))
        ax.set_yticks(range(len(top_cols)))
        ax.set_xticklabels(top_cols, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(top_cols, fontsize=7)

    ax.set_title(
        f"Pearson Correlation — Top {top_n} Features",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    return _save(fig, "correlation_heatmap", config)


def plot_metrics_comparison(
    results: Dict[str, Dict[str, float]],
    config: Dict[str, Any],
) -> Path:
    """Plot grouped bar chart comparing all model evaluation metrics.

    Parameters
    ----------
    results : Dict[str, Dict[str, float]]
        Mapping from model name to dict with keys 'rmse', 'mae', 'mape'.
    config : Dict[str, Any]
        Project configuration dictionary.

    Returns
    -------
    Path
        Path to the saved figure.
    """
    metrics = ["rmse", "mae", "mape"]
    model_names = list(results.keys())
    x = np.arange(len(model_names))
    width = 0.25

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for i, metric in enumerate(metrics):
        values = [results[m].get(metric, 0.0) for m in model_names]
        bars = axes[i].bar(x, values, color=plt.cm.Set2(np.linspace(0, 1, len(model_names))))
        axes[i].set_title(metric.upper(), fontsize=12, fontweight="bold")
        axes[i].set_xticks(x)
        axes[i].set_xticklabels(model_names, rotation=30, ha="right", fontsize=9)
        axes[i].grid(True, axis="y", alpha=0.4)

        for bar, val in zip(bars, values):
            axes[i].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(values) * 0.01,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    fig.suptitle("Model Evaluation Metrics Comparison", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return _save(fig, "metrics_comparison", config)
