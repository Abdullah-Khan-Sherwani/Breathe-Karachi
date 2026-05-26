import numpy as np


class PerHorizonWrapper:
    """Trains one base model per prediction horizon and stacks predictions."""

    models: list

    def __init__(self, models: list):
        self.models = models

    def predict(self, X) -> np.ndarray:
        preds = np.column_stack([m.predict(X) for m in self.models])
        return preds

    # Expose estimators_ so SHAP/sklearn tools can access per-horizon models
    @property
    def estimators_(self):
        return self.models
