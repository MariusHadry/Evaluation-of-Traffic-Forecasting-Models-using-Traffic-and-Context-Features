import numpy as np


def smape(actuals: np.ndarray, predicted: np.ndarray):
    return 1 / len(actuals) * np.sum(2 * np.abs(predicted - actuals) / (np.abs(actuals) + np.abs(predicted)))
