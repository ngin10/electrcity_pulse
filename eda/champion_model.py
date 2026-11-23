"""The champion as one deployable model: the point prediction and its calibrated 80% interval together.

Input: the feature store's columns for each target day. Output, per day:
  predicted_share  the equal average of Bayesian ridge and LightGBM, % of GB generation from renewables
  lower_80         the prediction minus the calibrated half-width, floored at 0
  upper_80         the prediction plus the calibrated half-width, capped at 100

The half-width comes from conformal calibration on earlier out-of-sample errors (calibrate_intervals.py) and is
refreshed whenever the model is recalibrated.

MLflow reloads this class from this file (it is logged with the model as code).
"""
import mlflow
import numpy as np
import pandas as pd


class ChampionModel(mlflow.pyfunc.PythonModel):

    def __init__(self, point_model, half_width, calibration):
        self.point_model = point_model
        self.half_width = float(half_width)
        self.calibration = calibration

    def predict(self, context, model_input, params=None):
        point = np.asarray(self.point_model.predict(model_input), dtype=float)
        return pd.DataFrame({'predicted_share': point.clip(0, 100),
                             'lower_80': (point - self.half_width).clip(0, 100),
                             'upper_80': (point + self.half_width).clip(0, 100)}, index=model_input.index)
