from marketify.models.drift import (DriftResult, check_feature_drift,
                                    check_prediction_drift)
from marketify.models.ensemble import (TradeIdeaInput, build_trade_idea,
                                       compute_cvar_95)

try:
    from marketify.models.xgb_model import XGBModel, rolling_train_predict
except ModuleNotFoundError as exc:
    if getattr(exc, "name", None) != "xgboost":
        raise
    XGBModel = None

    def rolling_train_predict(*args, **kwargs):
        raise RuntimeError("xgboost is required for rolling_train_predict")

__all__ = [
    "XGBModel",
    "rolling_train_predict",
    "TradeIdeaInput",
    "build_trade_idea",
    "compute_cvar_95",
    "DriftResult",
    "check_prediction_drift",
    "check_feature_drift",
]
