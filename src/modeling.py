import numpy as np
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor

def get_models(random_state=42):
    models = {
        "ols": LinearRegression(),
        "ridge": RidgeCV(alphas=np.logspace(-3,3,50)),
        "lasso": LassoCV(alphas=np.logspace(-3,1,30), random_state=random_state, max_iter=5000),
        "rf": RandomForestRegressor(n_estimators=400, random_state=random_state, n_jobs=-1),
        "svr": SVR(C=5, epsilon=0.1, kernel="rbf"),
        "xgb": XGBRegressor(
            n_estimators=600,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            reg_lambda=1.0,
            tree_method="hist",
            base_score=0.5,
        )
    }
    return models
