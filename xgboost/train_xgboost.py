import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import MinMaxScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error

from xgboost import XGBRegressor

print("Loading Dataset...")

train_df = pd.read_csv("../data/train_split.csv")
val_df = pd.read_csv("../data/validation_split.csv")

X_train = train_df.iloc[:, :181].values
y_train = train_df.iloc[:, 181:].values

X_val = val_df.iloc[:, :181].values
y_val = val_df.iloc[:, 181:].values

print("X_train:", X_train.shape)
print("y_train:", y_train.shape)
print("X_val:", X_val.shape)
print("y_val:", y_val.shape)

# =====================================
# NORMALIZATION
# =====================================

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# =====================================
# XGBOOST MODEL
# =====================================

model = MultiOutputRegressor(
    XGBRegressor(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1
    )
)

print("\nTraining XGBoost...")

model.fit(
    X_train,
    y_train
)

# =====================================
# VALIDATION
# =====================================

predictions = model.predict(X_val)

mae = mean_absolute_error(
    y_val,
    predictions
)

accuracy = (1 - mae) * 100

print("\nValidation MAE :", mae)
print("Estimated Accuracy :", accuracy)

# =====================================
# SAVE MODEL
# =====================================

joblib.dump(
    model,
    "best_xgboost_model.pkl"
)

joblib.dump(
    scaler,
    "xgboost_scaler.pkl"
)

print("\nModel Saved")
print("best_xgboost_model.pkl")
print("xgboost_scaler.pkl")