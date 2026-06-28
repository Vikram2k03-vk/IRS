import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error

from xgboost import XGBRegressor

# =====================================
# LOAD DATASET
# =====================================

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
# NORMALIZE INPUT
# =====================================

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# =====================================
# TRAIN 8 INDIVIDUAL MODELS
# =====================================

models = []

print("\nTraining Individual XGBoost Models...\n")

for i in range(8):

    print(f"Training Model {i+1}/8")

    model = XGBRegressor(

        objective="reg:squarederror",

        n_estimators=1000,

        learning_rate=0.03,

        max_depth=8,

        subsample=0.9,

        colsample_bytree=0.9,

        random_state=42,

        n_jobs=-1

    )

    model.fit(

        X_train,

        y_train[:, i]

    )

    models.append(model)

print("\nAll Models Trained Successfully")

# =====================================
# VALIDATION
# =====================================

predictions = np.column_stack(

    [

        model.predict(X_val)

        for model in models

    ]

)

mae = mean_absolute_error(

    y_val,

    predictions

)

accuracy = (1 - mae) * 100

print("\n==============================")
print(" XGBOOST VALIDATION RESULTS")
print("==============================")

print(f"Validation MAE       : {mae:.6f}")
print(f"Estimated Accuracy   : {accuracy:.2f}%")

# =====================================
# PER OUTPUT MAE
# =====================================

print("\nPer Output MAE")

for i in range(8):

    output_mae = mean_absolute_error(

        y_val[:, i],

        predictions[:, i]

    )

    if i % 2 == 0:
        label = f"Cos(Antenna {(i//2)+1})"
    else:
        label = f"Sin(Antenna {(i//2)+1})"

    print(f"{label:<18}: {output_mae:.6f}")

# =====================================
# SAVE MODELS
# =====================================

joblib.dump(

    models,

    "best_xgboost_models.pkl"

)

joblib.dump(

    scaler,

    "xgboost_scaler.pkl"

)

print("\n==============================")
print("Models Saved Successfully")
print("==============================")

print("best_xgboost_models.pkl")
print("xgboost_scaler.pkl")