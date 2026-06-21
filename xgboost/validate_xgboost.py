import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import MinMaxScaler

# =====================================
# PHASE CONVERSION
# =====================================

def sincos_to_phase(cos_val, sin_val):

    phase = np.degrees(
        np.arctan2(
            sin_val,
            cos_val
        )
    )

    if phase < 0:
        phase += 360

    return phase


def circular_error(actual, predicted):

    diff = np.abs(
        actual - predicted
    )

    return np.minimum(
        diff,
        360 - diff
    )


# =====================================
# LOAD DATA
# =====================================

print("Loading Validation Dataset...")

val_df = pd.read_csv(
    "../data/validation_split.csv"
)

X_val = val_df.iloc[:, :181].values
y_val = val_df.iloc[:, 181:].values

# =====================================
# LOAD SCALER
# =====================================

print("Loading Scaler...")

scaler = joblib.load(
    "xgboost_scaler.pkl"
)

X_val = scaler.transform(
    X_val
)

# =====================================
# LOAD MODEL
# =====================================

print("Loading Model...")

model = joblib.load(
    "best_xgboost_model.pkl"
)

# =====================================
# PREDICT
# =====================================

print("Predicting...")

predictions = model.predict(
    X_val
)

# =====================================
# PHASE ERROR
# =====================================

actual_phases = []
predicted_phases = []

for i in range(len(y_val)):

    actual_row = []
    predicted_row = []

    for antenna in range(4):

        idx = antenna * 2

        actual_cos = y_val[i][idx]
        actual_sin = y_val[i][idx + 1]

        pred_cos = predictions[i][idx]
        pred_sin = predictions[i][idx + 1]

        actual_phase = sincos_to_phase(
            actual_cos,
            actual_sin
        )

        predicted_phase = sincos_to_phase(
            pred_cos,
            pred_sin
        )

        actual_row.append(
            actual_phase
        )

        predicted_row.append(
            predicted_phase
        )

    actual_phases.append(
        actual_row
    )

    predicted_phases.append(
        predicted_row
    )

actual_phases = np.array(
    actual_phases
)

predicted_phases = np.array(
    predicted_phases
)

errors = circular_error(
    actual_phases,
    predicted_phases
)

mean_error = np.mean(
    errors
)

max_error = np.max(
    errors
)

accuracy = (
    1 - mean_error / 180
) * 100

print("\n===== XGBOOST RESULTS =====")

print(
    f"Average Phase Error : {mean_error:.2f} degrees"
)

print(
    f"Maximum Phase Error : {max_error:.2f} degrees"
)

print(
    f"Estimated Accuracy  : {accuracy:.2f}%"
)