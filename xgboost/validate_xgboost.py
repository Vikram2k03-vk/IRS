import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
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

X_all = val_df.iloc[:, :181].values
y_all = val_df.iloc[:, 181:].values

print(f"\nAvailable Validation Samples : {len(X_all)}")

sample = int(input(f"Select Sample (0-{len(X_all)-1}): "))

if sample < 0 or sample >= len(X_all):
    print("Invalid sample.")
    exit()

X_selected = X_all[sample].reshape(1, -1)
y_selected = y_all[sample].reshape(1, -1)

# =====================================
# LOAD SCALER
# =====================================

print("Loading Scaler...")

scaler = joblib.load(
    "xgboost_scaler.pkl"
)

X_val_original = X_selected.copy()

X_selected = scaler.transform(X_selected)

# =====================================
# LOAD MODEL
# =====================================

print("Loading Model...")

models = joblib.load(
    "best_xgboost_models.pkl"
)

print("Predicting...")

predictions = np.column_stack(
    [
        model.predict(X_selected)
        for model in models
    ]
)

# =====================================
# PHASE ERROR
# =====================================

actual_phases = []
predicted_phases = []

for i in range(1):

    actual_row = []
    predicted_row = []

    for antenna in range(4):

        idx = antenna * 2

        actual_cos = y_selected[i][idx]
        actual_sin = y_selected[i][idx + 1]

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

print("\nActual Phases")

for i in range(4):
    print(
        f"Antenna {i+1}: {actual_phases[0][i]:.2f}°"
    )

print("\nPredicted Phases")

for i in range(4):
    print(
        f"Antenna {i+1}: {predicted_phases[0][i]:.2f}°"
    )

# =====================================
# PLOT RADIATION PATTERN
# =====================================

angles = np.arange(181)

plt.figure(figsize=(12, 6))

plt.plot(
    angles,
    X_val_original[0],
    color="blue",
    linewidth=2,
    label="Validation Radiation Pattern"
)

plt.title("Validation Radiation Pattern")
plt.xlabel("Angle (Degrees)")
plt.ylabel("Gain")
plt.grid(True)
plt.legend()

plt.show()