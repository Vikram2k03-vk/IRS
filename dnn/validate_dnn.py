import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler


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

    diff = np.abs(actual - predicted)

    return np.minimum(
        diff,
        360 - diff
    )


print("Loading Model...")

model = keras.models.load_model(
    "best_dnn_v2.keras"
)

print("Loading Validation Dataset...")

val_df = pd.read_csv(
    "../data/validation_split.csv"
)

# INPUTS
X_val = val_df.iloc[:, :181].values

# TARGETS
y_val = val_df.iloc[:, 181:].values

# NORMALIZE INPUTS
x_scaler = MinMaxScaler()

train_df = pd.read_csv(
    "../data/train_split.csv"
)

X_train = train_df.iloc[:, :181].values

x_scaler.fit(
    X_train
)

X_val = x_scaler.transform(
    X_val
)

print("Predicting...")

predictions = model.predict(
    X_val
)

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

print(
    f"\nAverage Phase Error: {mean_error:.2f} degrees"
)

print(
    f"Maximum Phase Error: {max_error:.2f} degrees"
)

accuracy = (
    1 -
    (mean_error / 180)
) * 100

print(
    f"Accuracy: {accuracy:.2f}%"
)

# PLOT
plt.figure(figsize=(10,6))

plt.hist(
    errors.flatten(),
    bins=30
)

plt.xlabel(
    "Phase Error (Degrees)"
)

plt.ylabel(
    "Count"
)

plt.title(
    "Phase Error Distribution"
)

plt.grid(True)

plt.savefig(
    "../plots/dnn/phase_error_distribution.png"
)

plt.show()

# ACTUAL VS PREDICTED

plt.figure(figsize=(10,6))

plt.scatter(
    actual_phases[:,1],
    predicted_phases[:,1]
)

plt.xlabel(
    "Actual Phase"
)

plt.ylabel(
    "Predicted Phase"
)

plt.title(
    "Actual vs Predicted Phase (Antenna 2)"
)

plt.grid(True)

plt.savefig(
    "../plots/dnn/actual_vs_predicted.png"
)

plt.show()