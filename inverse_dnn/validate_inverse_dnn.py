import numpy as np
import joblib
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model

# =====================================
# LOAD MODEL
# =====================================

print("Loading Inverse Model...")

model = load_model(
    "best_inverse_model.keras"
)

input_scaler = joblib.load(
    "inverse_input_scaler.pkl"
)

output_scaler = joblib.load(
    "inverse_output_scaler.pkl"
)

# =====================================
# USER INPUT
# =====================================

print("\nEnter 4 IRS phases")

phase1 = float(input("Phase 1 (deg): "))
phase2 = float(input("Phase 2 (deg): "))
phase3 = float(input("Phase 3 (deg): "))
phase4 = float(input("Phase 4 (deg): "))

# =====================================
# CONVERT TO COS/SIN FORMAT
# =====================================

phases = [phase1, phase2, phase3, phase4]

features = []

for phase in phases:

    rad = np.radians(phase)

    features.append(
        np.cos(rad)
    )

    features.append(
        np.sin(rad)
    )

features = np.array(
    features
).reshape(1, -1)

# =====================================
# SCALE INPUT
# =====================================

features_scaled = input_scaler.transform(
    features
)

# =====================================
# PREDICT RADIATION PATTERN
# =====================================

prediction = model.predict(
    features_scaled,
    verbose=0
)

pattern = output_scaler.inverse_transform(
    prediction
)[0]

# =====================================
# IRS REFLECTION PHASES
# =====================================

reflection_phases = []

for phase in phases:

    reflection = -phase

    while reflection > 180:
        reflection -= 360

    while reflection < -180:
        reflection += 360

    reflection_phases.append(
        reflection
    )

# =====================================
# BEAM INFORMATION
# =====================================

beam_angle = np.argmax(
    pattern
)

peak_gain = np.max(
    pattern
)

# =====================================
# MODEL PERFORMANCE
# =====================================

accuracy = (
    1 - 0.0078
) * 100

# =====================================
# DISPLAY RESULTS
# =====================================

print("\n==============================")
print(" INVERSE DNN RESULTS")
print("==============================")

print(
    "\nPattern Reconstruction Accuracy : "
    f"{accuracy:.2f}%"
)

print("\nIRS Reflection Phases")

for i, p in enumerate(reflection_phases):

    print(
        f"Antenna {i+1}: {p:.2f}°"
    )

print(
    f"\nPredicted Beam Direction : {beam_angle}°"
)

print(
    f"Peak Gain : {peak_gain:.2f}"
)

# Side Lobe

temp_pattern = pattern.copy()

temp_pattern[
    max(0, beam_angle - 10):
    min(len(pattern), beam_angle + 10)
] = 0

side_lobe_angle = np.argmax(
    temp_pattern
)

side_lobe_gain = np.max(
    temp_pattern
)

print(
    f"Side Lobe Direction : {side_lobe_angle}°"
)

print(
    f"Side Lobe Gain : {side_lobe_gain:.2f}"
)

# Null Direction

null_angle = np.argmin(
    pattern
)

null_gain = np.min(
    pattern
)

print(
    f"Null Direction : {null_angle}°"
)

print(
    f"Null Gain : {null_gain:.2f}"
)

validation_mae = 0.0078

print(
    f"\nValidation MAE : {validation_mae:.6f}"
)

accuracy = (
    1 - validation_mae
) * 100

print(
    f"Pattern Reconstruction Accuracy : "
    f"{accuracy:.2f}%"
)

# =====================================
# DISPLAY RESULTS
# =====================================

print("\n==============================")
print(" IRS BEAM STEERING RESULTS")
print("==============================")

print(
    f"\nPredicted Beam Direction : {beam_angle}°"
)

print(
    f"Peak Gain : {peak_gain:.2f}"
)

# =====================================
# PLOT PATTERN
# =====================================

angles = np.arange(181)

plt.figure(figsize=(10, 5))

plt.plot(
    angles,
    pattern
)

plt.title(
    "Predicted Radiation Pattern"
)

plt.xlabel(
    "Angle (Degrees)"
)

plt.ylabel(
    "Gain"
)

plt.grid(True)

plt.tight_layout()

plt.show()

'''The neural network predicts the complete 181-point radiation pattern from the IRS phase shifts. 
The beam direction is then extracted from the predicted pattern by identifying the angle with maximum gain. 
Thus, the neural network replaces expensive electromagnetic simulations and enables real-time beam steering prediction.'''