import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Conv1D,
    BatchNormalization,
    ReLU,
    Add,
    GlobalAveragePooling1D,
    Dense,
    Dropout
)
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)

print("Loading Dataset...")

# =====================================
# LOAD DATA
# =====================================

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

# Save scaler
import joblib
joblib.dump(
    scaler,
    "resnet_scaler.pkl"
)

# =====================================
# RESHAPE FOR CNN
# =====================================

X_train = X_train.reshape(
    X_train.shape[0],
    181,
    1
)

X_val = X_val.reshape(
    X_val.shape[0],
    181,
    1
)

print("Reshaped X_train:", X_train.shape)
print("Reshaped X_val:", X_val.shape)

# =====================================
# RESIDUAL BLOCK
# =====================================

def residual_block(x, filters):

    shortcut = x

    x = Conv1D(
        filters,
        kernel_size=3,
        padding="same"
    )(x)

    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv1D(
        filters,
        kernel_size=3,
        padding="same"
    )(x)

    x = BatchNormalization()(x)

    if shortcut.shape[-1] != filters:

        shortcut = Conv1D(
            filters,
            kernel_size=1,
            padding="same"
        )(shortcut)

    x = Add()([
        x,
        shortcut
    ])

    x = ReLU()(x)

    return x

# =====================================
# RESNET MODEL
# =====================================

inputs = Input(
    shape=(181, 1)
)

x = Conv1D(
    64,
    kernel_size=5,
    padding="same"
)(inputs)

x = BatchNormalization()(x)
x = ReLU()(x)

x = residual_block(x, 64)
x = residual_block(x, 64)

x = residual_block(x, 128)
x = residual_block(x, 128)

x = GlobalAveragePooling1D()(x)

x = Dense(
    256,
    activation="relu"
)(x)

x = Dropout(0.2)(x)

x = Dense(
    128,
    activation="relu"
)(x)

x = Dense(
    64,
    activation="relu"
)(x)

outputs = Dense(
    8,
    activation="linear"
)(x)

model = Model(
    inputs,
    outputs
)

model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)

model.summary()

# =====================================
# CALLBACKS
# =====================================

callbacks = [

    ModelCheckpoint(
        "best_resnet_model.keras",
        save_best_only=True,
        monitor="val_loss"
    ),

    EarlyStopping(
        monitor="val_loss",
        patience=50,
        restore_best_weights=True
    ),

    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=20,
        verbose=1
    )
]

# =====================================
# TRAINING
# =====================================

history = model.fit(
    X_train,
    y_train,
    validation_data=(
        X_val,
        y_val
    ),
    epochs=500,
    batch_size=64,
    callbacks=callbacks,
    verbose=1
)

# =====================================
# SAVE HISTORY
# =====================================

history_df = pd.DataFrame(
    history.history
)

history_df.to_csv(
    "resnet_training_history.csv",
    index=False
)

# =====================================
# PLOT LOSS
# =====================================

plt.figure(figsize=(8,5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("ResNet Training History")

plt.legend()
plt.grid(True)

plt.savefig(
    "resnet_loss_curve.png",
    dpi=300
)

plt.show()

# =====================================
# FINAL EVALUATION
# =====================================

loss, mae = model.evaluate(
    X_val,
    y_val,
    verbose=0
)

accuracy = (
    1 - mae
) * 100

print("\nValidation Loss :", loss)
print("Validation MAE  :", mae)
print(f"Accuracy : {accuracy:.2f}%")

print("\nTraining Complete")