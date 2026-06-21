import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

from tensorflow import keras
from tensorflow.keras.layers import (
    Dense,
    BatchNormalization,
    Dropout
)

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)

print("Loading Dataset...")

train_df = pd.read_csv(
    "../data/train_split.csv"
)

val_df = pd.read_csv(
    "../data/validation_split.csv"
)

# INPUTS
X_train = train_df.iloc[:, :181].values
X_val = val_df.iloc[:, :181].values

# OUTPUTS
y_train = train_df.iloc[:, 181:].values
y_val = val_df.iloc[:, 181:].values

print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("X_val:", X_val.shape)
print("y_val:", y_val.shape)

# NORMALIZE INPUTS
x_scaler = MinMaxScaler()

X_train = x_scaler.fit_transform(
    X_train
)

X_val = x_scaler.transform(
    X_val
)

# MODEL
model = keras.Sequential([

    Dense(
        256,
        activation='relu',
        input_shape=(181,)
    ),

    BatchNormalization(),
    Dropout(0.20),

    Dense(
        128,
        activation='relu'
    ),

    BatchNormalization(),
    Dropout(0.20),

    Dense(
        64,
        activation='relu'
    ),

    Dense(
        32,
        activation='relu'
    ),

    Dense(
        8,
        activation='linear'
    )

])

model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

model.summary()

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=50,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=20,
    verbose=1
)

checkpoint = ModelCheckpoint(
    "best_dnn_v2.keras",
    save_best_only=True,
    monitor='val_loss'
)

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=500,
    batch_size=100,
    callbacks=[
        early_stop,
        reduce_lr,
        checkpoint
    ],
    verbose=1
)

loss, mae = model.evaluate(
    X_val,
    y_val,
    verbose=0
)

print("\nValidation Loss:", loss)
print("Validation MAE :", mae)

# SAVE HISTORY
history_df = pd.DataFrame(
    history.history
)

history_df.to_csv(
    "training_history.csv",
    index=False
)

# LOSS CURVE
plt.figure(figsize=(10,5))

plt.plot(
    history.history['loss'],
    label='Training Loss'
)

plt.plot(
    history.history['val_loss'],
    label='Validation Loss'
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.title(
    "DNN V2 Loss Curve"
)

plt.legend()
plt.grid(True)

plt.savefig(
    "../plots/dnn/loss_curve.png"
)

plt.show()

print("\nTraining Complete")