import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau

print("Loading Dataset...")

train_df = pd.read_csv("../data/train_split.csv")
val_df = pd.read_csv("../data/validation_split.csv")

# INPUT = phases
X_train = train_df.iloc[:, 181:].values
X_val = val_df.iloc[:, 181:].values

# OUTPUT = radiation pattern
y_train = train_df.iloc[:, :181].values
y_val = val_df.iloc[:, :181].values

print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("X_val:", X_val.shape)
print("y_val:", y_val.shape)

x_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()

X_train = x_scaler.fit_transform(X_train)
X_val = x_scaler.transform(X_val)

y_train = y_scaler.fit_transform(y_train)
y_val = y_scaler.transform(y_val)

joblib.dump(
    x_scaler,
    "inverse_input_scaler.pkl"
)

joblib.dump(
    y_scaler,
    "inverse_output_scaler.pkl"
)

model = Sequential()

model.add(
    Dense(
        256,
        activation="relu",
        input_shape=(8,)
    )
)

model.add(
    Dense(
        512,
        activation="relu"
    )
)

model.add(
    Dense(
        512,
        activation="relu"
    )
)

model.add(
    Dense(
        256,
        activation="relu"
    )
)

model.add(
    Dense(
        181,
        activation="linear"
    )
)

model.summary()

model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=20,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=10
)

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=500,
    batch_size=64,
    callbacks=[
        early_stop,
        reduce_lr
    ]
)

model.save(
    "best_inverse_model.keras"
)

print("\nInverse Model Saved")