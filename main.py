import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('data - FractureCase.csv')
print("Dane wczytano poprawnie.")

X = data[['Volume', 'Players', 'Events']]

y = data['Price'].str.replace('zł', '').str.replace(',', '.').astype(float)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))

inputs = tf.keras.layers.Input(shape=(3,))
x = tf.keras.layers.Dense(32, activation='relu')(inputs)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Dense(16, activation='relu')(x)
x = tf.keras.layers.Dropout(0.1)(x)
outputs = tf.keras.layers.Dense(1)(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
loss_function = 'mean_squared_error'
metrics = ['mean_absolute_error']
model.compile(optimizer=optimizer, loss=loss_function, metrics=metrics)


model.fit(X_train, y_train, epochs=100, batch_size=10, validation_data=(X_test, y_test))
last_day_data = data.iloc[-1]

new_data = pd.DataFrame({
    'Volume': [last_day_data['Volume']],
    'Players': [last_day_data['Players']],
    'Events': [last_day_data['Events']]
})

new_data_scaled = scaler_X.transform(new_data)


predictions = model.predict(new_data_scaled)

predicted_prices = scaler_y.inverse_transform(predictions)

for price in predicted_prices:
    print(f"Przewidywana cena: {price[0]:.2f} zł")

