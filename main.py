import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
# Wczytanie danych
data = pd.read_csv('data - FractureCase.csv')
print("Dane wczytano poprawnie.")

# Wybór zmiennych objaśniających (X)
X = data[['Volume', 'Players', 'Events']]

y = data['Price'].str.replace('zł', '').str.replace(',', '.').astype(float)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standaryzacja zmiennych objaśniających
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

# Standaryzacja wartości objaśnianej (tylko dla zbioru treningowego)
scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))

# Zdefiniuj model sieci neuronowej
inputs = tf.keras.layers.Input(shape=(3,))
x = tf.keras.layers.Dense(32, activation='relu')(inputs)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Dense(16, activation='relu')(x)
x = tf.keras.layers.Dropout(0.1)(x)
outputs = tf.keras.layers.Dense(1)(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)

# Kompilacja modelu
# = 'RMSprop'
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
loss_function = 'mean_squared_error'
metrics = ['mean_absolute_error']
model.compile(optimizer=optimizer, loss=loss_function, metrics=metrics)

# Trenowanie modelu
model.fit(X_train, y_train, epochs=100, batch_size=10, validation_data=(X_test, y_test))
# Wybierz dane z ostatniego dnia
last_day_data = data.iloc[-1]

# Przygotuj dane wejściowe dla predykcji
new_data = pd.DataFrame({
    'Volume': [last_day_data['Volume']],
    'Players': [last_day_data['Players']],
    'Events': [last_day_data['Events']]
})

# Przekształć nowe dane za pomocą skalera
new_data_scaled = scaler_X.transform(new_data)

# Dokonaj predykcji na podstawie nowych danych
predictions = model.predict(new_data_scaled)

# Odwróć skalowanie przewidywanych cen
predicted_prices = scaler_y.inverse_transform(predictions)

# Wypisz przewidywane ceny
for price in predicted_prices:
    print(f"Przewidywana cena: {price[0]:.2f} zł")

