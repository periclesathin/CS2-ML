import tensorflow as tf
import pandas as pd

# Wczytywanie danych
data = pd.read_csv('data.xlsx')

# Przygotowanie danych
X = data[['Data', 'Players', 'Events', 'Volume']]
y = data['Price']

# Konwersja daty na format liczbowy
X['Data'] = pd.to_datetime(X['Data']).view('int64') // 10**9

# Konwersja kolumny 'Events' na wartości 0/1
X['Events'] = X['Events'].apply(lambda x: 1 if x else 0)

# Podzielenie danych na zbiór treningowy i testowy
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizacja danych
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Budowa modelu
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])

# Kompilacja modelu
model.compile(optimizer='adam', loss='mse')

# Trenowanie modelu
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

# Przewidywanie cen
future_data = [[1679616000, 1519457, 1, 100000]]  # Przykładowe dane dla przyszłej daty
future_data = scaler.transform(future_data)
predicted_price = model.predict(future_data)[0][0]
print(f"Przewidywana cena: {predicted_price}")