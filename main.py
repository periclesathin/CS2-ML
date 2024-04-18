import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Wczytanie danych
data = pd.read_csv('data - FractureCase.csv')
print(data.head())

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
