#CS2PricePredcitor


Przygotowanie danych:
//a. Podziel zbiór danych na zbiór treningowy i testowy za pomocą train_test_split ze sklearn.
//b. Zidentyfikuj zmienne objaśniające (X) i zmienną objaśnianą (y) w zbiorze danych.
//c. Zastosuj standaryzację lub normalizację do zmiennych objaśniających za pomocą StandardScaler ze sklearn.

Zdefiniuj architekturę sieci neuronowej:
//a. Zaimportuj moduły TensorFlow, takie jak keras lub tf.keras.
b. Zdefiniuj warstwę wejściową sieci, określając liczbę neuronów równą liczbie zmiennych objaśniających.
c. Dodaj jedną lub więcej warstw ukrytych, określając liczbę neuronów i funkcję aktywacji dla każdej warstwy.
d. Zdefiniuj warstwę wyjściową, określając liczbę neuronów równą liczbie klas (dla problemów klasyfikacji) lub 1 (dla problemów regresji) oraz odpowiednią funkcję aktywacji.

Skompiluj model:
//a. Zdefiniuj optimizer, funkcję straty i metryki do oceny modelu.
//b. Skompiluj model, przekazując optimizer, funkcję straty i metryki.

Trenuj model:
a. Wywołaj metodę fit na skompilowanym modelu, przekazując zbiór treningowy (X_train, y_train) oraz opcjonalnie zbiór walidacyjny.
b. Możesz również określić liczbę epok, rozmiar partii danych (batch size) i inne parametry treningu.

Oceń model:
a. Oceń wydajność modelu na zbiorze testowym (X_test, y_test) za pomocą metody evaluate lub innych metryk ewaluacyjnych.
b. Możesz również przewidzieć wyniki dla nowych danych za pomocą metody predict.

Opcjonalnie, dostosuj model:
a. Jeśli wyniki nie są zadowalające, możesz dostosować architekturę sieci, zmienić hiperparametry lub zastosować techniki takie jak regularyzacja, dropout lub wczesne zatrzymanie (early stopping).
b. Powtórz kroki 3-5 z nową konfiguracją modelu.