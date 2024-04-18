import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Wczytanie danych
data = pd.read_csv('data - FractureCase.csv')
print(data.head())

