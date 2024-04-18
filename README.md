
# Wizualizacja krzywych uczenia się
plt.figure(figsize=(12, 6))

# Krzywa straty
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Strata treningowa')
plt.plot(history.history['val_loss'], label='Strata walidacyjna')
plt.title('Krzywa straty')
plt.xlabel('Epoka')
plt.ylabel('Strata')
plt.legend()

# Krzywa metryki mean_absolute_error
plt.subplot(1, 2, 2)
plt.plot(history.history['mean_absolute_error'], label='MAE treningowe')
plt.plot(history.history['val_mean_absolute_error'], label='MAE walidacyjne')
plt.title('Krzywa mean_absolute_error')
plt.xlabel('Epoka')
plt.ylabel('MAE')
plt.legend()

plt.tight_layout()
plt.show()