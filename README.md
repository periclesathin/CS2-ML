# Predicting Item Prices in CS2
## Key Features:
📊 Data Loading & Preprocessing:
The dataset is read from a .csv file, relevant features are selected, and data is split into training and testing sets.
🔄 Data Standardization:
Both features (X) and target values (y) are standardized to improve model training performance.
🧠 Neural Network Model:
A simple feedforward neural network is defined using Keras with dropout layers to prevent overfitting.
🏋️‍♂️ Model Training:
The model is trained over 100 epochs with mean squared error as the loss function and mean absolute error (MAE) as a metric.
📈 Learning Curves Visualization:
Plots are generated to track training vs validation loss and MAE over epochs.
🤖 Prediction on New Data:
The model makes predictions for the most recent row in the dataset (simulating a real-time forecast). The prediction is rescaled back to the original price format.

## Tools & Technologies
Python 
TensorFlow / Keras 
Scikit-learn 
Pandas / NumPy 
Matplotlib 
