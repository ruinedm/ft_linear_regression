
import numpy as np

data_array = np.loadtxt('data.csv', delimiter=',', skiprows=1)
mileage_data = data_array[:, 0] #inputs
price_data = data_array[:, 1] #targets

learning_rate = 0.1
tetha0 = 0
tetha1 = 0
epochs = 1000
# θ0 (theta0): The base price (y-intercept)
# θ1 (theta1): The rate at which price changes with mileage (slope)
def predict_price(mileage):
    return tetha0 + (tetha1 * mileage)

for epoch in range(epochs):
    predictions = [predict_price(i) for i in mileage_data]
    costs = [target - prediction for prediction, target in zip(predictions, price_data)]
    tetha0 += learning_rate * sum(costs) / len(mileage_data)
    tetha1 += learning_rate * sum(m * c for m, c in zip(mileage_data, costs)) / len(mileage_data)
    print(f"Epoch: {epoch} // Tetha0: {tetha0} // Tetha1: {tetha1}")

print(predict_price(mileage_data[0]))