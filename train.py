

import numpy as np
import matplotlib.pyplot as plt
import json

data_array = np.loadtxt('data.csv', delimiter=',', skiprows=1)
mileage_data = data_array[:, 0] #inputs
price_data = data_array[:, 1] #targets

class LinearRegression:
    def __init__(self, inputs, targets, weight_init: int = 0, bias_init: int = 0, inputs_name: str ="Inputs", targets_name: str ="Targets"):
        if not isinstance(inputs, np.ndarray):
            raise ValueError(f"{inputs_name} must be a NumPy array.")
        if not isinstance(targets, np.ndarray):
            raise ValueError(f"{targets_name} must be a NumPy array.")

        if inputs.ndim != 1:
            raise ValueError(f"{inputs_name} must be a 1D NumPy array.")
        if targets.ndim != 1:
            raise ValueError(f"{targets_name} must be a 1D NumPy array.")
    
        if inputs.shape[0] != targets.shape[0]:
            raise ValueError(f"{inputs_name} and {targets_name} must have the same number of samples.")
        self.__inputs = np.array([(x - inputs.mean()) / inputs.std() for x in inputs])
        self.__original_mean = inputs.mean()
        self.__original_std = inputs.std()
        self.__learning_done = False
        self.__m = len(inputs)
        self.__targets = targets
        self.__weight = weight_init
        self.__bias = bias_init
        self.__learning_rate = 0.01
        self.__max_epoch = 500


    def __predict_error(self, x):
        return self.__weight * x + self.__bias
    
    def train(self):
        for epoch in range(self.__max_epoch):
            errors = self.__predict_error(self.__inputs) - self.__targets
            self.__bias -= self.__learning_rate * np.sum(errors) / self.__m
            self.__weight -= self.__learning_rate * np.sum(errors * self.__inputs) / self.__m
        self.__learning_done = True

    def get_final_parameters(self):
        if not self.__learning_done:
            raise Exception("Trying to get final parameters before training")
        theta1 = self.__weight / self.__original_std
        theta0 = self.__bias - self.__original_mean * self.__weight / self.__original_std
        return theta0, theta1


if __name__ == "__main__":
    try:
        data_array = np.loadtxt('data.csv', delimiter=',', skiprows=1)
        mileage_data = data_array[:, 0]
        price_data = data_array[:, 1]
    except Exception as e:
        print(f"Error while loading data: {e}")
    model = LinearRegression(mileage_data, price_data, 0, 0, "Mileage", "Price")
    model.train()
    theta0, theta1 = model.get_final_parameters()
    data = {
        "theta0" : theta0,
        "theta1" : theta1
    }
    with open("model_params.json", "w") as file:
        json.dump(data, file)
