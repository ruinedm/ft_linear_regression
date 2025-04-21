
import numpy as np
import matplotlib.pyplot as plt

data_array = np.loadtxt('data.csv', delimiter=',', skiprows=1)
mileage_data = data_array[:, 0] #inputs
price_data = data_array[:, 1] #targets


# θ0 (theta0): The base price (y-intercept)
# θ1 (theta1): The rate at which price changes with mileage (slope)
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
        # should my code firstly analyze and clean data and make learning_rate init dynamically or what??
        self.__inputs = inputs
        self.__targets = targets
        self.__inputs_name = inputs_name
        self.__targets_name = targets_name
        self.__weight = weight_init
        self.__bias = bias_init
        self.__learning_rate = 0.001
        self.__max_epoch = 0
        

    def __predict(self, x):
        return self.__weight * x + self.__bias
    
    # def train(self):
    #     while True:
            

    def plot_data(self, mode: str = "both"):
        plt.scatter(self.__inputs, self.__targets, color='red', marker='o')
        plt.xlabel(self.__inputs_name)
        plt.ylabel(self.__targets_name)
        plt.title(f"{self.__inputs_name} vs {self.__targets_name}")
        plt.grid(True)
        mode = mode.lower()
        if mode == "show":
            plt.show()
        elif mode == "save":
            plt.savefig("plot.png")
            print("Saved plot as plot.png")
        elif mode == "both":
            plt.savefig("plot.png")
            print("Saved plot as plot.png")
            plt.show()
        else:
            raise ValueError("Invalid mode")
    


if __name__ == "__main__":
    data_file = "data.csv"
    data_array = np.loadtxt('data.csv', delimiter=',', skiprows=1)
    mileage_data = data_array[:, 0] #inputs
    price_data = data_array[:, 1] #targets

    model = LinearRegression(mileage_data, price_data, 0, 0, "Mileage", "Price")
    model.plot_data()