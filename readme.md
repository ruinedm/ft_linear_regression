# 🚗 Simple Linear Regression: Car Price Estimator

This project implements a simple linear regression model to estimate the price of a car based on its mileage. It includes:

- A **training program** that reads a dataset and learns from it
- A **prediction program** that estimates a price given a mileage
- **Bonus features** like graphing and precision evaluation

---

## 🧠 Concept

The model uses the hypothesis function:

estimatePrice(mileage) = θ₀ + θ₁ × mileage

The training algorithm updates `θ₀` and `θ₁` using gradient descent.


## 🚀 Getting Started
- **Initialization**: pip install -r requirements.txt (Use a virtual env)
- **Training**: python3 train.py [dataset.csv]
- **Predicting**: python3 predict.py
**Save your dataset in the root of your directory under the name data.csv**
Your dataset should be a CSV with the following structure:
mileage,price
24000,20000
46000,12000
...


## 📁 File Structure
train.py            # Training script
predict.py          # Prediction script
data.csv            # Dataset file (mileage,price)
model_params.json   # Saved model parameters (outputs of train.py)
visualize.py        # Graphs the dataset + regression line (TO-DO)
evaluate.py         # Evaluates model precision (TO-DO)
README.md           # This file



## 🏅 Bonus Features (In Progress)

Feature	Status
Plotting data points	...
Plotting the regression line	...
Calculating model precision (MSE)	...

Made with ❤️ for learning and experimentation.
