"""
This program predict the estimated age given mileage.
It uses the weight's file to compute the result.
"""
import os.path
import numpy as np
from train import normalize

def predict(mileage):
    """Predict the price given the mileage and the weights."""
    if os.path.exists('weights.csv'):
        print("OUIIIIIII")
        w = np.loadtxt('weights.csv')
    else:
        w = np.zeros(2)
    mileage = normalize(mileage)
    return w[0] + w[1] * float(mileage)

if __name__ == '__main__':
    while True:
        u_mileage = input("Please, indicate mileage to predict a price: ")
        prediction = predict(u_mileage)
        print("Predicted price for mileage {} is {}.".format(u_mileage, prediction))
