import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from train import load_datas, normalize
sns.set(style="ticks")

options = """Please, type one of theses options to run:
             - [1] scatter plot the datas 
             - [2] visualize the linear function 
             - [3] plot the loss 
             - [q] quit 
          """

def scatter_plot():
    df = pd.DataFrame()
    df['X'], df['y'] = load_datas("data.csv")
    sns.lmplot('X', 'y', df, fit_reg=False)
    plt.title('Datas Scatter Plot')
    plt.xlabel('km')
    plt.ylabel('price')
    plt.show()

def linear_plot(scatter=False):
    X_datas, y_datas = load_datas("data.csv")
    w = np.loadtxt('weights.csv')
    X = np.array(range(60000, 250000, 500))
    X_norm = normalize(X)
    y = w[0] + w[1] * X_norm
    print(X)
    print(y)
    if scatter is True:
        plt.scatter(X_datas, y_datas)
    plt.plot(X, y)
    plt.show()


if __name__ == '__main__':
    loop = True
    # while loop:
    #     option = input(options)
    #     if option == 'q':
    #         loop = False
    #     elif option == '1':
    #         scatter_plot()

    linear_plot(scatter=True)
