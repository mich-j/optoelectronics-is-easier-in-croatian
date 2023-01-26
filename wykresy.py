import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
from scipy.signal import savgol_filter

from os import listdir
from os.path import isfile, join

import constants as const
import python_scripts as pys


def PlotVoltageTime(dataset:pd.DataFrame, title = ' '):
    col = dataset.columns
    # print(dataset[col[0]][0])
    for i in range(1, len(dataset[col[0]])):
      dataset[col[0]][i] = dataset[col[0]][i] - dataset[col[0]][0]
      dataset[col[0]][i] *= 10**6
    dataset[col[0]][0] = 0
    plt.figure()
    plt.plot(dataset[col[0]], dataset[col[1]])
    plt.ylabel("Voltage [V]")
    plt.xlabel("Time [us]")
    plt.title(title)
    plt.grid()
    plt.show()

def PrintDataFromDirectoryVoltageTime(directoryPath: str, filterValues = True):
    onlyfiles = [
        f
        for f in listdir(directoryPath)
        if isfile(join(directoryPath, f)) and f[-4:] == ".csv"
    ]
    print(onlyfiles)
    for f in onlyfiles:
        dataset = pd.read_csv(directoryPath + f, sep=',', skiprows=[1])
        PlotVoltageTime(dataset=dataset, title=f)
        break
PrintDataFromDirectoryVoltageTime('datasets/APS/')