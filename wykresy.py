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
def exp_func(x, a, b):
    return a * np.exp(b * x)


dtF = pd.read_csv(
    "datasets/RED_LED_PD_mj/RED_LED_PD_mj/LVJPD_RED_1cm_BenchVue_30112022G1/Ipd_Popt6_UpdF_30112022 2022-11-30 12-02-00 0 2022-11-30 12-42-42 0.csv",
    sep=",",
    skiprows=5,
)

dtR = pd.read_csv(
    "datasets/RED_LED_PD_mj/RED_LED_PD_mj/LVJPD_RED_1cm_BenchVue_30112022G1/Ipd_Popt6_UpdR_30112022 2022-11-30 12-02-00 0 2022-11-30 12-03-12 0.csv",
    sep=",",
    skiprows=6,
)

# dtR['3 - Get Measurement Value'] = dtR['3 - Get Measurement Value']*(-1)

cF = [
    "2 - Get Measurement Value",
    "Get Measurement Value",
    "1 - Sweep CH2 Voltage Setting",
    "1 - Get CH2 Current Measurement",
]
cR = [
    "2 - Get Measurement Value",
    "3 - Get Measurement Value",
    "1 - Sweep CH2 Voltage Setting",
    "1 - Get CH2 Current Measurement",
]

muFR = pys.SweepPlotForwardReverse(dtF, dtR, cF, cR, 0, 27, exp_func)

plt.suptitle("Photodiode U / I characterictics")
plt.title("Variable red LED illumination")

plt.show()
