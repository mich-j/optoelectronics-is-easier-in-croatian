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
    files = [
        'Uout_aps_redu5vf1khz1_2.csv',      
        'Ureset_aps_redu5vf1khz1_1.csv',    
        'Vout_apsgreenu5vf1khz3_2.csv',  
        'Vout_apsgreen_u5vf1khz2_2.csv',  
        'Vout_apsrredv5vf1khz2x4_2.csv',
        'Vreset_apsgreenu5vf1khz2_1.csv',
        'Vreset_apsgreenu5vf1khz3_1.csv',
        'Vreset_apsred_v5vf1khz2x2_1.csv',
        'Vreset_apsblue_u5vf1khz6_1.csv',
        'Ureset_apsgreen_v5vf1khz2x1_1.csv'
    ]
    print(onlyfiles)
    pl = list()
    fig, axVoltage = plt.subplots()
    axVoltage.set_ylabel("Voltage [V]")
    axReset = axVoltage.twinx()
    axReset.set_ylabel('Reset voltage [V]')
    for f in onlyfiles:
        if f in files:
            continue
        dataset = pd.read_csv(directoryPath + f, sep=',', skiprows=[1])
        col = dataset.columns
        dataset[col[1]] /= 10
        for i in range(1, len(dataset[col[0]])):
            dataset[col[0]][i] = dataset[col[0]][i] - dataset[col[0]][0]
            dataset[col[0]][i] *= 10**6
        dataset[col[0]][0] = 0
        color = 'black'
        colors = ['red', 'green', 'blue']
        if('reset' in f):
            
            pl.append(axReset.plot(dataset[col[0]], dataset[col[1]], label = 'RESET', color = color))
        else:
            for c in colors:
                if(c in f):
                    color = c
            pl.append(axVoltage.plot(dataset[col[0]], dataset[col[1]], label = f, color = color))
        # PlotVoltageTime(dataset=dataset, title=f)
        # break
    lns = pl[0]
    for i in range(1, len(pl)):
        lns +=pl[i]
    labs = [l.get_label() for l in lns]
    axVoltage.legend(lns, labs)
    axVoltage.set_ylabel("Voltage [V]")
    axVoltage.grid()
    plt.yscale = 'log'
    plt.title('title')

    # fig.tight_layout()
    plt.show()
PrintDataFromDirectoryVoltageTime('datasets/APS/')
# files = [
# 'Vout_apsblue_u5vf1khz6_2.csv',
# 'Vout_apsrredv5vf1khz2x4_2.csv',
# 'Uout_apsgreen_v5vf1khz2x1_2.csv']

# directoryPath = 'datasets/APS/'
# # data = list()
# # for f in files:
# #     dataset = pd.read_csv(directoryPath + f, sep=',', skiprows=[1])
# #     col = dataset.columns
# #     data.append(dataset[col[0]])
# #     data.append(dataset[col[1]])
# fig, axVoltage = plt.subplots()

# pl = list()
# ds = pd.DataFrame
# i = 0
# for f in files:
#     dataset = pd.read_csv(directoryPath + f, sep=',', skiprows=[1])
#     col = dataset.columns
#     dataset[col[1]] /= 10
#     for i in range(1, len(dataset[col[0]])):
#         dataset[col[0]][i] = dataset[col[0]][i] - dataset[col[0]][0]
#         dataset[col[0]][i] *= 10**6
#     dataset[col[0]][0] = 0
#     color = 'black'
#     colors = ['red', 'green', 'blue']
#     for c in colors:
#         if(c in f):
#             color = c
#     pl.append(axVoltage.plot(dataset[col[0]], dataset[col[1]], label = f, color = color))

    
#     print(dataset)
# #     if(i == 0):
# #         ds= dataset
# #         i+=1
# #     else:
# #         pd.DataFrame.merge(dataset, left_on='lkey', right_on='rkey')
# # ds.to_csv('foo.csv')
# lns = pl[0]
# for i in range(1, len(pl)):
#     lns +=pl[i]
# labs = [l.get_label() for l in lns]
# axVoltage.legend(lns, labs)
# axVoltage.set_ylabel("Voltage [V]")
# axVoltage.grid()
# plt.yscale = 'log'
# plt.title('title')

# # fig.tight_layout()
# plt.show()
# plt.savefig()
# Uout_apsgreen_v5vf1khz2x1_2.csv  
# # Uout_aps_redu5vf1khz1_2.csv      
# Ureset_apsgreen_v5vf1khz2x1_1.csv
# # Ureset_aps_redu5vf1khz1_1.csv    
# Vout_apsblue_u5vf1khz6_2.csv     
# # Vout_apsgreenu5vf1khz3_2.csv     
# # Vout_apsgreen_u5vf1khz2_2.csv    
# Vout_apsred_v5vf1khz2x2_2.csv
# # Vout_apsrredv5vf1khz2x4_2.csv
# Vreset_apsblue_u5vf1khz6_1.csv
# # Vreset_apsgreenu5vf1khz2_1.csv
# # Vreset_apsgreenu5vf1khz3_1.csv
# Vreset_apsredv5vf1khz2x4_1.csv
# # Vreset_apsred_v5vf1khz2x2_1.csv

# def exp_func(x, a, b):
#     return a * np.exp(b * x)


# dtF = pd.read_csv(
#     "datasets/RED_LED_PD_mj/RED_LED_PD_mj/LVJPD_RED_1cm_BenchVue_30112022G1/Ipd_Popt6_UpdF_30112022 2022-11-30 12-02-00 0 2022-11-30 12-42-42 0.csv",
#     sep=",",
#     skiprows=5,
# )

# dtR = pd.read_csv(
#     "datasets/RED_LED_PD_mj/RED_LED_PD_mj/LVJPD_RED_1cm_BenchVue_30112022G1/Ipd_Popt6_UpdR_30112022 2022-11-30 12-02-00 0 2022-11-30 12-03-12 0.csv",
#     sep=",",
#     skiprows=6,
# )

# # dtR['3 - Get Measurement Value'] = dtR['3 - Get Measurement Value']*(-1)

# cF = [
#     "2 - Get Measurement Value",
#     "Get Measurement Value",
#     "1 - Sweep CH2 Voltage Setting",
#     "1 - Get CH2 Current Measurement",
# ]
# cR = [
#     "2 - Get Measurement Value",
#     "3 - Get Measurement Value",
#     "1 - Sweep CH2 Voltage Setting",
#     "1 - Get CH2 Current Measurement",
# ]

# muFR = pys.SweepPlotForwardReverse(dtF, dtR, cF, cR, 0, 27, exp_func)

# plt.suptitle("Photodiode U / I characterictics")
# plt.title("Variable red LED illumination")

# plt.show()
