import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
from scipy.signal import savgol_filter
from decimal import Decimal
import os.path
from os import path

def CheckDataset(
  columns: pd.Index
  ):
  order = pd.Index(['x-axis', '1', '3', 'F1'])
  if(columns.equals(order) == True):
    return 'ok'
  else:
    print('Wrong column labels, should be:')
    print(order)
    print('In the file there is:')
    print(columns)
    return 'wrong'



def SweepPlot(
    data: pd.DataFrame,
    xName: str,
    yName: str,
    SweepColumnName: str,
    labelColumn: str,
    skipRows: int,
    xScale=1.0,
    yScale=1.0,
    labelScale=1.0,
):

    n = data[SweepColumnName].max()
    n = n.astype(int)
    if labelColumn != None:
        labels = data[labelColumn].dropna().reset_index(drop=True) * labelScale

    curr = 0
    index = list()

    for i in range(skipRows, len(data) - 1):
        curr = data[SweepColumnName][i]
        if curr != data[SweepColumnName][i + 1]:
            index.append(i)

    plt.figure()
    fig, ax = plt.subplots()
    plt.grid()

    prevIndex = 0
    for i in range(n):
        if labelColumn != None:
            plt.plot(
                data[xName][skipRows + prevIndex : index[i]] * xScale,
                data[yName][skipRows + prevIndex : index[i]] * yScale,
                label=f"{labels[i]:.2f}",
            )
        else:
            plt.plot(
                data[xName][skipRows + prevIndex : index[i]] * xScale,
                data[yName][skipRows + prevIndex : index[i]] * yScale,
            )
        prevIndex = index[i]

    return ax


def SetScaleType(type):
    plt.yscale(type)


def PlotOscVoltCurr(
    path_to_dir: str, 
    datasetBaseName, 
    endings:list, 
    title=" ", 
    resistance=100,
    filterValues = False,  
    filteringCurrent = (-1,-1), 
    filteringVoltage=(-1,-1)
    ):
    if(path_to_dir[-1] != '/'):
        path_to_dir += '/'
    dtsets = list()
    for var in endings:
        var = pd.read_csv(path_to_dir + datasetBaseName + var + '.csv', sep=",", skiprows=1)
        var["second"] = (
            var["second"] - var["second"][0]
        ) * 10**9  # to nanoseconds, substract offset
        dtsets.append(var)

    dtsets[2]["Volt"] = dtsets[2]["Volt"] / resistance * 1000  # to mA

    if(filterValues):
        dtsets[2]["Volt"] = savgol_filter(x=dtsets[2]["Volt"], window_length=filteringCurrent[0], polyorder=filteringCurrent[1])
        dtsets[1]["Volt"] = savgol_filter(x=dtsets[1]["Volt"], window_length=filteringVoltage[0], polyorder=filteringVoltage[1])

    fig, axVoltage = plt.subplots()

    pl1 = axVoltage.plot(dtsets[0]["second"], dtsets[0]["Volt"], label="$U_{generator}$")
    pl2 = axVoltage.plot(dtsets[1]["second"], dtsets[1]["Volt"], label="$U_{LED}$")
    axVoltage.set_xlabel("time [ns]")
    axVoltage.set_ylabel("Voltage [V]")

    axCurrent = axVoltage.twinx()

    pl3 = axCurrent.plot(
        dtsets[2]["second"], dtsets[2]["Volt"], label="$\mathcal{I}$", color="green"
    )
    axCurrent.set_ylabel("Current [mA]")
    # axVoltage.legend(loc='center left')
    # axCurrent.legend(loc='center right')

    #join labels
    lns = pl1+pl2+pl3
    labs = [l.get_label() for l in lns]
    axVoltage.legend(lns, labs, loc=0) 

    plt.title(title)
    plt.grid()
    fig.tight_layout()
    plt.show()



def PlotVoltageCurrentDataset(
  dataset: pd.DataFrame
  ):  
  fig, ax1 = plt.subplots()
  ax1.set_xlabel('Time [ns]')
  ax1.set_ylabel('Voltage [V]')
  lns1 = ax1.plot(dataset["x-axis"], dataset['1'], label='U_generator', color = 'black')
  lns2 = ax1.plot(dataset["x-axis"], dataset['3'], label='U_photodiode', color = 'blue')

  ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
  ax2.set_ylabel('I [mA]', color='red')  # we already handled the x-label with ax1
  lns3 = ax2.plot(dataset['x-axis'], dataset['F1'], label = 'I_photodiode', color = 'red')
  ax2.tick_params(axis='y', labelcolor='red')

  #join labels
  lns = lns1+lns2+lns3
  labs = [l.get_label() for l in lns]
  ax1.legend(lns, labs, loc=0) 
  
  plt.grid()
  fig.tight_layout()  # otherwise the right y-label is slightly clipped
  plt.show()

"""
filename: name of the csv file
filepath: name of the filepath
filterValues: flag determining if the Voltage and Current values must be filtered
filteringCurrent: tuple containing window_length and polyorder for current filtering
filteringVoltage: tuple containing window_length and polyorder for voltage filtering
"""
def PlotSingleFile_OPHO(
  fileName = 'x.csv', 
  filePath = 'datasets/LVJ_OPHO2022/LVJ_OPHO2022',
  filterValues = False,  
  filteringCurrent = (-1,-1), 
  filteringVoltage=(-1,-1)
  ):
  if(filePath[-1] != '/'):
    filePath += '/'

  filePath+=fileName
  try:
    dataset=pd.read_csv(filePath, sep=',', skiprows=[1,2,3])
  except:
    print('Error while reading file')
    print('filePath: ' + filePath)
  else:
    if(CheckDataset(dataset.columns) != 'ok'):
      return
    resistance = 100.0 #
    # Convert the voltage on the resistor (U_generator - U_photodiode) to current [mA] 
    dataset['F1'] = dataset['F1'] / resistance * 1000
    dataset['x-axis'] = (dataset['x-axis'] - dataset['x-axis'][0]) * 10**9# to nanoseconds, substract offset
    #filtering values
    if(filterValues):
      dataset['F1'] = savgol_filter(x=dataset['F1'], window_length=filteringCurrent[0], polyorder=filteringCurrent[1])
      dataset['3'] = savgol_filter(x=dataset['3'], window_length=filteringVoltage[0], polyorder=filteringVoltage[1])
    PlotVoltageCurrentDataset(dataset)