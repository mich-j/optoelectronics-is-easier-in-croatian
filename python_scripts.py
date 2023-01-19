import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit

import os.path
from os import path

from os import listdir
from os.path import isfile, join

import constants as const

colors = ["blue", "green", "orange"]


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
    curveFitFunc=None,
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
    mu = list()

    for i in range(n + 1):
        x = data[xName][skipRows + prevIndex : index[i]]
        y = data[yName][skipRows + prevIndex : index[i]]
        if labelColumn != None:
            plt.plot(
                x * xScale,
                y * yScale,
                label=f"{labels[i]:.2f}",
            )
            if curveFitFunc != None:
                popt, pcov = curve_fit(curveFitFunc, x.dropna(), y.dropna())
                mu.append(const.q / (popt[1] * const.k * const.T))
        else:
            plt.plot(
                x * xScale,
                y * yScale,
            )
        prevIndex = index[i]

    if curveFitFunc != None:
        pd.options.display.float_format = '{:,.3f}'.format
        d = {"i_led": labels, "mu": mu}
        df = pd.DataFrame(data=d)
    else:
        df = None

    return ax, df


def SetScaleType(type):
    plt.yscale(type)


def PlotOscVoltCurr(
    path_to_dir: str,
    datasetBaseName,
    endings: list,
    title=" ",
    resistance=100,
    filterValues=False,
    filteringCurrent=(-1, -1),
    filteringVoltage=(-1, -1),
):
    if path_to_dir[-1] != "/":
        path_to_dir += "/"
    dtsets = list()
    for var in endings:
        var = pd.read_csv(
            path_to_dir + datasetBaseName + var + ".csv", sep=",", skiprows=1
        )
        var["second"] = (
            var["second"] - var["second"][0]
        ) * 10**9  # to nanoseconds, substract offset
        dtsets.append(var)

    dtsets[2]["Volt"] = dtsets[2]["Volt"] / resistance * 1000  # to mA

    if filterValues:
        dtsets[2]["Volt"] = savgol_filter(
            x=dtsets[2]["Volt"],
            window_length=filteringCurrent[0],
            polyorder=filteringCurrent[1],
        )
        dtsets[1]["Volt"] = savgol_filter(
            x=dtsets[1]["Volt"],
            window_length=filteringVoltage[0],
            polyorder=filteringVoltage[1],
        )

    fig, axVoltage = plt.subplots()

    pl1 = axVoltage.plot(
        dtsets[0]["second"], dtsets[0]["Volt"], label="$U_{generator}$", color=colors[0]
    )
    pl2 = axVoltage.plot(
        dtsets[1]["second"], dtsets[1]["Volt"], label="$U_{LED}$", color=colors[1]
    )
    axVoltage.set_xlabel("time [ns]")
    axVoltage.set_ylabel("Voltage [V]")

    axCurrent = axVoltage.twinx()
    axCurrent.tick_params(axis="y", labelcolor=colors[2])
    pl3 = axCurrent.plot(
        dtsets[2]["second"], dtsets[2]["Volt"], label="$\mathcal{I}$", color=colors[2]
    )
    axCurrent.set_ylabel("Current [mA]", color=colors[2])

    # join labels
    lns = pl1 + pl2 + pl3
    labs = [l.get_label() for l in lns]
    axVoltage.legend(lns, labs, loc=0)

    plt.title(title)
    plt.grid()
    fig.tight_layout()
    plt.show()


def PlotVoltageCurrentDataset(dataset: pd.DataFrame, title=" "):
    columnNames = dataset.columns
    fig, ax1 = plt.subplots()
    ax1.set_xlabel("Time [ns]")
    ax1.set_ylabel("Voltage [V]")
    lns1 = ax1.plot(
        dataset[columnNames[0]],
        dataset[columnNames[1]],
        label="U_generator",
        color=colors[0],
    )
    lns2 = ax1.plot(
        dataset[columnNames[0]],
        dataset[columnNames[2]],
        label="U_photodiode",
        color=colors[1],
    )

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel("I [mA]", color=colors[2])  # we already handled the x-label with ax1
    lns3 = ax2.plot(
        dataset[columnNames[0]],
        dataset[columnNames[3]],
        label="I_photodiode",
        color=colors[2],
    )
    ax2.tick_params(axis="y", labelcolor=colors[2])

    # join labels
    lns = lns1 + lns2 + lns3
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc=0)

    plt.title(title)
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
    fileName="x.csv",
    filePath="datasets/LVJ_OPHO2022/LVJ_OPHO2022",
    filterValues=False,
    filteringCurrent=(-1, -1),
    filteringVoltage=(-1, -1),
    title=" ",
):
    if filePath[-1] != "/":
        filePath += "/"

    filePath += fileName
    try:
        dataset = pd.read_csv(filePath, sep=",", skiprows=[1, 2, 3])
    except:
        print("Error while reading file")
        print("filePath: " + filePath)
    else:
        # if(CheckDataset(dataset.columns, columnNames) != 'ok'):
        #   return
        resistance = 100.0  #
        columnNames = dataset.columns
        # Convert the voltage on the resistor (U_generator - U_photodiode) to current [mA]
        dataset[columnNames[3]] = dataset[columnNames[3]] / resistance * 1000
        dataset[columnNames[0]] = (
            dataset[columnNames[0]] - dataset[columnNames[0]][0]
        ) * 10**9  # to nanoseconds, substract offset
        # filtering values
        if filterValues:
            dataset[columnNames[3]] = savgol_filter(
                x=dataset[columnNames[3]],
                window_length=filteringCurrent[0],
                polyorder=filteringCurrent[1],
            )
            dataset[columnNames[2]] = savgol_filter(
                x=dataset[columnNames[2]],
                window_length=filteringVoltage[0],
                polyorder=filteringVoltage[1],
            )
        PlotVoltageCurrentDataset(dataset=dataset, title=title)


def GetBaseName(s=""):
    return s[0 : next(i for i in reversed(range(len(s))) if s[i] == "_") + 1]


def GetEnding(s=""):
    return s[next(i for i in reversed(range(len(s))) if s[i] == "_") + 1 : -4]


def PrintDataFromDirectory(directoryPath: str):
    onlyfiles = [
        f
        for f in listdir(directoryPath)
        if isfile(join(directoryPath, f)) and f[-4:] == ".csv"
    ]
    # print(onlyfiles)
    files_dataThreeFiles = [f for f in onlyfiles if f[-6] == "_" or f[-7] == "_"]
    fileBaseName = [GetBaseName(files_dataThreeFiles[0])]
    end = [GetEnding(files_dataThreeFiles[0])]
    fileEndings = end
    for f in files_dataThreeFiles:
        if not (GetBaseName(f) in fileBaseName):
            fileEndings.append(end)
            end.clear()
            fileBaseName.append(GetBaseName(f))
        end.append(GetEnding(f))

    for f in fileBaseName:
        PlotOscVoltCurr(
            path_to_dir=directoryPath, datasetBaseName=f, endings=fileEndings, title=f
        )

    files_dataOneFile = [f for f in onlyfiles if f[-6] != "_" and f[-7] != "_"]
    for f in files_dataOneFile:
        PlotSingleFile_OPHO(fileName=f, filePath=directoryPath, title=f)
