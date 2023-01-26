import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

import numpy as np
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit

import os.path
from os import path

from os import listdir
from os.path import isfile, join

import constants as const

colors = ["blue", "green", "orange"]

filesNotToPlot = [
    "led_b_nt_u4v1khz2_",
    "led_b_tru4v1khz6_",
    "led_b_tru5v1khz7_",
    "irtr_v3vf1kof1_",
    "irtr_v3vf1kof2_",
    "irtr_v3vf1kof7_",
    "irtr_v3vf1kof_",
    "irtr_v3vf1k_",
    "irtr_v3vf1k1.csv",
    "irtr_v3vf1kof4.csv",
    "irtr_v3vf1kon2.csv",
    "i_rof_vf_0_5vf1k3.csv",
    "i_ron_vf_0_5vf1k8.csv",
    "i_r_vf_0_5vf1k9.csv",
    "idtr_v2vf1k3_",
    "idtr_v3vf1kof4_",
    "idtr_v2vf1k2.csv",
    "idtr_v2vf1kof2.csv",
    "idtr_v2vf1kon1.csv",
]


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
    colormap=None,
    color_normalize=None,
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

    # fig, ax = plt.subplots()
    plt.grid()

    prevIndex = 0
    mu = list()

    for i in range(n + 1):
        x = data[xName][skipRows + prevIndex : index[i]]
        y = data[yName][skipRows + prevIndex : index[i]]

        if colormap != None:
            plt.plot(
                x * xScale,
                y * yScale,
                label=f"{labels[i]:.2f}",
                color=colormap(color_normalize(labels[i])),
            )
        else:
            plt.plot(
                x * xScale,
                y * yScale,
                label=f"{labels[i]:.2f}",
            )
            # mu ideality factor calculation
        if curveFitFunc != None:
            popt, pcov = curve_fit(curveFitFunc, x.dropna(), y.dropna())
            mu.append(const.q / (popt[1] * const.k * const.T))
        prevIndex = index[i]

    if curveFitFunc != None:
        pd.options.display.float_format = "{:,.3f}".format
        d = {"i_led": labels, "mu": mu}
        df = pd.DataFrame(data=d)
    else:
        df = None
    return None, df


def SweepPlotForwardReverse(
    dfForw: pd.DataFrame,
    dfRev: pd.DataFrame,
    clmnForw: list,
    clmnRev: list,
    vmin,
    vmax,
    curveFitFunc,
    yscale = 'linear'
):
    cmap = mpl.colormaps["viridis"]
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    mu = list()

    plt.figure(figsize=(8, 5))
    axF, muF = SweepPlot(
        dfForw,
        clmnForw[0],
        clmnForw[1],
        clmnForw[2],
        clmnForw[3],
        2,
        yScale=10**3,
        labelScale=1000,
        curveFitFunc=curveFitFunc,
        colormap=cmap,
        color_normalize=norm,
    )
    mu.append(muF)

    axR, muR = SweepPlot(
        dfRev,
        clmnRev[0],
        clmnRev[1],
        clmnRev[2],
        clmnRev[3],
        2,
        yScale=10**3,
        labelScale=1000,
        curveFitFunc=curveFitFunc,
        colormap=cmap,
        color_normalize=norm,
    )
    mu.append(muR)

    plt.grid()
    plt.yscale(yscale)
    plt.xlabel("$U_{pdF} [V]$")
    plt.ylabel("$I_{pdF} [nA]$")

    plt.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=plt.gca(),
        label="$I_{LED} [mA]$",
    )

    return mu


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
        dtsets[0]["second"], dtsets[0]["Volt"], label="$U_{~}$", color=colors[0]
    )
    pl2 = axVoltage.plot(
        dtsets[1]["second"], dtsets[1]["Volt"], label="$U_{d}$", color=colors[1]
    )

    axVoltage.set_ylabel("Voltage [V]")
    axVoltage.grid()

    axCurrent = axVoltage.twinx()
    axCurrent.tick_params(axis="y", labelcolor=colors[2])
    pl3 = axCurrent.plot(
        dtsets[2]["second"], dtsets[2]["Volt"], label="$\mathcal{I}$", color=colors[2]
    )
    axCurrent.set_ylabel("Current [mA]", color=colors[2])

    # join labels
    lns = pl1 + pl2 + pl3
    labs = [l.get_label() for l in lns]
    axVoltage.legend(lns, labs)

    # diode_R = dtsets[1]["Volt"] / dtsets[2]["Volt"] * 1000

    # plt.plot(dtsets[0]["second"], diode_R, label="Resistance")
    # plt.ylabel("Resistance $[k \Omega]$")
    # plt.xlabel("time [ns]")
    # plt.grid()

    plt.title(title)

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

    diodeResistance = dataset[columnNames[2]] / (dataset[columnNames[3]] / 1000)
    plt.figure()
    plt.plot(dataset[columnNames[0]], diodeResistance, label="Resistance")
    plt.ylabel("Resistance [Ohm]")
    plt.xlabel("Time [ns]")
    plt.grid()
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


def PrintDataFromDirectory(directoryPath: str, filterValues=True):
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
        if f in filesNotToPlot:
            print("skipping file: " + f)
            continue
        print("Plotting file: " + f)
        if "nt" in f or "on" in f or "of" in f:
            PlotOscVoltCurr(
                path_to_dir=directoryPath,
                datasetBaseName=f,
                endings=fileEndings,
                title=f,
                filterValues=filterValues,
                filteringCurrent=(100, 5),
                filteringVoltage=(100, 5),
            )
        else:
            PlotOscVoltCurr(
                path_to_dir=directoryPath,
                datasetBaseName=f,
                endings=fileEndings,
                title=f,
            )

    files_dataOneFile = [f for f in onlyfiles if f[-6] != "_" and f[-7] != "_"]
    for f in files_dataOneFile:
        if f in filesNotToPlot:
            print("skipping file: " + f)
            continue
        print("Plotting file: " + f)
        if "nt" in f or "on" in f or "of" in f:
            PlotSingleFile_OPHO(
                fileName=f,
                filePath=directoryPath,
                title=f,
                filterValues=filterValues,
                filteringCurrent=(100, 5),
                filteringVoltage=(100, 5),
            )
        else:
            PlotSingleFile_OPHO(fileName=f, filePath=directoryPath, title=f)

def PlotVoltageTime(dataset:pd.DataFrame):
    plt.figure()
    plt.plot(dataset[0], dataset[1])
    plt.show()

def PrintDataFromDirectoryVoltageTime(directoryPath: str, filterValues = True):
    onlyfiles = [
        f
        for f in listdir(directoryPath)
        if isfile(join(directoryPath, f)) and f[-4:] == ".csv"
    ]
    for f in onlyfiles:
        dataset = pd.read_csv(directoryPath + f)
        PlotVoltageTime(dataset=dataset)

