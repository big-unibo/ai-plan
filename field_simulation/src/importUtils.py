import pandas as pd
import os
import tkinter.filedialog
from configparser import ConfigParser
from ast import literal_eval
from array import array
import math

from dataStructures import *
import criteria3D
import assimilation
import waterBalance
import rectangularMesh
import crop

def configToDict(settingsFilename, params):
    config = ConfigParser(converters={"any": lambda x: literal_eval(x)})
    config.optionxform=str
    modelSettings = config.read(settingsFilename)
    if len(modelSettings) == 0:
        print("ERROR!\nMissing setting file: " + settingsFilename)
        return False

    dictionary = {}
    for section in config.sections():
        dictionary[section] = {}
        for option in config.options(section):
            dictionary[section][option] = (
                params[option]
                if params["iteration"] != -1 and option in params else
                config.getany(section, option)
            )

    return dictionary


def readModelParameters(settingsFilename, params):
    configDict = configToDict(settingsFilename, params)
    print(f"model parameters: {configDict}")

    try:
        C3DParameters.waterRetentionCurve = configDict['model']['waterRetentionCurve']
    except:
        print("ERROR!\nWrong or missing model.waterRetentionCurve in the model settings: " + settingsFilename)
        print("Valid values: 1 Campbell  2 modified Van Genuchten")
        return False
    if C3DParameters.waterRetentionCurve != 1 and C3DParameters.waterRetentionCurve != 2:
        print("ERROR!\nWrong model.waterRetentionCurve in the model settings: " + settingsFilename)
        print("Valid values: 1 Campbell  2 modified Van Genuchten")
        return False

    try:
        C3DParameters.conductivityMean = configDict['model']['conductivityMean']
    except:
        print("ERROR!\nWrong or missing model.conductivityMean in the model settings: " + settingsFilename)
        print("Valid values: 1 LOGARITHMIC 2 HARMONIC 3 GEOMETRIC")
        return False
    if C3DParameters.conductivityMean < 1 or C3DParameters.conductivityMean > 3:
        print("ERROR!\nWrong model.conductivityMean in the model settings: " + settingsFilename)
        print("Valid values: 1 LOGARITHMIC 2 HARMONIC 3 GEOMETRIC")
        return False

    try:
        C3DParameters.conductivityHVRatio = configDict['model']['conductivityHVRatio']
    except:
        print("ERROR!\nWrong or missing model.conductivityHVRatio in the model settings: " + settingsFilename)
        print("Valid values: ]0,10]")
        return False
    if C3DParameters.conductivityHVRatio <= 0 or C3DParameters.conductivityHVRatio > 10:
        print("ERROR!\nWrong model.conductivityHVRatio in the model settings: " + settingsFilename)
        print("Valid values: ]0,10]")
        return False

    # [processes]
    try:
        C3DParameters.computeInfiltration = configDict['processes']['computeInfiltration']
    except:
        print("WARNING!\nWrong or missing processes.computeInfiltration in the model settings: " + settingsFilename)
        print("The default will be set: computeInfiltration = True")
        C3DParameters.computeInfiltration = True

    try:
        C3DParameters.computeSurfaceFlow = configDict['processes']['computeSurfaceFlow']
    except:
        print("WARNING!\nWrong or missing processes.computeSurfaceFlow in the model settings: " + settingsFilename)
        print("The default will be set: computeSurfaceFlow = False")
        C3DParameters.computeSurfaceFlow = False

    try:
        C3DParameters.computeEvaporation = configDict['processes']['computeEvaporation']
    except:
        print("WARNING!\nWrong or missing processes.computeEvaporation in the model settings: " + settingsFilename)
        print("The default will be set: computeEvaporation = True")
        C3DParameters.computeEvaporation = True

    try:
        C3DParameters.computeTranspiration = configDict['processes']['computeTranspiration']
    except:
        print("WARNING!\nWrong or missing processes.computeTranspiration in the model settings: " + settingsFilename)
        print("The default will be set: computeTranspiration = True")
        C3DParameters.computeTranspiration = True

    try:
        C3DParameters.assignIrrigation = configDict['processes']['assignIrrigation']
    except:
        print("WARNING!\nWrong or missing processes.assignIrrigation in the model settings: " + settingsFilename)
        print("The default will be set: assignIrrigation = True")
        C3DParameters.assignIrrigation = True

    # [boundary]
    try:
        C3DParameters.isFreeDrainage = configDict['boundary']['isFreeDrainage']
    except:
        print("WARNING!\nWrong or missing boundary.isFreeDrainage in the model settings: " + settingsFilename)
        print("The default will be set: isFreeDrainage = True")
        C3DParameters.isFreeDrainage = True

    try:
        C3DParameters.isFreeLateralDrainage = configDict['boundary']['isFreeLateralDrainage']
    except:
        print("WARNING!\nWrong or missing boundary.isFreeLateralDrainage in the model settings: " + settingsFilename)
        print("The default will be set: isFreeLateralDrainage = True")
        C3DParameters.isFreeLateralDrainage = True

    try:
        C3DParameters.isSurfaceRunoff = configDict['boundary']['isSurfaceRunoff']
    except:
        print("WARNING!\nWrong or missing boundary.isSurfaceRunoff in the model settings: " + settingsFilename)
        print("The default will be set: isSurfaceRunoff = False")
        C3DParameters.isSurfaceRunoff = False

    try:
        C3DParameters.isWaterTable = configDict['boundary']['isWaterTable']
    except:
        print("WARNING!\nWrong or missing boundary.isWaterTable in the model settings: " + settingsFilename)
        print("The default will be set: isWaterTable = False")
        C3DParameters.isWaterTable = False

    # [initial_conditions]
    try:
        C3DParameters.initialWaterPotential = configDict['initial_conditions']['initialWaterPotential']
    except:
        print("WARNING!\nWrong or missing initial_conditions.initialWaterPotential in the model settings: " + settingsFilename)
        print("The default will be set: initialWaterPotential = -3.0 [m]")
        C3DParameters.initialWaterPotential = -3.0

    try:
        C3DParameters.waterTableDepth = configDict['initial_conditions']['waterTableDepth']
    except:
        print("WARNING!\nWrong or missing initial_conditions.waterTableDepth in the model settings: " + settingsFilename)
        print("The default will be set: waterTableDepth = -3.0 [m]")
        C3DParameters.waterTableDepth = -3.0

    # [surface_properties]
    try:
        C3DParameters.roughness = configDict['surface_properties']['roughness']
    except:
        print("WARNING!\nWrong or missing surface_properties.roughness in the model settings: " + settingsFilename)
        print("The default will be set: roughness = 0.24 [s m-0.33]")
        C3DParameters.roughness = 0.24

    try:
        C3DParameters.pond = configDict['surface_properties']['pond']
    except:
        print("WARNING!\nWrong or missing surface_properties.pond in the model settings: " + settingsFilename)
        print("The default will be set: pond = 0.005 [m]")
        C3DParameters.pond = 0.005

    # [numerical_solution]
    try:
        C3DParameters.maxIterationsNr = configDict['numerical_solution']['maxIterationsNr']
    except:
        print("WARNING!\nWrong or missing numerical_solution.maxIterationsNr in the model settings: " + settingsFilename)
        print("The default will be set: maxIterationsNr = 100")
        C3DParameters.maxIterationsNr = 100

    try:
        C3DParameters.maxApproximationsNr = configDict['numerical_solution']['maxApproximationsNr']
    except:
        print(
            "WARNING!\nWrong or missing numerical_solution.maxIterationsNr in the model settings: " + settingsFilename)
        print("The default will be set: maxApproximationsNr = 10")
        C3DParameters.maxApproximationsNr = 10

    try:
        C3DParameters.residualTolerance = configDict['numerical_solution']['residualTolerance']
    except:
        print(
            "WARNING!\nWrong or missing numerical_solution.residualTolerance in the model settings: " + settingsFilename)
        print("The default will be set: residualTolerance = 1E-12")
        C3DParameters.residualTolerance = 1E-12

    try:
        C3DParameters.MBRThreshold = configDict['numerical_solution']['MBRThreshold']
    except:
        print(
            "WARNING!\nWrong or missing numerical_solution.MBRThreshold in the model settings: " + settingsFilename)
        print("The default will be set: MBRThreshold = 1E-5")
        C3DParameters.MBRThreshold = 1E-5

    try:
        C3DParameters.deltaT_min = configDict['numerical_solution']['minDeltaT']
    except:
        print(
            "WARNING!\nWrong or missing numerical_solution.minDeltaT in the model settings: " + settingsFilename)
        print("The default will be set: minDeltaT = 6")
        C3DParameters.deltaT_min = 6

    try:
        C3DParameters.deltaT_max = configDict['numerical_solution']['maxDeltaT']
    except:
        print(
            "WARNING!\nWrong or missing numerical_solution.maxDeltaT in the model settings: " + settingsFilename)
        print("The default will be set: maxDeltaT = 3600")
        C3DParameters.deltaT_max = 6

    # [simulation_type]
    try:
        C3DParameters.isForecast = configDict['simulation_type']['isForecast']
    except:
        print("WARNING!\nWrong or missing simulation_type.isForecast in the model settings: " + settingsFilename)
        print("The default will be set: isForecast = False")
        C3DParameters.isForecast = False

    try:
        C3DParameters.isFirstAssimilation = configDict['simulation_type']['isFirstAssimilation']
    except:
        print("WARNING!\nWrong or missing simulation_type.isFirstAssimilation in the model settings: " + settingsFilename)
        print("The default will be set: isFirstAssimilation = False")
        C3DParameters.isFirstAssimilation = False

    try:
        C3DParameters.isPeriodicAssimilation = configDict['simulation_type']['isPeriodicAssimilation']
    except:
        print("WARNING!\nWrong or missing simulation_type.isPeriodicAssimilation in the model settings: " + settingsFilename)
        print("The default will be set: isPeriodicAssimilation = False")
        C3DParameters.isPeriodicAssimilation = False

    if C3DParameters.isPeriodicAssimilation and C3DParameters.isForecast:
        print("WARNING!\nisPeriodicAssimilation and isForecast cannot both be assigned true in the model settings: " + settingsFilename)
        print("The default will be set: isPeriodicAssimilation = False")
        C3DParameters.isPeriodicAssimilation = False

    try:
        C3DParameters.isIrrigation = configDict['simulation_type']['isIrrigation']
    except:
        C3DParameters.isIrrigation = True

    try:
        C3DParameters.isVisual = configDict['simulation_type']['isVisual']
    except:
        C3DParameters.isVisual = True

    try:
        C3DParameters.assimilationInterval = configDict['simulation_type']['assimilationInterval']
    except:
        C3DParameters.assimilationInterval = NODATA
    if C3DParameters.isPeriodicAssimilation or C3DParameters.isForecast:
        if C3DParameters.assimilationInterval == NODATA or C3DParameters.assimilationInterval < 1:
            print("ERROR!\nWrong or missing simulation_type.assimilationInterval in the model settings: " + settingsFilename)
            print("Valid values: greater than or equal to 1 hour")
            return False

    try:
        C3DParameters.forecastPeriod = configDict['simulation_type']['forecastPeriod']
    except:
        C3DParameters.forecastPeriod = NODATA
    if C3DParameters.isForecast:
        if C3DParameters.forecastPeriod == NODATA or C3DParameters.forecastPeriod < 24:
            print("ERROR!\nWrong or missing simulation_type.forecastPeriod in the model settings: " + settingsFilename)
            print("Valid values: greater than or equal to 24 hours")
            return False

    return True


def readFieldParameters(fieldSettingsFilename, params):
    configDict = configToDict(fieldSettingsFilename, params)
    print(f"field parameters: {configDict}")

    # [location]
    try:
        C3DStructure.latitude = configDict['location']['lat']
    except:
        print("ERROR! Missing location.lat in field.ini")
        return False

    try:
        C3DStructure.longitude = configDict['location']['lon']
    except:
        print("ERROR! Missing location.lon in field.ini")
        return False

    try:
        C3DStructure.z = configDict['location']['z']
    except:
        print("ERROR! Missing location.z in field.ini")
        return False

    try:
        C3DStructure.timeZone = configDict['location']['timeZone']
    except:
        print("ERROR! Missing location.timeZone in field.ini")
        return False

    # [size]
    try:
        width = configDict['size']['width']
    except:
        print("ERROR! Missing size.width in field.ini")
        return False

    try:
        height = configDict['size']['height']
    except:
        print("ERROR! Missing size.height in field.ini")
        return False

    try:
        C3DStructure.gridDepth = configDict['size']['depth']
    except:
        print("ERROR! Missing size.depth in field.ini")
        return False

    try:
        C3DStructure.cellSize = configDict['size']['cellSize']
    except:
        print("ERROR! Missing size.cellSize in field.ini")
        return False

    initialize3DStructure(width, height, C3DStructure.cellSize)

    # [slope]
    try:
        C3DStructure.slopeX = configDict['slope']['slopeX']
    except:
        C3DStructure.slopeX = 0

    try:
        C3DStructure.slopeY = configDict['slope']['slopeY']
    except:
        C3DStructure.slopeY = 0

    try:
        C3DStructure.plantSlope = configDict['slope']['plantSlope']
    except:
        C3DStructure.plantSlope = 0

    try:
        C3DStructure.plantSlopeWidth = configDict['slope']['plantSlopeWidth']
    except:
        C3DStructure.plantSlopeWidth = 0

    # [dripper]
    try:
        C3DStructure.flowRate = configDict['dripper']['flowRate']
    except:
        C3DStructure.flowRate = 0

    # Build rectangular mesh
    print("Building rectangle mesh...")
    rectangularMesh.rectangularMeshCreation()
    print("Nr. of rectangles:", C3DStructure.nrRectangles)
    print("Total area [m^2]:", C3DStructure.totalArea)

    # set [plant]
    plantIndices.clear()
    if C3DParameters.computeTranspiration:
        if len(configDict['plant']['x']) > 1 and len(configDict['plant']['y']) > 1:
            print("set plant positions...")
            crop.x = configDict['plant']['x']
            crop.y = configDict['plant']['y']

            if len(crop.x) != len(crop.y):
                print("ERROR! Different number of plant.x,y in field.ini")
                return False

            for i in range(len(crop.x)):
                surfaceIndex = rectangularMesh.getSurfaceIndex(crop.x[i], crop.y[i])
                if surfaceIndex != NODATA:
                    plantIndices.append(surfaceIndex)
        else:
            print("ERROR! Missing plant positions in field.ini")
            print("Set at least two plants to define the row direction.")
            return False

    # set [dripper]
    dripperIndices.clear()
    if len(configDict['dripper']['x']) > 0 and len(configDict['dripper']['y']) > 0:
        print("set dripper positions...")
        x = configDict['dripper']['x']
        y = configDict['dripper']['y']

        if len(x) != len(y):
            print("ERROR: different number of dripper x,y positions.")
            return False

        for i in range(len(x)):
            surfaceIndex = rectangularMesh.getSurfaceIndex(x[i], y[i])
            if surfaceIndex != NODATA:
                dripperIndices.append(surfaceIndex)

    return True


def readCropParameters(cropSettingsFilename, params):
    configDict = configToDict(cropSettingsFilename, params)
    print(f"crop parameters: {configDict}")

    # [LAI]
    if len(configDict['LAI']['laiMonth']) != 12:
        print("ERROR!\nWrong LAI.laiMonth in the crop settings: " + cropSettingsFilename)
        print("Values must be 12")
        return False

    try:
        crop.currentCrop.laiMonth = configDict['LAI']['laiMonth']
    except:
        print("ERROR!\nWrong LAI.laiMonth values in the crop settings: " + cropSettingsFilename)
        print("Values must be numeric.")
        return False

    try:
        crop.currentCrop.rootDepthZero = configDict['root']['rootDepthZero']
    except:
        crop.currentCrop.rootDepthZero = 0.05

    try:
        crop.currentCrop.rootDepthMax = configDict['root']['rootDepthMax']
    except:
        print("ERROR!\nWrong or missing root.rootDepthMax in the crop settings: " + cropSettingsFilename)
        return False

    if crop.currentCrop.rootDepthMax <= crop.currentCrop.rootDepthZero:
        print("ERROR!\nWrong root.rootDepthMax in the crop settings: " + cropSettingsFilename)
        print("rootDepthMax must be greater than rootDepthZero.")
        return False

    try:
        crop.currentCrop.rootWidth = configDict['root']['rootWidth']
    except:
        print("ERROR!\nWrong or missing root.rootWidth in the crop settings: " + cropSettingsFilename)
        return False
    if crop.currentCrop.rootWidth <= 0:
        print("ERROR!\nWrong root.rootWidth in the crop settings: " + cropSettingsFilename)
        print("Value must be greater than zero.")
        return False

    try:
        crop.currentCrop.rootXDeformation = configDict['root']['rootXDeformation']
    except:
        print("ERROR!\nWrong or missing root.rootXDeformation in the crop settings: " + cropSettingsFilename)
        return False
    if crop.currentCrop.rootXDeformation < 0 or crop.currentCrop.rootXDeformation > 1:
        print("ERROR!\nWrong root.rootXDeformation in the crop settings: " + cropSettingsFilename)
        print("Valid range: [0,1]")
        return False

    try:
        crop.currentCrop.rootZDeformation = configDict['root']['rootZDeformation']
    except:
        print("ERROR!\nWrong or missing root.rootZDeformation in the crop settings: " + cropSettingsFilename)
        return False
    if crop.currentCrop.rootZDeformation < -1 or crop.currentCrop.rootZDeformation > 1:
        print("ERROR!\nWrong root.rootZDeformation in the crop settings: " + cropSettingsFilename)
        print("Valid range: [-1,1]")
        return False

    try:
        crop.currentCrop.kcMax = configDict['transpiration']['kcMax']
    except:
        print("ERROR!\nWrong or missing transpiration.kcMax in the crop settings: " + cropSettingsFilename)
        return False
    if crop.currentCrop.kcMax < 0:
        print("ERROR!\nWrong transpiration.kcMax in the crop settings: " + cropSettingsFilename)
        print("Value must be greater than zero.")
        return False

    try:
        crop.currentCrop.fRAW = configDict['transpiration']['fRAW']
    except:
        print("ERROR!\nWrong or missing transpiration.fRAW in the crop settings: " + cropSettingsFilename)
        return False
    if crop.currentCrop.fRAW < 0 or crop.currentCrop.fRAW > 1:
        print("ERROR!\nWrong transpiration.kcMax in the crop settings: " + cropSettingsFilename)
        print("Valid range: [0,1]")
        return False
    print(crop.currentCrop.fRAW)

    return True


def getStateFileName(isSave):
    root = tkinter.Tk()
    options = {'defaultextension': ".csv", 'filetypes': [("Comma separated values", ".csv")], 'initialdir': "data"}
    if isSave:
        fileName = tkinter.filedialog.asksaveasfilename(**options)
    else:
        fileName = tkinter.filedialog.askopenfilename(**options)
    root.destroy()
    return fileName


def readWaterTable(waterPath):
    date = []
    depth = []
    waterTableData = pd.read_csv(os.path.join(waterPath, "watertable.csv"))
    for _, waterTable in waterTableData.iterrows():
        date.append(pd.to_datetime(waterTable['date']))
        depth.append(waterTable['depth'])
    return date, depth


def readMeteoData(meteoPath):
    humidityFile = "air_humidity.csv"
    humidity = pd.read_csv(os.path.join(meteoPath, humidityFile))

    radiationsFile = "solar_radiation.csv"
    radiations = pd.read_csv(os.path.join(meteoPath, radiationsFile))

    temperatureFile = "air_temperature.csv"
    temperature = pd.read_csv(os.path.join(meteoPath, temperatureFile))

    windFile = "wind_speed.csv"
    wind = pd.read_csv(os.path.join(meteoPath, windFile))

    meteoData = [humidity, radiations, temperature, wind]

    for i in range(len(meteoData) - 1):
        if meteoData[i].iloc[0]["timestamp"] != meteoData[i + 1].iloc[0]["timestamp"]:
            raise Exception("Meteo files have different time spans")

    for i in range(len(meteoData)):
        meteoData[i] = meteoData[i].set_index(['timestamp'])

    mergedDf = meteoData[0]
    for i in range(1, len(meteoData)):
        mergedDf = mergedDf.merge(meteoData[i], how='outer', left_index=True, right_index=True)

    return mergedDf.reset_index()


def readWaterData(waterPath, meteo_start, meteo_end):
    irrigationFile = "irrigation.csv"
    irrigation = pd.read_csv(os.path.join(waterPath, irrigationFile))

    precipitationsFile = "precipitation.csv"
    precipitations = pd.read_csv(os.path.join(waterPath, precipitationsFile))

    if irrigation.iloc[0]["timestamp"] != precipitations.iloc[0]["timestamp"]:
        raise Exception("Water files have different time spans")

    if irrigation.iloc[-1]["timestamp"] != precipitations.iloc[-1]["timestamp"]:
        raise Exception("Water files have different time spans")

    irrigation = irrigation.set_index(['timestamp'])
    precipitations = precipitations.set_index(['timestamp'])

    mergedDf = irrigation.merge(precipitations, how='outer', left_index=True, right_index=True)
    mergedDf = mergedDf.reset_index()

    if meteo_start != NODATA and meteo_end != NODATA:
        if meteo_start != mergedDf.iloc[0]["timestamp"] or meteo_end != mergedDf.iloc[-1]["timestamp"]:
            raise Exception("Irrigation file has a different time span")

    return mergedDf


def transformDates(meteoData, waterData):
    meteoData["time"] = pd.to_datetime(meteoData["timestamp"], unit="s")
    waterData["time"] = pd.to_datetime(waterData["timestamp"], unit="s")
    return meteoData, waterData


def loadObsData(fileName):
    obsState = pd.read_csv(fileName)
    assimilation.assimilate(obsState)
    waterBalance.updateStorage()
    return True


def writeObsData(fileName, obsData, timeStamp):
    header = "x,y,z,value\n"
    f = open(fileName, "w")
    f.write(header)

    df = obsData[obsData["timestamp"] == timeStamp]
    if not df.empty:
        for column in [column for column in list(df.columns) if column != "timestamp"]:
            splitted_column = column.split("_")
            value = df[column].values[0]
            if math.isnan(value) or (value <= -2500):
                value = NODATA
            f.write(
                "{:.2f}".format(float(splitted_column[2][1:]) / 100) + ","  # x
                + "{:.1f}".format(float(splitted_column[1][1:]) / 100) + ","  # y
                + "{:.1f}".format(float(splitted_column[0][1:]) / 100) + ","  # z
                + "{:.1f}".format(value) + "\n"  # psi
            )
    return not df.empty


def saveCurrentModelState(stateFileName):
    float_array = array('d')
    for i in range(len(C3DCells)):
        float_array.append(C3DCells[i].H)

    f = open(stateFileName, "wb")
    float_array.tofile(f)
    f.close()


def loadModelState(stateFileName):
    f = open(stateFileName, "rb")
    float_array = array('d')
    float_array.fromfile(f, len(C3DCells))
    f.close()

    for i in range(len(C3DCells)):
        H = float_array[i]
        signPsi = H - C3DCells[i].z
        criteria3D.setMatricPotential(i, signPsi)

    waterBalance.updateStorage()
