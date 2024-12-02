# soil.py

import numpy as np
from math import fabs, log, sqrt
from dataStructures import *
import pandas as pd


class CSoilHorizon:
    upperDepth = NODATA     # [m]
    lowerDepth = NODATA     # [m]
    sand = NODATA           # [%]
    silt = NODATA           # [%]
    clay = NODATA           # [%]
    Campbell_he = NODATA    # [m]
    Campbell_b = NODATA     # [-]
    Campbell_n = NODATA     # [-]
    VG_he = NODATA          # [m]
    VG_alpha = NODATA       # [m-1]
    VG_n = NODATA           # [-]
    VG_m = NODATA           # [-]
    VG_Sc = NODATA          # [-]
    VG_thetaR = NODATA      # [m3 m-3]
    Mualem_L = NODATA       # [-]
    thetaS = NODATA         # [m3 m-3]
    Ks = NODATA             # [m s-1]


# global arrays
depth = np.array([], np.float64)
thickness = np.array([], np.float64)
horizon = CSoilHorizon()


def readHorizon(soilFileName, params):
    global horizon
    soilDataFrame = pd.read_csv(soilFileName)
    # Pay attention: this works only with one horizon!
    soilData = soilDataFrame.loc[0]
    if params["iteration"] != -1:
        for key, value, in params.items():
            if key in soilData:
                soilData[key] = value
    print(soilData)

    horizon.upperDepth = soilData["upper_depth"]
    horizon.lowerDepth = soilData["lower_depth"]
    horizon.sand = soilData["sand"]
    horizon.silt = soilData["silt"]
    horizon.clay = soilData["clay"]
    horizon.Campbell_he = soilData["Campbell_he"]
    horizon.Campbell_b = soilData["Campbell_b"]
    horizon.Campbell_n = 2.0 + (3.0 / horizon.Campbell_b)
    horizon.VG_he = soilData["VG_he"]
    horizon.VG_alpha = soilData["VG_alpha"]
    horizon.VG_n = soilData["VG_n"]
    horizon.VG_m = 1. - (1. / horizon.VG_n)
    horizon.VG_Sc = pow(1. + pow(horizon.VG_alpha * fabs(horizon.VG_he),
                                 horizon.VG_n), -horizon.VG_m)
    horizon.VG_thetaR = soilData["thetaR"]
    horizon.thetaS = soilData["thetaS"]
    horizon.Ks = soilData["Ks"]
    horizon.Mualem_L = 0.5
    return horizon


def searchProgressionFactor(minThickness, maxThickness, maxThicknessDepth):
    factor = 1.01
    bestError = 9999
    bestFactor = factor
    while factor <= 2.0:
        myThickness = minThickness
        currentDepth = minThickness * 0.5
        while myThickness < maxThickness:
            nextThickness = min(maxThickness, myThickness * factor)
            currentDepth += (myThickness + nextThickness) * 0.5
            myThickness = nextThickness
        error = fabs(currentDepth - maxThicknessDepth)
        if error < bestError:
            bestError = error
            bestFactor = factor
        factor += 0.01

    return bestFactor


# set depth and thickness of layers
def setLayers(totalDepth, minThickness, maxThickness, maxThicknessDepth):
    # search progression factor
    factor = searchProgressionFactor(minThickness, maxThickness, maxThicknessDepth)

    nrLayers = 1
    prevThickness = minThickness
    currentDepth = minThickness * 0.5
    while currentDepth < totalDepth:
        nextThickness = min(maxThickness, prevThickness * factor)
        currentDepth += (prevThickness + nextThickness) * 0.5
        prevThickness = nextThickness
        nrLayers += 1

    z = np.zeros(nrLayers, np.float64)
    thick = np.zeros(nrLayers, np.float64)
    z[0] = 0.0
    thick[0] = 0.0
    for i in range(1, nrLayers):
        top = z[i - 1] + thick[i - 1] * 0.5
        if i == 1:
            thick[i] = minThickness
        else:
            if i == (nrLayers - 1):
                thick[i] = totalDepth - top
            else:
                thick[i] = min(maxThickness, thick[i - 1] * factor)
        z[i] = top + thick[i] * 0.5
    return nrLayers, z, thick


# [m3 m-3] volumetric water content
def getVolumetricWaterContent(i):
    if C3DCells[i].isSurface:
        return NODATA
    curve = C3DParameters.waterRetentionCurve
    Se = C3DCells[i].Se
    return waterContent(curve, Se)


# [-] degree of saturation
def getDegreeOfSaturation(i):
    if C3DCells[i].isSurface:
        if C3DCells[i].H > C3DCells[i].z:
            return 1.0
        else:
            return 0.0
    curve = C3DParameters.waterRetentionCurve
    signPsi = C3DCells[i].H - C3DCells[i].z
    return degreeOfSaturation(curve, signPsi)


# [m s-1] hydraulic conductivity
def getHydraulicConductivity(i):
    if C3DCells[i].isSurface:
        return NODATA
    curve = C3DParameters.waterRetentionCurve
    return hydraulicConductivity(curve, C3DCells[i].Se)


# [m] air entry potential (with sign)
def airEntryPotential(curve):
    if curve == CAMPBELL:
        return -horizon.Campbell_he
    elif curve == IPPISCH_VG:
        return -horizon.VG_he
    else:
        return NODATA


# [m] water potential from degree of saturation [-]
def waterPotential(curve, Se):
    if curve == CAMPBELL:
        return -horizon.Campbell_he * Se ** (-horizon.Campbell_b)
    elif curve == IPPISCH_VG:
        return -(1. / horizon.VG_alpha) * ((1. / (Se * horizon.VG_Sc))
                                           ** (1. / horizon.VG_m) - 1.) ** (1. / horizon.VG_n)
    else:
        return NODATA


# [m3 m-3] volumetric water content from degree of saturation [-]
def waterContent(curve, Se):
    if curve == CAMPBELL:
        return Se * horizon.thetaS
    elif curve == IPPISCH_VG:
        return Se * (horizon.thetaS - horizon.VG_thetaR) + horizon.VG_thetaR
    else:
        return NODATA


# [m s-1] hydraulic conductivity from degree of saturation [-]
def hydraulicConductivity(curve, Se):
    k = NODATA

    if curve == CAMPBELL:
        psi = horizon.Campbell_he * Se ** (-horizon.Campbell_b)
        k = horizon.Ks * (horizon.Campbell_he / psi) ** horizon.Campbell_n

    if curve == IPPISCH_VG:
        num = 1. - pow(1. - pow(Se * horizon.VG_Sc, 1. / horizon.VG_m), horizon.VG_m)
        den = 1. - pow(1. - pow(horizon.VG_Sc, 1. / horizon.VG_m), horizon.VG_m)
        k = horizon.Ks * pow(Se, horizon.Mualem_L) * pow((num / den), 2.)
    return k


# [-] degree of saturation from water potential [m]
def degreeOfSaturation(curve, signPsi):
    airEntry = airEntryPotential(curve)
    if signPsi >= airEntry:
        return 1.0

    Se = NODATA
    if curve == CAMPBELL:
        Se = pow(fabs(signPsi) / horizon.Campbell_he, -1. / horizon.Campbell_b)
    elif curve == IPPISCH_VG:
        Se = (1. / horizon.VG_Sc) * pow(1. + pow(horizon.VG_alpha
                                                 * fabs(signPsi), horizon.VG_n), -horizon.VG_m)
    return Se


# [-] degree of saturation from volumetric water content [m3 m-3]
def SeFromTheta(curve, theta):
    if theta >= horizon.thetaS:
        return 1.
    if curve == CAMPBELL:
        return theta / horizon.thetaS
    elif curve == IPPISCH_VG:
        return (theta - horizon.VG_thetaR) / (horizon.thetaS - horizon.VG_thetaR)
    else:
        return NODATA


def psiFromTheta(curve, theta):
    Se = SeFromTheta(curve, theta)
    return waterPotential(curve, Se)


def thetaFromPsi(curve, signPsi):
    Se = degreeOfSaturation(curve, signPsi)
    return waterContent(curve, Se)


def dTheta_dPsi(curve, signPsi):
    airEntry = airEntryPotential(curve)
    if signPsi > airEntry:
        return 0.0
    if curve == CAMPBELL:
        theta = horizon.thetaS * degreeOfSaturation(curve, signPsi)
        return -theta / (horizon.Campbell_b * signPsi)
    elif curve == IPPISCH_VG:
        dSe_dPsi = horizon.VG_alpha * horizon.VG_n * \
                   (horizon.VG_m * pow(1. + pow(horizon.VG_alpha * fabs(signPsi), horizon.VG_n), -(horizon.VG_m + 1.))
                    * pow(horizon.VG_alpha * fabs(signPsi), horizon.VG_n - 1.))
        dSe_dPsi *= (1. / horizon.VG_Sc)
        return dSe_dPsi * (horizon.thetaS - horizon.VG_thetaR)


def get_dTheta_dH(i):
    if C3DCells[i].isSurface:
        return NODATA
    curve = C3DParameters.waterRetentionCurve
    return dTheta_dH(curve, C3DCells[i].H0, C3DCells[i].H, C3DCells[i].z)


def dTheta_dH(curve, H0, H1, z):
    psi0 = H0 - z
    psi1 = H1 - z
    if fabs(psi1 - psi0) < EPSILON_METER:
        return dTheta_dPsi(curve, psi0)
    else:
        theta0 = thetaFromPsi(curve, psi0)
        theta1 = thetaFromPsi(curve, psi1)
        return (theta1 - theta0) / (psi1 - psi0)


# [m s-1] mean value of hydraulic conductivity
def meanK(meanType, k1, k2):
    k = NODATA
    if meanType == LOGARITHMIC:
        if k1 != k2:
            k = (k1 - k2) / log(k1 / k2)
        else:
            k = k1
    elif meanType == HARMONIC:
        k = 2.0 / (1.0 / k1 + 1.0 / k2)
    elif meanType == GEOMETRIC:
        k = sqrt(k1 * k2)
    return k


# [m3 m-3] water content at field capacity
def getFieldCapacityWC():
    curve = C3DParameters.waterRetentionCurve
    fcMin = -10             # [kPa] clay < 20% : sandy soils
    fcMax = -33             # [kPa] clay > 50% : clay soils
    clayMin = 20            # [%]
    clayMax = 50            # [%]

    if horizon.clay < clayMin:
        fieldCapacity = fcMin
    elif horizon.clay >= clayMax:
        fieldCapacity = fcMax
    else:
        clayFactor = (horizon.clay - clayMin) / (clayMax - clayMin)
        fieldCapacity = fcMin + (fcMax - fcMin) * clayFactor

    FC = fieldCapacity / 9.81       # [m]
    return thetaFromPsi(curve, FC)


# [m3 m-3] water content at wilting point
def getWiltingPointWC():
    curve = C3DParameters.waterRetentionCurve
    WP = -1600.  # [kPa]
    WP /= 9.81   # [m]
    return thetaFromPsi(curve, WP)


# [m3 m-3] water content at hygroscopic moisture
def getHygroscopicWC():
    curve = C3DParameters.waterRetentionCurve
    HH = -3100.  # [kPa]
    HH /= 9.81   # [m]
    return thetaFromPsi(curve, HH)
