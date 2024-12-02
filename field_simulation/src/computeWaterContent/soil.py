from math import fabs
import pandas as pd
NODATA = -9999.


class CSoilHorizon:
    upperDepth = NODATA     # [m]
    lowerDepth = NODATA     # [m]
    sand = NODATA           # [%]
    silt = NODATA           # [%]
    clay = NODATA           # [%]
    VG_he = NODATA          # [J kg^-1]
    VG_alpha = NODATA       # [kg J^-1]
    VG_n = NODATA           # [-]
    VG_m = NODATA           # [-]
    VG_Sc = NODATA          # [-]
    VG_thetaR = NODATA      # [m^3 m^-3]
    thetaS = NODATA         # [m^3 m^-3]
    Ks = NODATA             # [kg s m^-3]


horizon = CSoilHorizon()


def readHorizon(soilFileName):
    global horizon
    soilDataFrame = pd.read_csv(soilFileName)
    soilData = soilDataFrame.loc[0]

    horizon.VG_he = soilData["VG_he"]
    horizon.VG_alpha = soilData["VG_alpha"]
    horizon.VG_n = soilData["VG_n"]

    horizon.VG_m = 1. - (1. / horizon.VG_n)
    horizon.VG_Sc = pow(1. + pow(horizon.VG_alpha * fabs(horizon.VG_he),
                                 horizon.VG_n), -horizon.VG_m)

    horizon.VG_thetaR = soilData["thetaR"]
    horizon.thetaS = soilData["thetaS"]


def degreeOfSaturation(signPsi):
    if signPsi >= horizon.VG_he:
        return 1.0
    else:
        return (1. / horizon.VG_Sc) * pow(1. + pow(horizon.VG_alpha * fabs(signPsi), horizon.VG_n), -horizon.VG_m)


def thetaFromPsi(signPsi):
    Se = degreeOfSaturation(signPsi)
    return Se * (horizon.thetaS - horizon.VG_thetaR) + horizon.VG_thetaR

