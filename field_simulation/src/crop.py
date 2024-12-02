# crop.py
import math
import numpy as np
import rectangularMesh
import soil
from dataStructures import *

MAX_EVAPORATION_DEPTH = 0.15  # [m]


class CCrop:
    laiMonth = [float]          # [m2 m-2]
    rootDepthZero = NODATA      # [m]
    rootDepthMax = NODATA       # [m]
    rootWidth = NODATA          # [m]
    rootXDeformation = NODATA   # [-]
    rootZDeformation = NODATA   # [-]
    kcMax = NODATA              # [-]
    fRAW = NODATA               # [-]
    currentLAI = NODATA
    currentRootDepth = NODATA
    currentRootLength = NODATA

    def setMaxRootDepth(self):
        self.currentRootDepth = self.rootDepthMax
        self.currentRootLength = self.currentRootDepth - self.rootDepthZero

    def setCurrentLAI(self, currentDate):
        m = currentDate.month
        lai0 = float(self.laiMonth[m - 1])
        lai1 = lai0
        if m < 12:
            lai1 = self.laiMonth[m]

        # TO DO improve for month length
        self.currentLAI = lai0 + (lai1 - lai0) * (currentDate.day - 1) / 30


# global variables
currentCrop = CCrop()
rootDensity = []
k_root = np.array([], np.float64)
global SAT, FC, WP, HH, wsThreshold
global maxRootFactor, x, y


def initializeCrop():
    global k_root, rootDensity, maxRootFactor
    global SAT, FC, WP, HH, wsThreshold
    global x, y

    currentCrop.setMaxRootDepth()

    SAT = soil.horizon.thetaS  # [m3 m-3] water content at saturation
    FC = soil.getFieldCapacityWC()  # [m3 m-3] water content at field capacity
    WP = soil.getWiltingPointWC()  # [m3 m-3] water content at wilting point
    HH = soil.getHygroscopicWC()  # [m3 m-3] water content at Hygroscopic moisture
    wsThreshold = FC - currentCrop.fRAW * (FC - WP)  # [m3 m-3] water scarcity stress threshold

    # initialize root factor
    k_root = np.zeros(C3DStructure.nrRectangles)

    # distance from plant row (x-axis)
    max_distance = currentCrop.rootWidth * 0.5
    a = y[1] - y[0]
    b = x[0] - x[1]
    c = y[0] * (x[1] - x[0]) - x[0] * (y[1] - y[0])
    denominator = math.sqrt(a * a + b * b)

    for i in range(C3DStructure.nrRectangles):
        [x, y, z] = rectangularMesh.C3DRM[i].centroid
        line_distance = math.fabs(a * x + b * y + c) / denominator

        if line_distance < max_distance or math.fabs(line_distance - max_distance) < EPSILON:
            factor = 1.0 - line_distance / (max_distance * 0.5)
            k_root[i] = 1.0 + factor * currentCrop.rootXDeformation
        else:
            k_root[i] = 0.0

    # set root density
    rootDensity.clear()
    maxRootFactor = 0
    for i in range(C3DStructure.nrRectangles):
        rootDensity.append(computeRootDensity(currentCrop, C3DStructure.nrLayers, k_root[i]))
        # update max root factor
        for layer in range(1, C3DStructure.nrLayers):
            root_factor = k_root[i] * rootDensity[i][layer] / soil.thickness[layer]
            maxRootFactor = max(root_factor, maxRootFactor)


# fraction of intercepted photosynthetically active radiation [-]
def fPARi(currentLAI):
    ke = 0.6  # light extinction coefficient [-]
    if (currentLAI == NODATA) or (currentLAI <= 0):
        return 0.
    else:
        # Beer–Lambert Equation
        return 1. - math.exp(-ke * currentLAI)


def getMaxEvaporation(currentLAI, ET0):
    return ET0 * (1. - fPARi(currentLAI))


def getMaxTranspiration(currentLAI, kcMax, ET0):
    if (currentLAI == NODATA) or (currentLAI <= 0):
        return 0.
    else:
        fPAR = fPARi(currentLAI)
        TC = 1 + (kcMax - 1) * fPAR
        return ET0 * fPAR * TC


def cardioidDistribution(deformationFactor, nrLayersWithRoot):
    # initialize
    halfCircle = np.zeros(nrLayersWithRoot)
    cardioid = np.zeros(nrLayersWithRoot * 2)

    for i in range(nrLayersWithRoot):
        sinAlfa = 1.0 - float(i + 1.0) / float(nrLayersWithRoot)
        cosAlfa = max(math.sqrt(1.0 - math.pow(sinAlfa, 2.0)), EPSILON)
        alfa = math.atan(sinAlfa / cosAlfa)
        halfCircle[i] = ((math.pi / 2.0) - alfa - sinAlfa * cosAlfa) / math.pi

    lastLayer = nrLayersWithRoot * 2 - 1
    cardioid[0] = halfCircle[0]
    cardioid[lastLayer] = cardioid[0]
    for i in range(1, nrLayersWithRoot):
        cardioid[i] = halfCircle[i] - halfCircle[i - 1]
        cardioid[lastLayer - i] = cardioid[i]

    # cardioid deformation
    LiMin = -math.log(0.2) / float(nrLayersWithRoot)
    LiMax = -math.log(0.05) / float(nrLayersWithRoot)
    k = LiMin + (LiMax - LiMin) * (deformationFactor - 1.0)

    rootDensitySum = 0
    for i in range(nrLayersWithRoot * 2):
        cardioid[i] *= math.exp(-k * (i + 0.5))
        rootDensitySum += cardioid[i]

    # normalize
    for i in range(nrLayersWithRoot * 2):
        cardioid[i] /= rootDensitySum

    # assign layer density
    layerDensity = np.zeros(nrLayersWithRoot, np.float64)
    for i in range(nrLayersWithRoot):
        layerDensity[i] = cardioid[2 * i] + cardioid[2 * i + 1]

    return layerDensity


def computeRootDensity(crop, nrLayers, rootFactor):
    # initialize
    myRootDensity = np.zeros(nrLayers, np.float64)
    if crop.currentRootLength <= 0 or rootFactor == 0:
        return myRootDensity

    rootLength = crop.currentRootLength * min(1.0, math.sqrt(rootFactor))
    rootZero = crop.rootDepthZero
    if rootLength < 0.001:
        return myRootDensity

    # smallest unit of computation (1 mm)
    atoms = np.zeros(nrLayers, np.int16)
    for i in range(nrLayers):
        atoms[i] = int(round(soil.thickness[i] * 1000))

    nrUnrootedAtoms = int(round(rootZero * 1000))
    nrRootedAtoms = int(round(rootLength * 1000))
    densityAtoms = cardioidDistribution(crop.rootZDeformation, nrRootedAtoms)

    # assign root density
    counter = 0
    for layer in range(nrLayers):
        for i in range(atoms[layer]):
            if (counter >= nrUnrootedAtoms) and (counter < nrUnrootedAtoms + nrRootedAtoms):
                myRootDensity[layer] += densityAtoms[counter - nrUnrootedAtoms]
            counter += 1

    # check (rootDensitySum == 1)
    rootDensitySum = 0.0
    for i in range(nrLayers):
        rootDensitySum += myRootDensity[i]
    if abs(rootDensitySum - 1.0) > EPSILON:
        print("WARNING! Sum of root density:", rootDensitySum)

    return myRootDensity


# assign hourly transpiration
def setTranspiration(surfaceIndex, maxTranspiration, myRootDensity):
    if maxTranspiration < EPSILON:
        return 0, 0

    # Initialize
    rootDensityWithoutStress = 0.0  # [-]
    actualTranspiration = 0.0  # [mm]

    nrLayers = len(myRootDensity)
    isLayerStressed = np.zeros(nrLayers, dtype=bool)
    layerTranspiration = np.zeros(nrLayers, np.float64)  # [mm]

    for layer in range(nrLayers):
        if myRootDensity[layer] > 0:
            i = surfaceIndex + C3DStructure.nrRectangles * layer
            theta = soil.getVolumetricWaterContent(i)

            if theta > FC:
                # water surplus
                fraction = 1.0 - (theta - FC) / (SAT - FC)
                fraction = fraction**2
                layerTranspiration[layer] = maxTranspiration * myRootDensity[layer] * fraction
                isLayerStressed[layer] = True
            else:
                if theta <= wsThreshold:
                    # water scarcity
                    if theta <= WP:
                        layerTranspiration[layer] = 0.0
                    else:
                        fraction = (theta - WP) / (wsThreshold - WP)
                        fraction = fraction ** 2
                        layerTranspiration[layer] = maxTranspiration * myRootDensity[layer] * fraction
                    isLayerStressed[layer] = True
                else:
                    # normal conditions
                    layerTranspiration[layer] = maxTranspiration * myRootDensity[layer]
                    isLayerStressed[layer] = False
                    rootDensityWithoutStress += myRootDensity[layer]

            actualTranspiration += layerTranspiration[layer]

    # Assign transpiration flux [m3 s-1]
    for layer in range(nrLayers):
        if layerTranspiration[layer] > 0:
            i = surfaceIndex + C3DStructure.nrRectangles * layer
            rate = (layerTranspiration[layer] * 0.001) / 3600.0     # [m s-1]
            C3DCells[i].sinkSource -= rate * C3DCells[i].area       # [m3 s-1]

    return actualTranspiration, rootDensityWithoutStress


# water uptake compensation: redistribution in the moist zone
def setTranspRedistribution(surfaceIndex, redistribution, myRootDensity):
    # initialize
    nrLayers = len(myRootDensity)
    isLayerStressed = np.zeros(nrLayers, dtype=bool)
    rootDensityWithoutStress = 0.0                          # [-]
    actualRedistribution = 0.0                              # [mm]

    for layer in range(nrLayers):
        if myRootDensity[layer] > 0:
            i = surfaceIndex + C3DStructure.nrRectangles * layer
            theta = soil.getVolumetricWaterContent(i)

            if theta > FC:
                # water surplus
                isLayerStressed[layer] = True
            else:
                if theta <= wsThreshold:
                    # water scarcity
                    isLayerStressed[layer] = True
                else:
                    # normal conditions
                    isLayerStressed[layer] = False
                    rootDensityWithoutStress += myRootDensity[layer]

    if rootDensityWithoutStress > EPSILON:
        # assign redistribution flux [m3 s-1]
        for layer in range(nrLayers):
            if (myRootDensity[layer] > 0) and (not isLayerStressed[layer]):
                transp = redistribution * (myRootDensity[layer] / rootDensityWithoutStress)   # [mm]
                i = surfaceIndex + C3DStructure.nrRectangles * layer
                theta = soil.getVolumetricWaterContent(i)
                transpMax = max(0, theta - wsThreshold) * soil.thickness[layer] * 1000.       # [mm]
                layerTranspiration = min(transp, transpMax)                 # [mm]
                rate = (layerTranspiration * 0.001) / 3600.0                # [m s-1]
                C3DCells[i].sinkSource -= rate * C3DCells[i].area           # [m3 s-1]

                actualRedistribution += layerTranspiration                  # [mm]

    return actualRedistribution


def setEvaporation(surfaceIndex, maxEvaporation):
    # TODO: enable surface evaporation - numerical problem
    """
    surfaceWater = (C3DCells[surfaceIndex].H - C3DCells[surfaceIndex].z)        # [m]
    surfaceEvaporation = min(maxEvaporation, surfaceWater * 1000.0)             # [mm]
    rate = (surfaceEvaporation * 0.001) / 3600.0                                # [m s-1]
    C3DCells[surfaceIndex].sinkSource -= rate * C3DCells[surfaceIndex].area     # [m3 s-1]
    """
    surfaceEvaporation = 0
    actualEvaporation = surfaceEvaporation
    residualEvaporation = maxEvaporation - surfaceEvaporation

    if residualEvaporation < EPSILON:
        return actualEvaporation

    # soil evaporation
    half_FC = HH + (FC - HH) * 0.5
    lastIndex = 0
    while (lastIndex < len(soil.depth)) and (soil.depth[lastIndex] <= MAX_EVAPORATION_DEPTH):
        lastIndex += 1

    nrEvapLayers = lastIndex

    # depth coefficient: 1 at first layer, ~0.1 at MAX_EVAPORATION_DEPTH
    coeffEvap = np.zeros(nrEvapLayers, np.float64)
    sumCoefficient = 0
    for i in range(1, nrEvapLayers):
        coeffDepth = max((soil.depth[i] - soil.depth[1]) / (MAX_EVAPORATION_DEPTH - soil.depth[1]), 0)
        coeffEvap[i] = math.exp(-coeffDepth * math.e)
        sumCoefficient += (coeffEvap[i] * soil.thickness[i])

    isWaterSupply = True
    while (residualEvaporation > EPSILON) and (isWaterSupply is True):
        isWaterSupply = False
        sumEvaporation = 0.0

        for layer in range(1, nrEvapLayers):
            index = surfaceIndex + C3DStructure.nrRectangles * layer
            theta = soil.getVolumetricWaterContent(index)  # [m3 m-3]
            evaporationThreshold = half_FC - coeffEvap[layer] * (half_FC - HH)  # [m3 m-3]
            evaporation = residualEvaporation * ((coeffEvap[layer]
                                                  * soil.thickness[layer]) / sumCoefficient)  # [mm]
            if theta < evaporationThreshold:
                evaporation = 0.0
            else:
                availableWC = (theta - evaporationThreshold) * soil.thickness[layer] * 1000.  # [mm]
                if availableWC <= evaporation:
                    evaporation = availableWC
                else:
                    isWaterSupply = True

            sumEvaporation += evaporation

            rate = (evaporation * 0.001) / 3600.  # [m s-1]
            C3DCells[index].sinkSource -= rate * C3DCells[index].area  # [m3 s-1]

        residualEvaporation -= sumEvaporation
        actualEvaporation += sumEvaporation

    return actualEvaporation


def setEvapotranspiration(currentDate, ET0):
    currentCrop.setCurrentLAI(currentDate)
    if C3DParameters.computeTranspiration:
        maxTranspiration = getMaxTranspiration(currentCrop.currentLAI, currentCrop.kcMax, ET0)
        sumTranspiration = 0.0
        sumMaxTranspiration = 0.0
        nrCells = 0
        sumMoistRoot = 0.0
        rootDensityWithoutStress = np.zeros(C3DStructure.nrRectangles, np.float64)
        for i in range(C3DStructure.nrRectangles):
            if k_root[i] > 0.0:
                maxTranspirationLayer = maxTranspiration * k_root[i]
                sumMaxTranspiration += maxTranspirationLayer
                actualTranspiration, availableRootDensity = setTranspiration(i, maxTranspirationLayer, rootDensity[i])
                rootDensityWithoutStress[i] = availableRootDensity
                sumTranspiration += actualTranspiration
                sumMoistRoot += availableRootDensity * k_root[i]
                nrCells += 1

        if sumMoistRoot > 0.0:
            # water uptake compensation: redistribution in the moist zone
            missingTranspiration = (sumMaxTranspiration - sumTranspiration) / sumMoistRoot
            if missingTranspiration > 0.01:
                for i in range(C3DStructure.nrRectangles):
                    if rootDensityWithoutStress[i] > 0.01:
                        maxTranspirationLayer = maxTranspiration * k_root[i]
                        redistribution = missingTranspiration * rootDensityWithoutStress[i] * k_root[i]
                        redistribution = min(redistribution, maxTranspirationLayer)
                        layerRedistribution = setTranspRedistribution(i, redistribution, rootDensity[i])
                        sumTranspiration += layerRedistribution

        maxTranspiration = sumMaxTranspiration / nrCells
        actualTranspiration = sumTranspiration / nrCells
    else:
        actualTranspiration = 0
        maxTranspiration = 0

    if C3DParameters.computeEvaporation:
        maxEvaporation = getMaxEvaporation(currentCrop.currentLAI, ET0)
        sumEvaporation = 0
        for i in range(C3DStructure.nrRectangles):
            actualEvaporation = setEvaporation(i, maxEvaporation)
            sumEvaporation += actualEvaporation
        actualEvaporation = sumEvaporation / C3DStructure.nrRectangles
    else:
        maxEvaporation = 0
        actualEvaporation = 0

    return maxTranspiration, maxEvaporation, actualTranspiration, actualEvaporation
