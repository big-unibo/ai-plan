# waterBalance.py

from math import fabs
from dataStructures import *
from soil import getVolumetricWaterContent


class C3DBalance:
    waterStorage = NODATA  # [m3]
    waterFlow = NODATA  # [m3]
    MBE = NODATA  # [m3] mass balance error
    MBR = NODATA  # [-] mass balance ratio

    def initialize(self):
        self.waterStorage = 0
        self.waterFlow = 0
        self.MBE = 0
        self.MBR = 0


# all variables are in [mm]
class C3DDailyWaterBalance:
    precipitation = NODATA
    irrigation = NODATA
    et0 = NODATA
    maxTranspiration = NODATA
    maxEvaporation = NODATA
    actualTranspiration = NODATA
    actualEvaporation = NODATA
    drainage = NODATA
    runoff = NODATA

    def initialize(self):
        self.precipitation = 0
        self.irrigation = 0
        self.et0 = 0
        self.maxTranspiration = 0
        self.maxEvaporation = 0
        self.actualTranspiration = 0
        self.actualEvaporation = 0
        self.drainage = 0
        self.runoff = 0


totalTime = 0.0
currentPrec = 0.0
currentIrr = 0.0
MBRMultiply: float = 1.0
maxCourant = 0.0
bestMBR = NODATA
nrMBRWrong = 0
forceExit = False
currentStep = C3DBalance()
previousStep = C3DBalance()
allSimulation = C3DBalance()
dailyBalance = C3DDailyWaterBalance()


def doubleTimeStep():
    global MBRMultiply
    if (C3DParameters.currentDeltaT == C3DParameters.deltaT_min) and (MBRMultiply > 1.0):
        decMBRThreshold()
    else:
        C3DParameters.currentDeltaT = min(C3DParameters.currentDeltaT * 2.0,
                                          C3DParameters.currentDeltaT_max)


def halveTimeStep():
    if C3DParameters.currentDeltaT == C3DParameters.deltaT_min:
        if C3DParameters.MBRThreshold < 1E-5:
            incMBRThreshold()
        else:
            return False
    else:
        C3DParameters.currentDeltaT = max(C3DParameters.currentDeltaT * 0.5,
                                          C3DParameters.deltaT_min)
    return True


def incMBRThreshold():
    global MBRMultiply
    MBRMultiply *= 2.0
    C3DParameters.MBRThreshold *= 2.0


def decMBRThreshold():
    global MBRMultiply
    if MBRMultiply > 1.0:
        MBRMultiply *= 0.5
        C3DParameters.MBRThreshold *= 0.5


def initializeBalance():
    global totalTime
    totalTime = 0.0

    storage = getWaterStorage()
    currentStep.initialize()
    currentStep.waterStorage = storage
    previousStep.initialize()
    previousStep.waterStorage = storage
    allSimulation.initialize()
    allSimulation.waterStorage = storage

    dailyBalance.initialize()


def updateStorage():
    storage = getWaterStorage()
    currentStep.waterStorage = storage
    previousStep.waterStorage = storage
    allSimulation.waterStorage = storage


def updateBalance(deltaT):
    global totalTime
    totalTime += deltaT
    # set previous step
    previousStep.waterStorage = currentStep.waterStorage
    previousStep.waterFlow = currentStep.waterFlow
    # update all simulation flow
    allSimulation.waterFlow += currentStep.waterFlow
    allSimulation.MBE += currentStep.MBE
    # update drainage in daily balance [mm]
    drainage = 1000. * sumDrainage(deltaT) / C3DStructure.totalArea   # [mm]
    dailyBalance.drainage += drainage
    # TODO: update runoff


# return water storage [m3]
def getWaterStorage():
    waterStorage = 0.0
    for i in range(C3DStructure.nrCells):
        if C3DCells[i].isSurface:
            if abs(C3DCells[i].H - C3DCells[i].z) > 0:
                waterStorage += (C3DCells[i].H - C3DCells[i].z) * C3DCells[i].area
        else:
            waterStorage += (getVolumetricWaterContent(i) * C3DCells[i].volume)
    return waterStorage


# sum of boundary water flows (only drainage) during the time step [m3]
def sumDrainage(deltaT):
    mySum = 0.0
    for i in range(C3DStructure.nrCells):
        if C3DCells[i].boundary.type == BOUNDARY_FREEDRAINAGE or C3DCells[i].boundary.type == BOUNDARY_FREELATERALDRAINAGE:
            if C3DCells[i].boundary.flow != NODATA:
                mySum += C3DCells[i].boundary.flow * deltaT
    return mySum


# sum of boundary water flows (all types) during the time step [m3]
def sumBoundaryFlow(deltaT):
    mySum = 0.0
    for i in range(C3DStructure.nrCells):
        if C3DCells[i].boundary.type != BOUNDARY_NONE:
            if C3DCells[i].boundary.flow != NODATA:
                mySum += C3DCells[i].boundary.flow * deltaT
    return mySum


# sum of all water sink/source during the time step [m3]
def sumSinkSource(deltaT):
    mySum = 0.0
    for i in range(C3DStructure.nrCells):
        if C3DCells[i].sinkSource != NODATA:
            mySum += C3DCells[i].sinkSource * deltaT
    return mySum


# sum of all water flows (it includes sink/source and boundary flows) during the time step [m3]
def sumWaterFlow(deltaT, isAbsoluteValue):
    mySum = 0.0
    for i in range(C3DStructure.nrCells):
        if C3DCells[i].flow != NODATA:
            if isAbsoluteValue:
                mySum += abs(C3DCells[i].flow * deltaT)
            else:
                mySum += C3DCells[i].flow * deltaT
    return mySum


def computeBalanceError(deltaT):
    currentStep.waterStorage = getWaterStorage()
    currentStep.waterFlow = sumWaterFlow(deltaT, False)
    currentStep.MBE = currentStep.waterStorage - (previousStep.waterStorage + currentStep.waterFlow)
    if previousStep.waterStorage > 0:
        currentStep.MBR = fabs(currentStep.MBE) / previousStep.waterStorage
    else:
        currentStep.MBR = fabs(currentStep.MBE)
    # print ("Mass Balance Error [l]:", format(currentStep.MBE * 1000,".5f"))
    # print("Mass Balance Ratio:", format(currentStep.MBR, ".5f"))


def waterBalance(deltaT, approximation):
    global forceExit, bestMBR, nrMBRWrong
    computeBalanceError(deltaT)

    if approximation == 1:
        bestMBR = currentStep.MBR
        nrMBRWrong = 0
        forceExit = False

    # case 1: step accepted
    if currentStep.MBR < C3DParameters.MBRThreshold:
        updateBalance(deltaT)
        if approximation < 3 and maxCourant < 0.3 and currentStep.MBR < (C3DParameters.MBRThreshold * 0.5) \
                and C3DParameters.currentDeltaT < C3DParameters.currentDeltaT_max:
            # print("Good MBR!")
            doubleTimeStep()
        return True

    # case 2: continue with next approximation
    if approximation == 1 or currentStep.MBR < bestMBR:
        bestMBR = currentStep.MBR
    else:
        nrMBRWrong += 1

    # case 3: decrease time step (or increase threshold)
    isLastApprox = (approximation == C3DParameters.maxApproximationsNr)
    if isLastApprox or nrMBRWrong > 0:
        if halveTimeStep():
            forceExit = True
        else:
            # accept error
            updateBalance(deltaT)
            return True

    return False
