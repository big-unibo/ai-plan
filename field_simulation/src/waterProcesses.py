# waterProcesses.py

from math import fabs, sqrt
from dataStructures import *
import waterBalance
import soil


if CYTHON:
    from solverC import meanK
else:
    from soil import meanK


def redistribution(i, link, isLateral):
    j = link.index
    k = meanK(C3DParameters.conductivityMean, C3DCells[i].k, C3DCells[j].k)
    if isLateral:
        k *= C3DParameters.conductivityHVRatio

    return (k * link.area) / link.distance


def infiltration(surf, sub, link, deltaT, isFirstApprox):
    # unsaturated
    if C3DCells[surf].z > C3DCells[sub].H:
        # surface water content[m]
        avgH = (C3DCells[surf].H + C3DCells[surf].H0) * 0.5
        psi = avgH - C3DCells[surf].z
        if isFirstApprox:
            rain = (C3DCells[surf].sinkSource / C3DCells[surf].area) * (deltaT * 0.5)
            psi += rain
        if psi < EPSILON:
            return 0.0

        # first soil layer: mean between current conductivity and k_sat
        interfaceK = meanK(C3DParameters.conductivityMean, C3DCells[sub].k, soil.horizon.Ks)
        dH = C3DCells[surf].H - C3DCells[sub].H
        maxK = (psi / deltaT) * (link.distance / dH)
        k = min(interfaceK, maxK)
    else:
        # saturated
        k = soil.horizon.Ks
    
    return (k * link.area) / link.distance


def runoff(i, link, deltaT, isFirstApprox):
    j = link.index
    dH = fabs(C3DCells[i].H - C3DCells[j].H)
    if dH < EPSILON_METER:
        return 0.

    maxZ = max(C3DCells[i].z, C3DCells[j].z)
    # maxH = max(C3DCells[i].H, C3DCells[j].H)
    maxH = max((C3DCells[i].H + C3DCells[i].H0) * 0.5, (C3DCells[j].H + C3DCells[j].H0) * 0.5)
    Hs = maxH - (maxZ + C3DParameters.pond)
    if Hs <= EPSILON_METER:
        return 0.
    # pond
    Hs = min(Hs, dH)
    # [m/s] Manning equation
    v = (pow(Hs, 2.0 / 3.0) * sqrt(dH/link.distance)) / C3DParameters.roughness
    Courant = v * deltaT / link.distance
    waterBalance.maxCourant = max(waterBalance.maxCourant, Courant)

    # link.area on surface = side length [m]
    area = link.area * Hs 
    return (v / dH) * area
