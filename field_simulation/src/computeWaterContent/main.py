import os
import soil
import pandas as pd


def main():
    print(os.getcwd())
    dataPath = os.path.join("data", "errano_all")
    settingsFolder = os.path.join(dataPath, "settings")

    print("Load soil...")
    soilFile = "soil.csv"
    soilPath = os.path.join(settingsFolder, soilFile)
    soil.readHorizon(soilPath)

    print("Load obs water potential...")
    obsPath = os.path.join(dataPath, "obs_data")
    obsFile = "waterPotential.csv"
    obsData = pd.read_csv(os.path.join(obsPath, obsFile))

    print("Write obs water content...")
    outputFileWC = os.path.join(obsPath, "waterContent.csv")

    if not os.path.exists(outputFileWC):
        open(outputFileWC, 'w').close()

    # header
    header = obsData.columns[0]
    for i in range(1, len(obsData.columns)):
        header += "," + obsData.columns[i]
    header += "\n"
    f = open(outputFileWC, "w")
    f.write(header)

    # data
    for i in range(len(obsData)):
        row = str(obsData.iloc[i].at["timestamp"])
        for j in range(1, len(obsData.columns)):
            # water potential - from [kPa] to [m]
            psi = obsData.iloc[i].at[obsData.columns[j]] / 9.81
            # water content
            theta = soil.thetaFromPsi(psi)
            row += "," + '{:.4f}'.format(theta)
        row += "\n"
        f.write(row)

    print("\nEnd.\n")


main()
