import pandas as pd

import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

import os

global case
case = "errano_all_evaluation_errano_2022_v6.1"

def ground_potential(
    evaluation_folder=os.path.join("data", case),
    output_folder=os.path.join("plots"),
):
    df = pd.concat([pd.DataFrame({"timestamp": [1655244000]}), pd.read_csv(os.path.join(evaluation_folder, "output", "output.csv")), pd.DataFrame({"timestamp": [1661983200]})], ignore_index=True)

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    # df["timestamp"] = df["timestamp"].dt.strftime('%Y-%m-%d')
    df = df.set_index("timestamp")
    df = df.reindex(sorted(df.columns), axis=1)
    df = df.interpolate(method="linear", limit_direction="forward", axis=0)
    # print(df)

    # define subplot layout
    nrows, ncols = 3, 4
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols)

    # add DataFrames to subplots

    for idx, meteo_var in reversed(list(enumerate(df.columns))):
        ax = axes[int(idx / ncols), idx % ncols]
        df[meteo_var].plot(ax=ax, sharex=True, sharey=True, color="C5")
        ax.set_ylim([df.min().min(), -10])
        ax.set_yscale("symlog")
        ax.set_xlabel("")

        # ax.set_xticks([0,  df.shape[0] - 1])
        # if meteo_var == "z60_y0_x80":
        #     xticks = ax.get_xticks()
        # ax.set_xticks(xticks)

        # ax.set_xticks([0, df.shape[0]])
        if idx < 8:
            ax.tick_params(length=0)
        ax.set_ylabel("cbar",  fontsize=12)
        ax.tick_params(axis="both", labelsize=12)
        ax.set_title(
            meteo_var.replace("y0_", "")
            .replace("_", " cm, ")
            .replace("z", "Depth = ")
            .replace("x", "Distance = ")
            + " cm",
            fontsize=15,
        )

    # _ = plt.xticks(rotation=0)
    fig.set_size_inches(18, 6)
    plt.tight_layout()
    fig.savefig(os.path.join(output_folder, f"ground_potential_{case}.pdf"))
    fig.savefig(os.path.join(output_folder, f"ground_potential_{case}.png"))

def main():
    ground_potential()


if __name__ == "__main__":
    main()
