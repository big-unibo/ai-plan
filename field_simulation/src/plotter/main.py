import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
from sklearn.metrics import mean_squared_error, r2_score
import warnings

warnings.filterwarnings("ignore")

import os
import time

forbidden_sensors = ["z20_y0_x0", "z20_y0_x25", "z40_y0_x0", "z40_y0_x50"]


def meteo(
    tuning_folder=os.path.join("data", "errano_tuning"),
    evaluation_folder=os.path.join("data", "errano_evaluation_1gg"),
    output_folder=os.path.join("plots"),
):
    meteo_dict = {}
    meteo_vars = {
        "air_humidity": "%",
        "air_temperature": "°C",
        "solar_radiation": "W/mq",
        "wind_speed": "m/s",
    }

    for meteo_var in meteo_vars.keys():
        meteo_dict[meteo_var] = (
            pd.read_csv(os.path.join(tuning_folder, "meteo", f"{meteo_var}.csv"))
            .append(
                pd.read_csv(
                    os.path.join(evaluation_folder, "meteo", f"{meteo_var}.csv")
                ),
                ignore_index=True,
            )
            .set_index("timestamp")
        )

    df = pd.concat(meteo_dict.values(), axis=1)
    df = df.reset_index()
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    # df["timestamp"] = df["timestamp"].dt.strftime('%Y-%m-%d')
    df = df.set_index("timestamp")

    # print(df)

    # define subplot layout
    nrows, ncols = 2, 2
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols)

    # add DataFrames to subplots

    for idx, meteo_var in reversed(list(enumerate(meteo_vars.keys()))):
        df[meteo_var].plot(
            ax=axes[int(idx / nrows), idx % nrows], sharex=True, color="C7"
        )
        # axes[int(idx / nrows), idx % nrows].set_xlim(
        #     [0, df.shape[0]]
        # )
        axes[int(idx / nrows), idx % nrows].set_ylim(
            [0, 1000]
            if meteo_var == "solar_radiation"
            else (
                [0, 15]
                if meteo_var == "wind_speed"
                else ([0, 100] if meteo_var == "air_humidity" else [0, 40])
            )
        )
        axes[int(idx / nrows), idx % nrows].set_xlabel("")
        if meteo_var == "wind_speed":
            xticks = axes[int(idx / nrows), idx % nrows].get_xticks()
        axes[int(idx / nrows), idx % nrows].set_xticks(xticks)

        # axes[int(idx / nrows), idx % nrows].set_xticks([0, df.shape[0]])
        if idx < 2:
            axes[int(idx / nrows), idx % nrows].tick_params(length=0)
        axes[int(idx / nrows), idx % nrows].set_ylabel(
            meteo_vars[meteo_var], fontsize=12
        )
        axes[int(idx / nrows), idx % nrows].set_title(
            meteo_var.replace("_", " ").capitalize(), fontsize=15
        )
        axes[int(idx / nrows), idx % nrows].tick_params(axis="both", labelsize=12)

    # _ = plt.xticks(rotation=0)
    plt.tight_layout()
    fig.set_size_inches(13, 6)
    fig.savefig(os.path.join(output_folder, "meteo.pdf"))
    fig.savefig(os.path.join(output_folder, "meteo.png"))


def water(
    tuning_folder=os.path.join("data", "errano_tuning"),
    evaluation_folder=os.path.join("data", "errano_evaluation_1gg"),
    output_folder=os.path.join("plots"),
):
    water_dict = {}
    water_vars = {
        "irrigation": "L",
        "precipitation": "mm",
    }

    for meteo_var in water_vars.keys():
        water_dict[meteo_var] = (
            pd.DataFrame({"timestamp": [1655251200], meteo_var: [0.0]})
            .append(
                pd.read_csv(os.path.join(tuning_folder, "water", f"{meteo_var}.csv")),
                ignore_index=True,
            )
            .append(
                pd.read_csv(
                    os.path.join(evaluation_folder, "water", f"{meteo_var}.csv")
                ),
                ignore_index=True,
            )
            .append(
                pd.DataFrame({"timestamp": [1661990400], meteo_var: [0.0]}),
                ignore_index=True,
            )
            .set_index("timestamp")
        )
    df = pd.concat(water_dict.values(), axis=1)
    df = df.reset_index()
    df["timestamp"] /= 3600 * 24
    df["timestamp"] = df["timestamp"].astype(int)
    df["timestamp"] *= 3600 * 24
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%d")
    df = df.rename(
        columns={
            "precipitation": "Precipitation in mm",
            "irrigation": "Irrigation in L",
        }
    )

    df = df.groupby("timestamp").sum()

    fig, ax = plt.subplots()
    df.plot(ax=ax, kind="bar", color=["C9", "C3"])
    ax.set_xlabel("")
    ax.set_xticks([0, 15, 29, 46, 60, df.shape[0] - 1])
    # ax.set_title("Precipitation and irrigation", fontdict={"fontsize": 18})
    ax2 = ax.twinx()
    ax.set_ylabel("L",  fontsize=15)
    ax2.set_ylabel("mm",  fontsize=15)
    ax2.set_ylim(0, 30)
    ax.set_ylim(0, 30)

    fig = ax.get_figure()
    fig.autofmt_xdate()

    plt.tight_layout()
    fig.set_size_inches(13, 6)
    fig.savefig(os.path.join(output_folder, "water.pdf"))
    fig.savefig(os.path.join(output_folder, "water.png"))


def ground_potential(
    tuning_folder=os.path.join("data", "errano_tuning"),
    evaluation_folder=os.path.join("data", "errano_evaluation_1gg"),
    output_folder=os.path.join("plots"),
):
    df = (
        pd.DataFrame({"timestamp": [1655251200]})
        .append(
            pd.read_csv(os.path.join(tuning_folder, "obs_data", "waterPotential.csv")),
            ignore_index=True,
        )
        .append(
            pd.read_csv(
                os.path.join(evaluation_folder, "obs_data", "waterPotential.csv")
            ),
            ignore_index=True,
        )
        .append(
            pd.DataFrame({"timestamp": [1661990400]}),
            ignore_index=True,
        )
    )

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
    fig.savefig(os.path.join(output_folder, "ground_potential.pdf"))
    fig.savefig(os.path.join(output_folder, "ground_potential.png"))


def forecast_avg(
    obs_folder=os.path.join("data", "errano_evaluation_1gg"),
    forecast_folder=os.path.join("data", "errano_evaluation"),
    output_folder=os.path.join("plots"),
    with_forbidden_sensors=True,
):
    support_dict = {
        "obs": os.path.join(obs_folder, "obs_data", "waterPotential.csv"),
    }
    for forecasting_day in ["1gg", "3gg", "7gg"]:
        support_dict[forecasting_day] = os.path.join(
            f"{forecast_folder}_{forecasting_day}", "output", "output.csv"
        )

    forecasting_dict = {}
    for data_type, input_path in support_dict.items():
        forecasting_dict[data_type] = (
            pd.DataFrame({"timestamp": [1655251200]})
            .append(
                pd.read_csv(input_path),
                ignore_index=True,
            )
            .set_index("timestamp")
            .drop(columns=[] if with_forbidden_sensors else forbidden_sensors)
        )
        forecasting_dict[data_type] *= -1
        forecasting_dict[data_type][forecasting_dict[data_type] < 20] = 20
        forecasting_dict[data_type] = forecasting_dict[data_type].apply(
            lambda x: np.log(x)
        )
        forecasting_dict[data_type] = forecasting_dict[data_type].reindex(
            sorted(forecasting_dict[data_type].columns), axis=1
        )
        forecasting_dict[data_type] = forecasting_dict[data_type].add_suffix(
            f"_{data_type}"
        )
    df = pd.concat(forecasting_dict.values(), axis=1)
    df = df.interpolate(method="linear", limit_direction="forward", axis=0)
    df = df.dropna(axis="index")
    df = df.reset_index()
    new_columns = []
    for data_type in support_dict.keys():
        if data_type != "obs":
            new_column = f"RMSE_{data_type}"
            new_columns += [new_column]
            # print(df[[c for c in df.columns if c.endswith("_obs")]].iloc[0])
            # print(df[[c for c in df.columns if c.endswith("_obs")]].iloc[:1])
            # print(df[[c for c in df.columns if c.endswith(f"_{data_type}")]].iloc[0])
            # print(df[[c for c in df.columns if c.endswith(f"_{data_type}")]].iloc[:1])
            df[new_column] = [
                mean_squared_error(
                    df[[c for c in df.columns if c.endswith("_obs")]].iloc[i : (i + 1)],
                    df[[c for c in df.columns if c.endswith(f"_{data_type}")]].iloc[
                        i : (i + 1)
                    ],
                    squared=False,
                )
                for i in range(df.shape[0])
            ]
    df = df.append(pd.DataFrame({"timestamp": [1655251200]}), ignore_index=True)
    df = df.set_index("timestamp")
    df = df[new_columns]
    df.index = pd.to_datetime(df.index, unit="s")

    fig, ax = plt.subplots()
    df = df.rename(
        columns={
            column: column.replace("RMSE_", "").replace("gg", "-day")
            + ("s" if column != "RMSE_1gg" else "")
            + " horizon"
            for column in df.columns
        }
    )

    df.plot(ax=ax)
    ax.set_ylim([0, 1.5])
    ax.set_xlabel("")
    ax.set_ylabel("Error",  fontsize=12)
    ax.tick_params(axis="both", labelsize=12)
    fig.set_size_inches(13, 6)
    plt.legend(fontsize=12)
    plt.tight_layout()
    is_forbidden_sensors_string = (
        "_with_forbidden_sensors" if with_forbidden_sensors else ""
    )
    fig.savefig(
        os.path.join(output_folder, f"forecasting_avg{is_forbidden_sensors_string}.pdf")
    )
    fig.savefig(
        os.path.join(output_folder, f"forecasting_avg{is_forbidden_sensors_string}.png")
    )


def forecast_sensor(
    obs_folder=os.path.join("data", "errano_evaluation_1gg"),
    forecast_folder=os.path.join("data", "errano_evaluation"),
    output_folder=os.path.join("plots"),
    with_predicted=True,
):
    support_dict = {
        "obs": os.path.join(obs_folder, "obs_data", "waterPotential.csv"),
    }
    for forecasting_day in ["1gg", "3gg", "7gg"]:
        support_dict[forecasting_day] = os.path.join(
            f"{forecast_folder}_{forecasting_day}", "output", "output.csv"
        )

    forecasting_dict = {}
    for data_type, input_path in support_dict.items():
        forecasting_dict[data_type] = (
            pd.DataFrame({"timestamp": [1655251200]})
            .append(
                pd.read_csv(input_path),
                ignore_index=True,
            )
            .set_index("timestamp")
            # .drop(columns=forbidden_sensors)
        )
        if not with_predicted:
            forecasting_dict[data_type] *= -1
            forecasting_dict[data_type][forecasting_dict[data_type] < 20] = 20
            forecasting_dict[data_type] = forecasting_dict[data_type].apply(
                lambda x: np.log(x)
            )
        else:
            forecasting_dict[data_type][forecasting_dict[data_type] > -20] = -20

        forecasting_dict[data_type] = forecasting_dict[data_type].reindex(
            sorted(forecasting_dict[data_type].columns), axis=1
        )
        forecasting_dict[data_type] = forecasting_dict[data_type].add_suffix(
            f"_{data_type}"
        )
    for forecasting_day in ["1gg", "3gg", "7gg"]:
        df = pd.concat(
            [forecasting_dict["obs"], forecasting_dict[forecasting_day]], axis=1
        )
        df = df.interpolate(method="linear", limit_direction="forward", axis=0)
        df = df.dropna(axis="index")
        df = df.reset_index()
        sensor_columns = [
            c.replace("_obs", "") for c in df.columns if c.endswith("_obs")
        ]
        if not with_predicted:
            new_columns = []
            for column in sensor_columns:
                new_column = f"RMSE_{column}"
                new_columns += [new_column]
                df[new_column] = [
                    mean_squared_error(
                        df[column + "_obs"].iloc[i : (i + 1)],
                        df[column + f"_{forecasting_day}"].iloc[i : (i + 1)],
                        squared=False,
                    )
                    for i in range(df.shape[0])
                ]
        df = df.append(pd.DataFrame({"timestamp": [1655251200]}), ignore_index=True)
        df = df.set_index("timestamp")
        if not with_predicted:
            df = df[new_columns]
        df.index = pd.to_datetime(df.index, unit="s")

        # define subplot layout
        nrows, ncols = 3, 4
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols)

        for idx, meteo_var in reversed(
            list(enumerate(sensor_columns if with_predicted else df.columns))
        ):
            ax = axes[int(idx / ncols), idx % ncols]
            columns = (
                [meteo_var + suffix for suffix in ["_obs", f"_{forecasting_day}"]]
                if with_predicted
                else meteo_var
            )
            df[columns].plot(ax=ax, sharex=True)
            ax.set_xlabel("")
            if not with_predicted:
                ax.set_ylim([0, 4])
                ax.set_ylabel("logRMSE")
            else:
                ax.set_ylim([df.min().min(), -10])
                ax.set_yscale("symlog")
                ax.set_ylabel("cbar")

            if idx < 8:
                ax.tick_params(length=0)
            ax.set_title(
                meteo_var.replace("RMSE_", "")
                .replace("y0", "")
                .replace("_", " ")
                .replace("z", "depth = ")
                .replace("x", "distance = "),
                color="red"
                if meteo_var.replace("RMSE_", "") in forbidden_sensors
                else "black",
            )

        fig.set_size_inches(18, 6)
        plt.tight_layout()
        suffix = "_with_predicted" if with_predicted else ""
        fig.savefig(
            os.path.join(
                output_folder, f"forecasting_sensor_{forecasting_day}{suffix}.pdf"
            )
        )
        fig.savefig(
            os.path.join(
                output_folder, f"forecasting_sensor_{forecasting_day}{suffix}.png"
            )
        )


def forecast_std(
    obs_folder=os.path.join("data", "errano_evaluation_1gg"),
    forecast_folder=os.path.join("data", "errano_evaluation"),
    output_folder=os.path.join("plots"),
    with_forbidden_sensors=True,
):
    support_dict = {
        "obs": os.path.join(obs_folder, "obs_data", "waterPotential.csv"),
    }
    for forecasting_day in ["1gg", "3gg", "7gg"]:
        support_dict[forecasting_day] = os.path.join(
            f"{forecast_folder}_{forecasting_day}", "output", "output.csv"
        )

    forecasting_dict = {}
    for data_type, input_path in support_dict.items():
        forecasting_dict[data_type] = (
            pd.DataFrame({"timestamp": [1655251200]})
            .append(
                pd.read_csv(input_path),
                ignore_index=True,
            )
            .set_index("timestamp")
            .drop(columns=[] if with_forbidden_sensors else forbidden_sensors)
        )
        forecasting_dict[data_type][forecasting_dict[data_type] > -20] = -20
        forecasting_dict[data_type] = forecasting_dict[data_type].reindex(
            sorted(forecasting_dict[data_type].columns), axis=1
        )
        forecasting_dict[data_type] = forecasting_dict[data_type].add_suffix(
            f"_{data_type}"
        )
    df = pd.concat(forecasting_dict.values(), axis=1)
    df = df.interpolate(method="linear", limit_direction="forward", axis=0)
    df = df.dropna(axis="index")
    df = df.reset_index()
    new_columns = []
    for data_type in support_dict.keys():
        if data_type != "obs":
            new_column = f"RMSE_{data_type}"
            new_columns += [new_column]
            # print(df[[c for c in df.columns if c.endswith("_obs")]].iloc[0])
            # print(df[[c for c in df.columns if c.endswith("_obs")]].iloc[:1])
            # print(df[[c for c in df.columns if c.endswith(f"_{data_type}")]].iloc[0])
            # print(df[[c for c in df.columns if c.endswith(f"_{data_type}")]].iloc[:1])
            df[new_column] = [
                mean_squared_error(
                    df[[c for c in df.columns if c.endswith("_obs")]].iloc[i : (i + 1)],
                    df[[c for c in df.columns if c.endswith(f"_{data_type}")]].iloc[
                        i : (i + 1)
                    ],
                    squared=False,
                )
                for i in range(df.shape[0])
            ]
    df["average"] = df[[c for c in df.columns if c.endswith("_obs")]].mean(axis=1)
    df = df.append(pd.DataFrame({"timestamp": [1655251200]}), ignore_index=True)
    df = df.set_index("timestamp")
    # df = df[new_columns]
    df.index = pd.to_datetime(df.index, unit="s")

    fig, ax = plt.subplots()
    # df = df.rename(
    #     columns={
    #         column: column.replace("RMSE_", "forecasting horizon = ")
    #         for column in df.columns
    #     }
    # )

    df["average"].plot(ax=ax, color="C5", label="observed")
    # ax.set_ylim([-800, 0])
    ax.fill_between(
        df.index,
        df["average"] - df["RMSE_7gg"],
        df["average"] + df["RMSE_7gg"],
        alpha=0.2,
        color="C2",
        label="7-days horizon",
    )
    ax.fill_between(
        df.index,
        df["average"] - df["RMSE_3gg"],
        df["average"] + df["RMSE_3gg"],
        alpha=0.4,
        color="C1",
        label="3-days horizon",
    )
    ax.fill_between(
        df.index,
        df["average"] - df["RMSE_1gg"],
        df["average"] + df["RMSE_1gg"],
        alpha=0.6,
        color="C0",
        label="1-day horizon",
    )
    ax.legend()
    ax.set_xlabel("")
    ax.set_ylabel("cbar",  fontsize=12)
    ax.tick_params(axis="both", labelsize=12)
    fig.set_size_inches(13, 6)
    plt.legend(fontsize=12)
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [0,3,2,1]
    plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order])
    plt.tight_layout()
    is_forbidden_sensors_string = (
        "_with_forbidden_sensors" if with_forbidden_sensors else ""
    )
    fig.savefig(
        os.path.join(output_folder, f"forecasting_std{is_forbidden_sensors_string}.pdf")
    )
    fig.savefig(
        os.path.join(output_folder, f"forecasting_std{is_forbidden_sensors_string}.png")
    )
    return df


def water_balance(
    input_folder=os.path.join("data", "errano_all"),
    output_folder=os.path.join("plots"),
):
    df = pd.read_csv(os.path.join(input_folder, "output", "dailyWaterBalance.csv"))
    df["date"] = pd.to_datetime(df["date"], infer_datetime_format=True)
    # df["date"] = df["date"].apply(
    #     lambda x: int(datetime.datetime.strptime(x, "%Y-%m-%d").timestamp())
    # )
    df = df.set_index("date")
    # df.index = pd.to_datetime(df.index, unit="s")
    df = df[df.index.to_series().between("2022-06-15", "2022-09-1")]
    df.index = df.index.astype(str)

    fig, ax = plt.subplots()
    # ax2 = ax.twinx()
    df = df.rename(
        columns={
            "maxTranpiration": "max tranpirat.",
            "ET0": "actual transpirat.",
            "realvaporation": "evaporation",
        }
    )
    df[["precipitation", "irrigation", "drainage"]].plot(
        ax=ax, kind="bar", color=["C9", "C3", "C6"]
    )  # , color=['C9', 'C3'])
    df[["max tranpirat.", "actual transpirat.", "evaporation"]].plot(
        ax=ax, kind="line", color=["black", "C8", "b"]
    )  # , color=['C9', 'C3'])
    plt.text(64, 20, 35, ha="center", bbox=dict(facecolor="white", edgecolor="white"))
    # ax.bar(df.index, df["precipitation"], label='precipitation')
    # ax.bar(df.index, df["irrigation"], label='irrigation')
    # ax.bar(df.index, df["drainage"], label='drainage')
    # ax.plot(df[["maxTranpiration"]], label='maximum transpiration', sw=1, ls='-')
    # ax.plot(df.index, df["ET0"], label='evaporation')
    # ax.plot(df.index, df["realvaporation"], label='transpiration')
    ax.set_xlabel("")
    ax.set_xticks([0, 16, 30, 47, 61, df.shape[0] - 1])

    # ax.set_xticks([
    #     datetime.date(2022, 6, 15),
    #     datetime.date(2022, 7, 1),
    #     datetime.date(2022, 7, 15),
    #     datetime.date(2022, 8, 1),
    #     datetime.date(2022, 8, 15),
    #     datetime.date(2022, 9, 1)
    # ])
    # ax.set_xlim(datetime.date(2022, 6, 15), datetime.date(2022, 9, 1))
    # ax.set_xticks([0, 15, 29, 46, 60, df.shape[0] - 1])
    # ax.set_title("Precipitation and irrigation", fontdict={"fontsize": 18})
    # ax2 = ax.twinx()
    # ax.set_ylabel("L",  fontsize=15)
    ax.set_ylabel("mm", fontsize=12)
    # ax2.set_ylim(0, 30)
    ax.set_ylim(0, 21)
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="both", labelsize=12)

    # fig = ax.get_figure()
    # fig.autofmt_xdate()

    plt.tight_layout()
    fig.set_size_inches(13, 6)
    fig.savefig(os.path.join(output_folder, "water_balance.pdf"))
    fig.savefig(os.path.join(output_folder, "water_balance.png"))


def correlation_wc(
    obs_folder=os.path.join("data", "errano_evaluation_1gg"),
    forecast_folder=os.path.join("data", "errano_evaluation"),
    output_folder=os.path.join("plots"),
    with_forbidden_sensors=True,
):
    support_dict = {
        "obs": os.path.join(obs_folder, "obs_data", "waterContent.csv"),
    }
    for forecasting_day in ["1gg", "3gg", "7gg"]:
        support_dict[forecasting_day] = os.path.join(
            f"{forecast_folder}_{forecasting_day}", "output", "outputWaterContent.csv"
        )

    forecasting_dict = {}
    for data_type, input_path in support_dict.items():
        forecasting_dict[data_type] = (
            pd.read_csv(input_path)
            .set_index("timestamp")
            .drop(columns=[] if with_forbidden_sensors else forbidden_sensors)
        )
        forecasting_dict[data_type] = forecasting_dict[data_type].reindex(
            sorted(forecasting_dict[data_type].columns), axis=1
        )
        forecasting_dict[data_type] = forecasting_dict[data_type].add_suffix(
            f"_{data_type}"
        )
    df = pd.concat(forecasting_dict.values(), axis=1)
    df = df.interpolate(method="linear", limit_direction="forward", axis=0)
    df = df.dropna(axis="index")
    df = df.reset_index()
    for data_type in support_dict.keys():
        df[f"mean_{data_type}"] = df[
            [c for c in df.columns if c.endswith(f"_{data_type}")]
        ].mean(axis=1)
    df = df.set_index("timestamp")
    df = df[[f"mean_{data_type}" for data_type in support_dict.keys()]]
    df.index = pd.to_datetime(df.index, unit="s")

    scores = {}
    fig, ax = plt.subplots(1, 3, sharey="row")
    for idx, data_type in enumerate(
        [elem for elem in support_dict.keys() if elem != "obs"]
    ):
        ax[idx].scatter(df["mean_obs"], df[f"mean_{data_type}"], color=f"C{idx}")
        slope, intercept = np.polyfit(df["mean_obs"], df[f"mean_{data_type}"], 1)
        # print(slope, intercept)
        # Create a list of values in the best fit line
        abline_values = [slope * i + intercept for i in [0, 1]]
        # Plot the best fit line over the actual values
        ax[idx].plot([0, 1], abline_values, color="black")
        scores[data_type] = [
            r2_score(np.array(df["mean_obs"]), np.array(df[f"mean_{data_type}"]))
        ]
        # ax[idx].plot([0, 1], [0, 1], transform=ax[idx].transAxes, color="black")
        ax[idx].grid()
        ax[idx].set_ylim([0.05, 0.25 if with_forbidden_sensors else 0.2])
        ax[idx].set_xlim([0.05, 0.25 if with_forbidden_sensors else 0.2])
        ax[idx].set_title(data_type.replace("gg", "-day") + ("s" if data_type != "1gg" else "") + " horizon", fontsize=22)
        ax[idx].set_xlabel("Observed WC", fontsize=18)
        ax[idx].tick_params(axis="both", labelsize=18)
        if idx == 0:
            ax[idx].set_ylabel("Forecasted WC", fontsize=18)
    fig.set_size_inches(20, 6)
    plt.tight_layout()
    is_forbidden_sensors_string = (
        "_with_forbidden_sensors" if with_forbidden_sensors else ""
    )
    # df[["mean_obs", "mean_1gg", "mean_3gg", "mean_7gg"]].to_csv(
    #     os.path.join(
    #         output_folder,
    #         f"temp.csv",
    #     ),
    #     index=False,
    # )
    pd.DataFrame(scores).to_csv(
        os.path.join(
            output_folder,
            f"correlation_wc{is_forbidden_sensors_string}.csv",
        ),
        index=False,
    )
    fig.savefig(
        os.path.join(
            output_folder,
            f"correlation_wc{is_forbidden_sensors_string}.pdf",
        )
    )
    fig.savefig(
        os.path.join(
            output_folder,
            f"correlation_wc{is_forbidden_sensors_string}.png",
        )
    )


def forecast_avg_tuning_errors(
    obs_folder=os.path.join("data", "errano_evaluation_1gg"),
    forecast_folder=os.path.join("data", "errano_evaluation"),
    output_folder=os.path.join("plots"),
    with_forbidden_sensors=True,
    budget_type="t",
):
    fig, ax = plt.subplots(1, 3)
    budget_labels = [25, 50, 75]
    budget_thresholds = [25, 50, 75] if budget_type == "t" else [125, 250, 375]
    for idx, budget in enumerate(budget_thresholds):
        support_dict = {
            "obs": os.path.join(obs_folder, "obs_data", "waterPotential.csv"),
        }
        for forecasting_day in ["1gg", "3gg", "7gg"]:
            support_dict[forecasting_day] = os.path.join(
                f"{forecast_folder}_{forecasting_day}_{budget}_{budget_type}",
                "output",
                "output.csv",
            )

        forecasting_dict = {}
        for data_type, input_path in support_dict.items():
            forecasting_dict[data_type] = (
                pd.DataFrame({"timestamp": [1655251200]})
                .append(
                    pd.read_csv(input_path),
                    ignore_index=True,
                )
                .set_index("timestamp")
                .drop(columns=[] if with_forbidden_sensors else forbidden_sensors)
            )
            forecasting_dict[data_type] *= -1
            forecasting_dict[data_type][forecasting_dict[data_type] < 20] = 20
            forecasting_dict[data_type] = forecasting_dict[data_type].apply(
                lambda x: np.log(x)
            )
            forecasting_dict[data_type] = forecasting_dict[data_type].reindex(
                sorted(forecasting_dict[data_type].columns), axis=1
            )
            forecasting_dict[data_type] = forecasting_dict[data_type].add_suffix(
                f"_{data_type}"
            )
        df = pd.concat(forecasting_dict.values(), axis=1)
        df = df.interpolate(method="linear", limit_direction="forward", axis=0)
        df = df.dropna(axis="index")
        df = df.reset_index()
        new_columns = []
        for data_type in support_dict.keys():
            if data_type != "obs":
                new_column = f"RMSE_{data_type}"
                new_columns += [new_column]
                df[new_column] = [
                    mean_squared_error(
                        df[[c for c in df.columns if c.endswith("_obs")]].iloc[
                            i : (i + 1)
                        ],
                        df[[c for c in df.columns if c.endswith(f"_{data_type}")]].iloc[
                            i : (i + 1)
                        ],
                        squared=False,
                    )
                    for i in range(df.shape[0])
                ]
        df = df.append(pd.DataFrame({"timestamp": [1655251200]}), ignore_index=True)
        df = df.set_index("timestamp")
        df = df[new_columns]
        df.index = pd.to_datetime(df.index, unit="s")
        df = df.rename(
            columns={
                column: column.replace("RMSE_", "forecasting horizon = ")
                for column in df.columns
            }
        )

        df.plot(ax=ax[idx])
        ax[idx].set_title(f"budget = {budget_labels[idx]}%")
        ax[idx].set_ylim([0, 1.5])
        ax[idx].set_xlabel("")
        ax[idx].set_ylabel("logRMSE")
    fig.set_size_inches(26, 6)
    plt.tight_layout()
    is_forbidden_sensors_string = (
        "_with_forbidden_sensors" if with_forbidden_sensors else ""
    )
    fig.savefig(
        os.path.join(
            output_folder,
            f"forecasting_avg{is_forbidden_sensors_string}_errors_{budget_type}.pdf",
        )
    )
    fig.savefig(
        os.path.join(
            output_folder,
            f"forecasting_avg{is_forbidden_sensors_string}_errors_{budget_type}.png",
        )
    )


def forecast_std_tuning_errors(
    obs_folder=os.path.join("data", "errano_evaluation_1gg"),
    forecast_folder=os.path.join("data", "errano_evaluation"),
    output_folder=os.path.join("plots"),
    with_forbidden_sensors=True,
    budget_type="t",
):
    fig, ax = plt.subplots(1, 3)
    budget_labels = [25, 50, 75]
    budget_thresholds = [25, 50, 75] if budget_type == "t" else [125, 250, 375]
    for idx, budget in enumerate(budget_thresholds):
        support_dict = {
            "obs": os.path.join(obs_folder, "obs_data", "waterPotential.csv"),
        }
        for forecasting_day in ["1gg", "3gg", "7gg"]:
            support_dict[forecasting_day] = os.path.join(
                f"{forecast_folder}_{forecasting_day}_{budget}_{budget_type}",
                "output",
                "output.csv",
            )

        forecasting_dict = {}
        for data_type, input_path in support_dict.items():
            forecasting_dict[data_type] = (
                pd.DataFrame({"timestamp": [1655251200]})
                .append(
                    pd.read_csv(input_path),
                    ignore_index=True,
                )
                .set_index("timestamp")
                .drop(columns=[] if with_forbidden_sensors else forbidden_sensors)
            )
            forecasting_dict[data_type][forecasting_dict[data_type] > -20] = -20
            forecasting_dict[data_type] = forecasting_dict[data_type].reindex(
                sorted(forecasting_dict[data_type].columns), axis=1
            )
            forecasting_dict[data_type] = forecasting_dict[data_type].add_suffix(
                f"_{data_type}"
            )
        df = pd.concat(forecasting_dict.values(), axis=1)
        df = df.interpolate(method="linear", limit_direction="forward", axis=0)
        df = df.dropna(axis="index")
        df = df.reset_index()
        new_columns = []
        for data_type in support_dict.keys():
            if data_type != "obs":
                new_column = f"RMSE_{data_type}"
                new_columns += [new_column]
                # print(df[[c for c in df.columns if c.endswith("_obs")]].iloc[0])
                # print(df[[c for c in df.columns if c.endswith("_obs")]].iloc[:1])
                # print(df[[c for c in df.columns if c.endswith(f"_{data_type}")]].iloc[0])
                # print(df[[c for c in df.columns if c.endswith(f"_{data_type}")]].iloc[:1])
                df[new_column] = [
                    mean_squared_error(
                        df[[c for c in df.columns if c.endswith("_obs")]].iloc[
                            i : (i + 1)
                        ],
                        df[[c for c in df.columns if c.endswith(f"_{data_type}")]].iloc[
                            i : (i + 1)
                        ],
                        squared=False,
                    )
                    for i in range(df.shape[0])
                ]
        df["average"] = df[[c for c in df.columns if c.endswith("_obs")]].mean(axis=1)
        df = df.append(pd.DataFrame({"timestamp": [1655251200]}), ignore_index=True)
        df = df.set_index("timestamp")
        # df = df[new_columns]
        df.index = pd.to_datetime(df.index, unit="s")

        # df = df.rename(
        #     columns={
        #         column: column.replace("RMSE_", "forecasting horizon = ")
        #         for column in df.columns
        #     }
        # )

        df["average"].plot(ax=ax[idx], color="C5", label="average")
        ax[idx].set_title(f"budget = {budget_labels[idx]}%")
        # ax.set_ylim([-800, 0])
        ax[idx].fill_between(
            df.index,
            df["average"] - df["RMSE_7gg"],
            df["average"] + df["RMSE_7gg"],
            alpha=0.2,
            color="C2",
            label="RMSE on 7gg",
        )
        ax[idx].fill_between(
            df.index,
            df["average"] - df["RMSE_3gg"],
            df["average"] + df["RMSE_3gg"],
            alpha=0.4,
            color="C1",
            label="RMSE on 3gg",
        )
        ax[idx].fill_between(
            df.index,
            df["average"] - df["RMSE_1gg"],
            df["average"] + df["RMSE_1gg"],
            alpha=0.6,
            color="C0",
            label="RMSE on 1gg",
        )
        ax[idx].legend()
        ax[idx].set_xlabel("")
        ax[idx].set_ylabel("cbar")
    fig.set_size_inches(26, 6)
    plt.tight_layout()
    is_forbidden_sensors_string = (
        "_with_forbidden_sensors" if with_forbidden_sensors else ""
    )
    fig.savefig(
        os.path.join(
            output_folder,
            f"forecasting_std{is_forbidden_sensors_string}_errors_{budget_type}.pdf",
        )
    )
    fig.savefig(
        os.path.join(
            output_folder,
            f"forecasting_std{is_forbidden_sensors_string}_errors_{budget_type}.png",
        )
    )


def summary_tuning_budget(
    obs_folder=os.path.join("data", "errano_evaluation_1gg"),
    forecast_folder=os.path.join("data", "errano_evaluation"),
    output_folder=os.path.join("plots"),
    with_forbidden_sensors=True,
):
    budget_labels = [25, 50, 75, 100]
    fig, ax = plt.subplots(1, 2)
    is_forbidden_sensors_string = (
        "_with_forbidden_sensors" if with_forbidden_sensors else ""
    )
    for axidx, budget_type in enumerate(["t", "b"]):
        result = pd.DataFrame()
        budget_thresholds = (
            [25, 50, 75, 100] if budget_type == "t" else [125, 250, 375, 500]
        )
        for idx, budget in enumerate(budget_thresholds):
            support_dict = {
                "obs": os.path.join(obs_folder, "obs_data", "waterPotential.csv"),
            }
            for forecasting_day in ["1gg", "3gg", "7gg"]:
                loading_path = (
                    f"{forecast_folder}_{forecasting_day}"
                    if budget == 100 or budget == 500
                    else f"{forecast_folder}_{forecasting_day}_{budget}_{budget_type}"
                )
                support_dict[forecasting_day] = os.path.join(
                    loading_path,
                    "output",
                    "output.csv",
                )

            forecasting_dict = {}
            for data_type, input_path in support_dict.items():
                forecasting_dict[data_type] = (
                    pd.DataFrame({"timestamp": [1655251200]})
                    .append(
                        pd.read_csv(input_path),
                        ignore_index=True,
                    )
                    .set_index("timestamp")
                    .drop(columns=[] if with_forbidden_sensors else forbidden_sensors)
                )
                forecasting_dict[data_type] *= -1
                forecasting_dict[data_type][forecasting_dict[data_type] < 20] = 20
                forecasting_dict[data_type] = forecasting_dict[data_type].apply(
                    lambda x: np.log(x)
                )
                forecasting_dict[data_type] = forecasting_dict[data_type].reindex(
                    sorted(forecasting_dict[data_type].columns), axis=1
                )
                forecasting_dict[data_type] = forecasting_dict[data_type].add_suffix(
                    f"_{data_type}"
                )
            df = pd.concat(forecasting_dict.values(), axis=1)
            df = df.interpolate(method="linear", limit_direction="forward", axis=0)
            df = df.dropna(axis="index")
            df = df.reset_index()
            for data_type in support_dict.keys():
                if data_type != "obs":
                    raw = [
                        mean_squared_error(
                            df[[c for c in df.columns if c.endswith("_obs")]].iloc[
                                i : (i + 1)
                            ],
                            df[
                                [c for c in df.columns if c.endswith(f"_{data_type}")]
                            ].iloc[i : (i + 1)],
                            squared=False,
                        )
                        for i in range(df.shape[0])
                    ]
                    result = result.append(
                        {
                            "budget": int(budget_labels[idx]),
                            f"{data_type}": np.mean(raw),
                            f"std_{data_type}": np.std(raw),
                        },
                        ignore_index=True,
                    )

        result = result.set_index("budget")
        result = result.groupby(level=0).sum(min_count=1)
        result[[c for c in result.columns if not c.startswith(f"std")]].plot.bar(
            ax=ax[axidx]
        )
        ax[axidx].set_xlabel("Percentage of budget")
        ax[axidx].set_ylabel("LogRMSE")
        budget_string = "n. iterarations" if budget_type == "b" else "time period"
        ax[axidx].set_title(f"budget = {budget_string}")
        ax[axidx].tick_params(axis="x", rotation=45)
        result.round(2).to_csv(
            os.path.join(
                output_folder,
                f"summary_tuning_erros{is_forbidden_sensors_string}_{budget_type}.csv",
            )
        )
    # fig.set_size_inches(13, 6)
    # plt.tight_layout()
    # fig.savefig(
    #     os.path.join(
    #         output_folder, f"summary_tuning_erros{is_forbidden_sensors_string}.pdf"
    #     )
    # )
    # fig.savefig(
    #     os.path.join(
    #         output_folder, f"summary_tuning_erros{is_forbidden_sensors_string}.png"
    #     )
    # )


def main():
    meteo()
    water_balance()
    ground_potential()
    forecast_avg()
    forecast_std()
    correlation_wc()
    summary_tuning_budget()


if __name__ == "__main__":
    main()
