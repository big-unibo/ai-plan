import os
import traceback

import pandas as pd
import numpy as np

import math
import re

from scipy.spatial import distance

import matplotlib.pyplot as plt

from sqlalchemy import create_engine

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import tensorflow as tf

from utils.parameters import Parameters

pd.set_option("display.max_columns", 500)

def load_agro_data_from_csv(path=os.path.join("/", "home", "data")):
    def _ground_potential_reconstruction(df):
        """Ground potential data likely contains missing values.
        Since this kind of data has to be treated properly, we define a custom imputation.
        Data from sensors are sampled every 15 minutes, but we are interested in hourly sampling.
        We are lucky if we have the data at the :00 of the hour at hand , bc we do not have to do nothing.
        Otherwise, we take the value of the previous quarter (15 minutes before).
        This is done until the minute :00 of the previous hour.

        Args:
            path (_type_): input data path

        Returns:
            _type_: output dataframe
        """
        dfs = {}
        for minute in [0, 15, 30, 45]:
            dfs[minute] = df[
                df.apply(
                    lambda x: pd.to_datetime(x["timestamp"], unit="s").minute == minute,
                    axis=1,
                )
            ]
            if minute != 0:
                dfs[minute]["timestamp"] = (
                    (dfs[minute]["timestamp"] + 3600) / 3600
                ).astype(int) * 3600
            dfs[minute] = dfs[minute].pivot(
                index="timestamp", columns=["z", "y", "x"], values="value"
            )
        for minute in [45, 30, 15]:
            dfs[0] = dfs[0].combine_first(dfs[minute])
        return dfs[0]

    real_sensors = [
        [0, 0, -0.2],
        [0, 0, -0.4],
        [0, 0, -0.6],
        [0.3, 0, -0.2],
        [0.3, 0, -0.4],
        [0.3, 0, -0.6],
        [0.6, 0, -0.2],
        [0.6, 0, -0.4],
        [0.6, 0, -0.6],
        [0.9, 0, -0.2],
        [0.9, 0, -0.4],
        [0.9, 0, -0.6],
    ]

    filtered_synthetic_sensors = [
        [0, 0, -60],
        [0, 0, -40],
        [0, 0, -20],
        [25, 0, -60],
        [25, 0, -40],
        [25, 0, -20],
        [50, 0, -60],
        [50, 0, -40],
        [50, 0, -20],
        [80, 0, -60],
        [80, 0, -40],
        [80, 0, -20],
    ]
    # Save sensors coordinates for later usage
    Parameters().set_real_sensors_coords(real_sensors)
    Parameters().set_synthetic_sensors_coords(filtered_synthetic_sensors)

    data = {}
    for variable in [
        "air_temperature",
        "air_humidity",
        "wind_speed",
        "solar_radiation",
        "precipitation",
        "irrigation",
        "ground_potential",
    ]:
        data[variable] = pd.read_csv(os.path.join(path, f"{variable}.csv"))
        if variable == "ground_potential":
            data[variable] = _ground_potential_reconstruction(data[variable].copy())
            data[variable].columns = [
                f"z{column[0]}_y{column[1]}_x{column[2]}"
                for column in data[variable].columns.values
            ]
        else:
            data[variable] = data[variable].rename(columns={"value": variable})
            data[variable] = data[variable].set_index("timestamp")

    result = pd.concat(list(data.values()), axis=1, join="inner")
    return result.fillna(method="ffill")


def open_db_connection(db_cfg):
    engine = create_engine(
        "postgresql://{user}:{password}@{address}:{db_port}/smart_irrigation".format(
            user=db_cfg["db_user"],
            password=db_cfg["db_password"],
            address=db_cfg["db_address"],
            db_port=db_cfg["db_port"],
        )
    )
    return engine.connect()


def close_db_connection(connection):
    connection.close()


def load_agro_data_from_db(run_cfg, db_cfg):
    # Fields and scenarios to consider for train, validation and test
    fields_scenarios_dict = {
        "train_field": {
            run_cfg["field_names"]["train_field_name"]: {
                run_cfg["scenario_names"]["train_scenario_name"]: []
            }
        },
        "val_field": {
            run_cfg["field_names"]["val_field_name"]: {
                run_cfg["scenario_names"]["val_scenario_name"]: []
            }
        },
        "test_field": {
            run_cfg["field_names"]["test_field_name"]: {
                run_cfg["scenario_names"]["test_scenario_name"]: []
            }
        },
    }

    # Open DB connection
    connection = open_db_connection(db_cfg)

    # Queries
    check_field_arrangement_consistency_query = "SELECT * FROM public.synthetic_field_arrangement \
        WHERE field_name = '{}' \
        AND arrangement_name = '{}'"

    years_scenarios_query = "SELECT scenario_name \
        FROM synthetic_scenario_composed_scenario \
        WHERE composed_scenario_name = '{}'"

    arpae_query = "SELECT unix_timestamp, value_type_name, \
        CASE \
            WHEN value_type_name = 'RADIATIONS' AND value < 0 THEN 0 \
            ELSE value \
        END AS value \
        FROM synthetic_scenario_arpae_data \
        WHERE scenario_arpae_name IN (SELECT scenario_arpae_name \
            FROM synthetic_scenario \
            WHERE name = '{}') \
        ORDER BY unix_timestamp ASC, value_type_name ASC"

    water_query = "SELECT unix_timestamp, value_type_name, value \
        FROM synthetic_scenario_water_data \
        WHERE scenario_water_name IN (SELECT scenario_water_name \
            FROM synthetic_scenario \
            WHERE name = '{}') \
        ORDER BY unix_timestamp ASC, value_type_name ASC"

    gp_query = "SELECT unix_timestamp, ROUND(x::numeric, 3) AS x, ROUND(y::numeric, 3) AS y, z, value \
        FROM synthetic_data \
        WHERE field_name = '{}' \
        AND scenario_name = '{}' \
        AND value_type_name = '{}' \
        ORDER BY unix_timestamp ASC, x ASC, y ASC, z DESC"

    sensors_query = "SELECT name, x, y, z \
        FROM synthetic_sensor_arrangement ssa \
        INNER JOIN synthetic_sensor ss \
        ON ssa.sensor_name = ss.name \
        WHERE arrangement_name = '{}' \
        ORDER BY x ASC, y ASC, z DESC"

    frequent_sensors_query = "SELECT x, y, z, COUNT(*) AS freq \
        FROM synthetic_data \
        WHERE field_name = '{field}' \
        AND scenario_name = '{scenario}' \
        AND value_type_name = '{value_type}' \
        GROUP BY x, y, z \
        HAVING COUNT(*) > ( \
            SELECT COUNT(DISTINCT unix_timestamp) / 4 \
            FROM synthetic_data \
            WHERE field_name = '{field}' \
            AND scenario_name = '{scenario}' \
            AND value_type_name = '{value_type}')"

    # Get the specific years scenarios for each scenario
    for field in fields_scenarios_dict:
        for field_name in fields_scenarios_dict[field]:
            field_arrangement_consistency = pd.read_sql(
                check_field_arrangement_consistency_query.format(
                    field_name, run_cfg["arrangement"]
                ),
                connection,
            )
            if field_arrangement_consistency.empty:
                raise Exception(
                    f"""The specified sensors arrangement [{run_cfg["arrangement"]}] is not compatible with the given field [{field_name}]"""
                )
            for scenario in fields_scenarios_dict[field][field_name]:
                years_scenario = pd.read_sql(
                    years_scenarios_query.format(scenario), connection
                )
                fields_scenarios_dict[field][field_name][scenario] = {
                    y: None for y in years_scenario.values.flatten().tolist()
                }
                # Get the scenarios data (weather data, watering data, ground potential data)
                for year_scenario in fields_scenarios_dict[field][field_name][scenario]:
                    year_scenario_name = re.sub(
                        " |\.",
                        "_",
                        (field.split("_", 1)[0] + "_" + year_scenario).lower(),
                    )  # add suffix to avoid duplicate name variables
                    globals()[year_scenario_name + "_arpae_df"] = pd.read_sql(
                        arpae_query.format(year_scenario), connection
                    )
                    globals()[year_scenario_name + "_water_df"] = pd.read_sql(
                        water_query.format(year_scenario), connection
                    )
                    globals()[year_scenario_name + "_gp_df"] = pd.read_sql(
                        gp_query.format(
                            field_name,
                            year_scenario,
                            run_cfg["tuning_parameters"]["value_type_name"],
                        ),
                        connection,
                    )

                    print(year_scenario)

    # Get the coordinates of the real sensors
    sensors_coordinates_df = pd.read_sql(
        sensors_query.format(run_cfg["arrangement"]), connection
    )
    ## Drop last sensors column ##
    sensors_coordinates_df.drop(sensors_coordinates_df[sensors_coordinates_df['name'] == '14'].index, inplace = True)
    sensors_coordinates_df.drop(sensors_coordinates_df[sensors_coordinates_df['name'] == '17'].index, inplace = True)
    sensors_coordinates_df.drop(sensors_coordinates_df[sensors_coordinates_df['name'] == '20'].index, inplace = True)
    sensors_coordinates_df = sensors_coordinates_df.drop(['name'], axis=1)
    ##
    sensors_coordinates_df.sort_values(
        ["x", "y", "z"], ascending=[True, True, False], inplace=True
    )  # sort sensors in ascending order by x and y values and in descending order by z value
    real_sensors = np.around(sensors_coordinates_df.to_numpy(), 2)

    print("REAL SENSORS:")
    print(real_sensors)

    # Pivot and filter data
    for field in fields_scenarios_dict:
        for field_name in fields_scenarios_dict[field]:
            for scenario in fields_scenarios_dict[field][field_name]:
                for year_scenario in fields_scenarios_dict[field][field_name][scenario]:
                    """
                    # Get the coordinates of the most frequent synthetic sensors
                    frequent_sensors_df = pd.read_sql(
                        frequent_sensors_query.format(
                            field=field_name,
                            scenario=year_scenario,
                            value_type=run_cfg["tuning_parameters"]["value_type_name"],
                        ),
                        connection,
                    )  # take the first batch of the training scenario samples
                    frequent_sensors_df = frequent_sensors_df.loc[:, "x":"z"]
                    frequent_sensors_df.sort_values(
                        ["x", "y", "z"], ascending=[True, True, False], inplace=True
                    )  # sort sensors in ascending order by x and y values and in descending order by z value
                    synthetic_sensors = np.around(frequent_sensors_df.to_numpy(), 2)

                    # Filter synthetic sensors
                    filtered_synthetic_sensors = []
                    for real_sensor in real_sensors:
                        min_distance = math.inf
                        for sensor in synthetic_sensors:
                            euclidean_distance = distance.euclidean(real_sensor, sensor)
                            if euclidean_distance <= min_distance or np.isclose(
                                euclidean_distance, min_distance
                            ):
                                min_distance = euclidean_distance
                                best_approx_sensor = sensor
                        filtered_synthetic_sensors.append(tuple(best_approx_sensor))

                    print("FILTERED SYNTHETIC SENSORS - {}:".format(year_scenario))
                    print(filtered_synthetic_sensors)
                    """
                    # Save sensors coordinates for later usage
                    Parameters().set_real_sensors_coords(real_sensors)
                    pd.DataFrame(real_sensors).to_csv(
                        os.path.join("resources", "real_sensors.csv"), index=False
                    )
                    Parameters().set_synthetic_sensors_coords(
                        real_sensors #filtered_synthetic_sensors
                    )

                    # Pivot weather and irrigation data
                    year_scenario_name = re.sub(
                        " |\.",
                        "_",
                        (field.split("_", 1)[0] + "_" + year_scenario).lower(),
                    )
                    globals()[year_scenario_name + "_arpae_df"] = globals()[
                        year_scenario_name + "_arpae_df"
                    ].pivot(
                        index="unix_timestamp",
                        columns="value_type_name",
                        values="value",
                    )
                    globals()[year_scenario_name + "_water_df"] = globals()[
                        year_scenario_name + "_water_df"
                    ].pivot(
                        index="unix_timestamp",
                        columns="value_type_name",
                        values="value",
                    )

                    # Filter ground potential data
                    globals()[year_scenario_name + "_gp_df"] = globals()[
                        year_scenario_name + "_gp_df"
                    ][
                        [
                            i in real_sensors #filtered_synthetic_sensors
                            for i in zip(
                                globals()[year_scenario_name + "_gp_df"].x,
                                globals()[year_scenario_name + "_gp_df"].y,
                                globals()[year_scenario_name + "_gp_df"].z,
                            )
                        ]
                    ]
                    # Pivot ground potential data
                    globals()[year_scenario_name + "_gp_df"] = globals()[
                        year_scenario_name + "_gp_df"
                    ].pivot(
                        index="unix_timestamp",
                        columns=["z", "y", "x"],
                        values="value",
                    )
                    globals()[year_scenario_name + "_gp_df"].columns = [
                        f"z{column[0]}_y{column[1]}_x{column[2]}"
                        for column in globals()[
                            year_scenario_name + "_gp_df"
                        ].columns.values
                    ]
                    globals()[year_scenario_name + "_gp_df"] = globals()[
                        year_scenario_name + "_gp_df"
                    ].reindex(
                        sorted(
                            globals()[year_scenario_name + "_gp_df"].columns,
                            reverse=True,
                        ),
                        axis=1,
                    )

                    # Find timestamp intersection boundaries
                    top_boundary = max(
                        globals()[year_scenario_name + "_arpae_df"].index.min(),
                        globals()[year_scenario_name + "_water_df"].index.min(),
                        globals()[year_scenario_name + "_gp_df"].index.min(),
                    )
                    bottom_boundary = min(
                        globals()[year_scenario_name + "_arpae_df"].index.max(),
                        globals()[year_scenario_name + "_water_df"].index.max(),
                        globals()[year_scenario_name + "_gp_df"].index.max(),
                    )
                    # Concatenate weather, watering and ground potential data
                    globals()[year_scenario_name + "_df"] = pd.concat(
                        [
                            globals()[year_scenario_name + "_arpae_df"],
                            globals()[year_scenario_name + "_water_df"],
                            globals()[year_scenario_name + "_gp_df"],
                        ],
                        axis=1,
                        join="outer",
                    )
                    # Keep only common timestamps
                    globals()[year_scenario_name + "_df"] = globals()[
                        year_scenario_name + "_df"
                    ].loc[top_boundary:bottom_boundary]

                    print(year_scenario, " - Pivoted")

                    """
                    # Imputation
                    globals()[year_scenario_name + "_df"] = (
                        globals()[year_scenario_name + "_df"]
                        .fillna(method="ffill")
                        .fillna(method="bfill")
                    )
                    """
                    # Rename columns
                    globals()[year_scenario_name + "_df"].index.names = [
                        "unix_timestamp"
                    ]
                    globals()[year_scenario_name + "_df"].rename(
                        columns={
                            "AIR_HUMIDITY": "air_humidity",
                            "AIR_TEMPERATURE": "air_temperature",
                            "RADIATIONS": "solar_radiation",
                            "WIND_SPEED": "wind_speed",
                            "IRRIGATIONS": "irrigation",
                            "PRECIPITATIONS": "precipitation",
                        },
                        inplace=True,
                    )
                    # Reorder columns
                    cols = [
                        "air_temperature",
                        "air_humidity",
                        "wind_speed",
                        "solar_radiation",
                        "precipitation",
                        "irrigation",
                    ] + list(globals()[year_scenario_name + "_df"].columns[6:].values)
                    globals()[year_scenario_name + "_df"] = globals()[
                        year_scenario_name + "_df"
                    ][cols]

                    # Add and impute missing timestamps
                    timestamps_range = list(
                        range(top_boundary, bottom_boundary + 1, 3600)
                    )
                    missing_timestamps = [
                        timestamp
                        for timestamp in timestamps_range
                        if timestamp not in globals()[year_scenario_name + "_df"].index
                    ]
                    for timestamp in missing_timestamps:
                        globals()[year_scenario_name + "_df"].loc[
                            timestamp
                        ] = pd.Series(
                            data=np.nan,
                            index=globals()[year_scenario_name + "_df"].columns,
                        )
                    globals()[year_scenario_name + "_df"].sort_index(inplace=True)
                    globals()[year_scenario_name + "_df"].interpolate(
                        limit_direction="both", inplace=True
                    )

                    # Rename sensors columns to real values
                    globals()[year_scenario_name + "_df"].rename(
                        columns={
                            col: new_col
                            for col, new_col in zip(
                                Parameters().get_synthetic_sensor_columns(),
                                Parameters().get_real_sensor_columns(),
                            )
                        },
                        inplace=True,
                    )

                    fields_scenarios_dict[field][field_name][scenario][
                        year_scenario
                    ] = globals()[year_scenario_name + "_df"]

    # Close DB connection
    close_db_connection(connection)

    return fields_scenarios_dict


def create_rolling_window(df, run_cfg):
    """Create the rolling window.

    Args:
        df (_type_): input dataset
        run_cfg (_type_): configuration information for the current run

    Returns:
        _type_: rolling window dataset
    """
    n_hours_ahead = run_cfg["window_parameters"]["n_hours_ahead"]
    n_hour_past = run_cfg["window_parameters"]["n_hours_past"]
    stride_ahead = run_cfg["window_parameters"]["stride_ahead"]
    stride_past = run_cfg["window_parameters"]["stride_past"]

    sensor_columns = Parameters().get_real_sensor_columns()

    agg_dict = {}
    for column in Parameters().get_weather_columns():
        agg_dict[column] = "mean"
    for column in Parameters().get_watering_columns():
        agg_dict[column] = "sum"

    ahead_indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=stride_ahead)
    ahead_df = df.rolling(ahead_indexer).agg(agg_dict)

    for column in sensor_columns:
        agg_dict[column] = lambda rows: rows.iloc[-1]

    past_df = df.rolling(stride_past).agg(agg_dict)

    rolling_window = []
    for past_hours in range(
        0,
        n_hour_past,
        stride_past,
    ):
        rolling_window.append(
            past_df.shift(past_hours).add_suffix(f"_p{past_hours+stride_past}")
        )
    rolling_window.reverse()
    # rolling_window.append(df)
    for ahead_hours in range(0, n_hours_ahead, stride_ahead):
        rolling_window.append(
            ahead_df.shift(-ahead_hours).add_suffix(f"_a{ahead_hours+stride_ahead}")
        )
    rolling_window.append(
        df.shift(-n_hours_ahead)[sensor_columns].add_suffix(f"_a{n_hours_ahead}")
    )
    rolling_window_df = pd.concat(rolling_window, axis=1, join="inner")
    rolling_window_df = rolling_window_df.dropna()

    return rolling_window_df


def create_train_val_test_sets(rolled_df_dict, fields_scenarios_dict, run_cfg):
    # Create train, validation and test sets
    train = pd.concat(
        [
            rolled_df_dict[
                re.sub(" |\.", "_", ("train_" + train_scenario).lower()) + "_df_rolled"
            ]
            for train_scenario in fields_scenarios_dict["train_field"][
                run_cfg["field_names"]["train_field_name"]
            ][run_cfg["scenario_names"]["train_scenario_name"]]
        ],
        axis=0,
    )
    val = pd.concat(
        [
            rolled_df_dict[
                re.sub(" |\.", "_", ("val_" + val_scenario).lower()) + "_df_rolled"
            ]
            for val_scenario in fields_scenarios_dict["val_field"][
                run_cfg["field_names"]["val_field_name"]
            ][run_cfg["scenario_names"]["val_scenario_name"]]
        ],
        axis=0,
    )
    test = pd.concat(
        [
            rolled_df_dict[
                re.sub(" |\.", "_", ("test_" + test_scenario).lower()) + "_df_rolled"
            ]
            for test_scenario in fields_scenarios_dict["test_field"][
                run_cfg["field_names"]["test_field_name"]
            ][run_cfg["scenario_names"]["test_scenario_name"]]
        ],
        axis=0,
    )

    return train, val, test


# To be used with "local" data for debugging purposes
def train_val_test_split(
    data, n_hours_ahead, test_ratio, val_ratio=None, shuffle=False
):
    """Split input dataframe in train, validation and test set.

    Args:
        data (_type_): input data to split
        output_horizon (_type_): output horizon (in hours) to predict
        test_ratio (_type_): percentage of test samples
        val_ratio (_type_, optional): percentage of validation samples, taken from train set. Defaults to None.
        shuffle (bool, optional): whether to shuffle samples. Defaults to False.

    Returns:
        _type_: X_train is the training set, y_train is the training label set,
                X_val is the (optional) validation set, y_val is the (optional) validation label set,
                X_test is the test set, y_test is the test label set
    """
    sensors_columns = [
        col + f"_a{n_hours_ahead}" for col in Parameters().get_real_sensor_columns()
    ]
    X = data.loc[:, data.columns[~data.columns.isin(sensors_columns)]]
    y = data.loc[:, sensors_columns]

    X_val = []
    y_val = []

    X_train, X_test, y_train, y_test = train_test_split(
        X.to_numpy(),
        y.to_numpy(),
        test_size=test_ratio,
        random_state=42,
        shuffle=shuffle,
    )

    if val_ratio != None:
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=val_ratio, random_state=42, shuffle=shuffle
        )

    X_train = pd.DataFrame(
        X_train, columns=X.columns, index=X.iloc[: len(X_train), :].index
    )
    X_val = pd.DataFrame(
        X_val,
        columns=X.columns,
        index=X.iloc[len(X_train) : len(X_train) + len(X_val), :].index,
    )
    X_test = pd.DataFrame(
        X_test, columns=X.columns, index=X.iloc[-len(X_test) :, :].index
    )

    y_train = pd.DataFrame(
        y_train, columns=y.columns, index=y.iloc[: len(y_train), :].index
    )
    y_val = pd.DataFrame(
        y_val,
        columns=y.columns,
        index=y.iloc[len(y_train) : len(y_train) + len(y_val), :].index,
    )
    y_test = pd.DataFrame(
        y_test, columns=y.columns, index=y.iloc[-len(y_test) :, :].index
    )

    return X_train, y_train, X_val, y_val, X_test, y_test


def get_data_labels(train_data, val_data, test_data, n_hours_ahead):
    """Split the given datasets to obtain the samples and the corresponding labels.

    Args:
        train_data (_type_): the training data
        val_data (_type_): the validation data
        test_data (_type_): the test data
        n_hours_ahead (_type_): the number of hours ahead considered in the prediction

    Returns:
        _type_: X_train is the training set, y_train is the training label set,
                X_val is the validation set, y_val is the validation label set,
                X_test is the test set, y_test is the test label set
    """
    sensors_columns = [
        col + f"_a{n_hours_ahead}" for col in Parameters().get_real_sensor_columns()
    ]

    X_train = train_data.loc[
        :, train_data.columns[~train_data.columns.isin(sensors_columns)]
    ]
    y_train = train_data.loc[:, sensors_columns]

    X_val = val_data.loc[:, val_data.columns[~val_data.columns.isin(sensors_columns)]]
    y_val = val_data.loc[:, sensors_columns]

    X_test = test_data.loc[
        :, test_data.columns[~test_data.columns.isin(sensors_columns)]
    ]
    y_test = test_data.loc[:, sensors_columns]

    return X_train, y_train, X_val, y_val, X_test, y_test


def normalize_data(X_train, y_train, X_val, X_test):
    """Normalization function used for Scikit-learn regressors.
    MinMaxScaler is used.

    Args:
        X_train (_type_): the training set
        y_train (_type_): the training label set
        X_val (_type_): the validation set
        X_test (_type_): the test set

    Returns:
        _type_: the normalized X_train, y_train, X_val and X_test sets,
                the y_train scaler
    """
    X_scaler = StandardScaler().fit(X_train)
    X_train = X_scaler.transform(X_train)
    X_val = X_scaler.transform(X_val)
    X_test = X_scaler.transform(X_test)

    ######

    y_scaler = StandardScaler().fit(y_train)
    y_train = y_scaler.transform(y_train)

    return X_train, y_train, X_val, X_test, y_scaler


def denormalize_data(y_train_pred, y_val_pred, y_test_pred, y_scaler):
    """Inverse normalization function used for Scikit-learn regressors.

    Args:
        y_train_pred (_type_): the predictions for y_train set
        y_val_pred (_type_): the predictions for y_val set
        y_test_pred (_type_): the predictions for y_test set
        y_scaler (_type_): the y_train scaler

    Returns:
        _type_: the de-normalized y_train, y_val and y_test sets
    """
    return (
        y_scaler.inverse_transform(y_train_pred),
        y_scaler.inverse_transform(y_val_pred),
        y_scaler.inverse_transform(y_test_pred),
    )


def normalization(tensor, X_scaler):
    """Normalization function used for Keras NNs.
    StandardScaler is implemented.

    Args:
        tensor (_type_): the input tensor
        X_scaler (_type_): the scaler used to normalize train samples

    Returns:
        _type_: the normalized tensor
    """

    """
    normalized_tensor = tf.convert_to_tensor(
        X_scaler.transform(tensor.numpy()),
        dtype=tf.float32,
    )
    """

    normalized_tensor = tf.divide(
        tf.subtract(
            tensor,
            tf.convert_to_tensor(X_scaler.mean_, dtype=tf.float32),
        ),
        tf.math.sqrt(tf.convert_to_tensor(X_scaler.var_, dtype=tf.float32)),
    )
    return replace_with_zeros(normalized_tensor)


def inverse_normalization(tensor, y_scaler):
    """Inverse normalization function used for Keras NNs.

    Args:
        tensor (_type_): the input tensor
        y_scaler (_type_): the scaler used to normalize train labels

    Returns:
        _type_: the de-normalized tensor
    """

    """
    denormalized_tensor = tf.convert_to_tensor(
        y_scaler.inverse_transform(tensor.numpy()), dtype=tf.float32
    )
    """

    denormalized_tensor = tf.add(
        tf.multiply(
            tf.math.sqrt(tf.convert_to_tensor(y_scaler.var_, dtype=tf.float32)),
            tensor,
        ),
        tf.convert_to_tensor(y_scaler.mean_, dtype=tf.float32),
    )

    return replace_with_zeros(denormalized_tensor)


def replace_with_zeros(normalized_tensor):
    # Replace NaN values with zeros
    return tf.where(
        tf.math.is_nan(normalized_tensor),
        tf.zeros_like(normalized_tensor),
        normalized_tensor,
    )


def plot_results(
    best_predictions,
    ground_truth,
    set_name,
    hours_ahead,
    output_path,
    min_gp_value,
    max_gp_value,
):
    """Plot the best configuration result for each sensor.

    Args:
        best_predictions (_type_): the predictions of the best configuration found
        ground_truth (_type_): the ground truth
        set_name (_type_): the set to consider for the plot
        hours_ahead (_type_): the specified hours ahead considered for each sample
        output_path (_type_): the output path where to save the plots
    """
    best_predictions = best_predictions.add_suffix("_pred")
    result = pd.concat([best_predictions, ground_truth], axis=1, join="inner")
    result.index = pd.to_datetime(result.index, unit="s")

    # Create plot
    fig, axes = plt.subplots(nrows=3, ncols=4)
    fig.set_size_inches(24, 24, forward=True)
    for idz, z in enumerate(Parameters().get_real_z_coords()):
        for _, y in enumerate(Parameters().get_real_y_coords()):
            for idx, x in enumerate(Parameters().get_real_x_coords()):
                result[
                    [
                        f"z{z}_y{y}_x{x}_a{hours_ahead}",
                        f"z{z}_y{y}_x{x}_a{hours_ahead}_pred",
                    ]
                ].plot(ax=axes[idz, idx])
                axes[idz, idx].set_title(f"z{z}_y{y}_x{x} - {hours_ahead} h ahead")
                axes[idz, idx].set_ylim([min_gp_value, max_gp_value])

    plt.savefig(os.path.join(output_path, f"{set_name}_best_results.png"))
