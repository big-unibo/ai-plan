import os

seed = int(os.environ["PYTHONHASHSEED"])
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
os.environ["TF_DETERMINISTIC_OPS"] = "1"

import random as python_random

python_random.seed(seed)

import numpy as np

np.random.seed(seed)

import tensorflow as tf

tf.random.set_seed(seed)
tf.experimental.numpy.random.seed(seed)
tf.keras.utils.set_random_seed(seed)

import re
import json
import time
import warnings

from utils.parameters import Parameters

warnings.filterwarnings("ignore")

import pandas as pd

pd.set_option("display.max_columns", 500)
# pd.set_option("display.max_rows", 5000)

from functools import partial
from flaml import tune

from utils.parameter_parsing import parse_args, load_conf
from utils.data_acquisition import (
    load_agro_data_from_csv,
    train_val_test_split,
    open_db_connection,
    close_db_connection,
    load_agro_data_from_db,
    create_rolling_window,
    create_train_val_test_sets,
    get_data_labels,
    plot_results,
)
from utils.json_to_csv import json_to_csv
from automl.optimization import objective, my_config_constraint
from automl.space_loading import get_space


def main(args, run_cfg, db_cfg):
    # Set meaningful information previously obtained
    real_sensors = pd.read_csv(os.path.join("resources", "real_sensors.csv")).values
    Parameters().set_real_sensors_coords(real_sensors)

    """
    ### LOCAL MODE
    df = load_agro_data_from_csv()
    df = create_rolling_window(df, run_cfg)
    X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(
        df, run_cfg["window_parameters"]["n_hours_ahead"], test_ratio=0.33, val_ratio=0.5, shuffle=False
    )
    """

    # Set baseline timestamp for re-training, depending on specific stride/horizon
    n_hours_ahead = run_cfg["window_parameters"]["n_hours_ahead"]
    if n_hours_ahead == 48:
        start_timestamp = 1656277200 # prediction of 28 June 2022 21:00 (+ 02:00)
    elif n_hours_ahead == 96:
        start_timestamp = 1656104400 # prediction of 28 June 2022 21:00 (+ 02:00)
    elif n_hours_ahead == 168:
        start_timestamp = 1655845200 # prediction of 28 June 2022 21:00 (+ 02:00)
    else:
        raise Exception("Wrong stride/horizon!")

    ### DB MODE
    # Load the datasets from CSV files
    case_study = re.sub(" |\.", "_", run_cfg["tuning_parameters"]["case_study"])
    data_path = os.path.join(
        "outcomes",
        re.sub(" |\.", "_", run_cfg["run_version"]) + "_retrain_4", # + "_retrain_1",
        f"""HA_{run_cfg["window_parameters"]["n_hours_ahead"]}_HP_{run_cfg["window_parameters"]["n_hours_past"]}_SA_{run_cfg["window_parameters"]["stride_ahead"]}_SP_{run_cfg["window_parameters"]["stride_past"]}_{case_study}""",
        "data",
    )
    X_train = pd.read_csv(
        os.path.join(data_path, "X_train.csv"), index_col="unix_timestamp"
    )
    y_train = pd.read_csv(
        os.path.join(data_path, "y_train.csv"), index_col="unix_timestamp"
    )
    X_val = pd.read_csv(
        os.path.join(data_path, "X_val.csv"), index_col="unix_timestamp"
    )
    y_val = pd.read_csv(
        os.path.join(data_path, "y_val.csv"), index_col="unix_timestamp"
    )
    X_test = pd.read_csv(
        os.path.join(data_path, "X_test.csv"), index_col="unix_timestamp"
    )
    y_test = pd.read_csv(
        os.path.join(data_path, "y_test.csv"), index_col="unix_timestamp"
    )

    ### DB MODE
    connection = open_db_connection(db_cfg)

    # Read from DB the hyperparameters of the best configuration previously found with the FLAML tuning
    hyperparameters_query = "SELECT hyperparameter_name, value \
        FROM synthetic_algorithm_hyperparamaters \
        WHERE algorithm_name = '{}'"
    hyperparameters_df = pd.read_sql(
        hyperparameters_query.format(run_cfg["tuning_parameters"]["algorithm_name"]),
        connection,
    )
    with open(os.path.join(args.run_directory_path, "automl_input.json"), "r") as jsonFile:
        data = json.load(jsonFile)

    for _, row in hyperparameters_df.iterrows():
        if row["hyperparameter_name"] != "n_hour_past" and row["hyperparameter_name"] != "n_hour_ahead" and row["hyperparameter_name"] != "stride_past" and row["hyperparameter_name"] != "stride_ahead" and row["hyperparameter_name"] != "recursion"and row["hyperparameter_name"] != "pca" and row["hyperparameter_name"] != "hyperparameter_tuning" and row["hyperparameter_name"] != "normalization":
            if row["hyperparameter_name"] == "fit_intercept" or row["hyperparameter_name"] == "positive" or row["hyperparameter_name"] == "bootstrap" or row["hyperparameter_name"] == "oob_score" or row["hyperparameter_name"] == "warm_start" or row["hyperparameter_name"] == "shrinking" or row["hyperparameter_name"] == "encoder":
                data["regression"]["choice"][0][row["hyperparameter_name"]]["choice"] = [bool(row["value"])]
            elif row["hyperparameter_name"] == "tol" or row["hyperparameter_name"] == "C" or row["hyperparameter_name"] == "epsilon" or row["hyperparameter_name"] == "dropout":
                data["regression"]["choice"][0][row["hyperparameter_name"]]["choice"] = [float(row["value"])]
            elif row["hyperparameter_name"] == "n_estimators" or row["hyperparameter_name"] == "degree" or row["hyperparameter_name"] == "num_epochs" or row["hyperparameter_name"] == "batch_size" or row["hyperparameter_name"] == "num_hidden_layers" or row["hyperparameter_name"] == "num_neurons":
                data["regression"]["choice"][0][row["hyperparameter_name"]]["choice"] = [int(row["value"])]
            else:
                data["regression"]["choice"][0][row["hyperparameter_name"]]["choice"] = [row["value"]]

    with open(os.path.join(args.run_directory_path, "automl_input.json"), "w") as jsonFile:
        json.dump(data, jsonFile)

    # Load the space
    space = get_space(os.path.join(args.run_directory_path, "automl_input.json"))

    # Get the name (ID) and the coordinates of the real sensors
    sensors_query = "SELECT sensor_name, x, y, z \
        FROM synthetic_sensor_arrangement ssa \
        INNER JOIN synthetic_sensor ss \
        ON ssa.sensor_name = ss.name \
        WHERE arrangement_name = '{}' \
        ORDER BY x ASC, y ASC, z DESC"
    sensors_df = pd.read_sql(
        sensors_query.format(run_cfg["arrangement"]),
        connection,
    )

    # Set tuning constraints
    config_constraints = [(my_config_constraint, ">=", True)]

    # Find best hyper-parameters
    start_time = time.time()
    analysis = tune.run(
        evaluation_function=partial(
            objective,
            X_train.copy(),
            y_train.copy(),
            X_val.copy(),
            y_val.copy(),
            X_test.copy(),
            y_test.copy(),
            run_cfg["window_parameters"]["stride_past"],
            run_cfg["tuning_parameters"]["metric"],
            seed,
            args.run_directory_path,
            sensors_df,
            run_cfg["re_training_parameters"]["re_training_offset"],
            start_timestamp,
        ),
        config=space,
        metric="val_score",
        mode="min",
        num_samples=1, #run_cfg["tuning_parameters"]["batch_size"],
        #time_budget_s=3600,
        config_constraints=config_constraints,
        verbose=0,
        # max_failure=run_cfg["tuning_parameters"]["batch_size"],
    )
    end_time = time.time()

    # Specify which information are needed for the output
    filtered_keys = [
        "train_raw_scores",
        "val_raw_scores",
        "test_raw_scores",
        "train_score",
        "val_score",
        "test_score",
        "status",
        "conf",
        "config",
        "optimizer_params",
        "time_total_s",
    ]
    # Prepare the output file
    automl_output = {
        "optimization_time": end_time - start_time,
        # Filter the information for the best config
        "best_config": {
            key: value
            for key, value in analysis.best_trial.last_result.items()
            if key in filtered_keys
        },
        # For each visited config, filter the information
        "results": [
            {
                key: value if value != float("-inf") else str(value)
                for key, value in values.items()
                if key in filtered_keys
            }
            for values in analysis.results.values()
        ],
    }

    print("Optimization process with FLAML finished")

    # Convert the result in CSV
    json_to_csv(automl_output=automl_output.copy(), args=args)

    # Compute min and max values for ground potential data
    sets_dict = {}
    for set in ["train", "test"]: ###
        sets_dict[set] = pd.read_csv(
            os.path.join(
                args.run_directory_path,
                "predictions",
                f"""conf_{automl_output["best_config"]["conf"]}_{set}.csv""",
            ),
            index_col="unix_timestamp",
        )
    concat_df = pd.concat(
        list(sets_dict.values()) + [y_train, y_val, y_test],
        axis=0,
    )
    min_gp_value = concat_df.min().min()
    max_gp_value = concat_df.max().max()

    for set in ["train", "test"]: ###
        best_pred_df = sets_dict[set].copy()

        # Plot best result for each set
        plot_results(
            best_pred_df,
            locals()[f"y_{set}"],
            set,
            run_cfg["window_parameters"]["n_hours_ahead"],
            args.run_directory_path,
            min_gp_value,
            max_gp_value,
        )

    print("Results stored in the DB")

    ### DB MODE
    close_db_connection(connection)


if __name__ == "__main__":
    args = parse_args()
    run_cfg = load_conf(args.config_file_path)
    db_cfg = load_conf(args.db_credentials_file_path)
    main(args, run_cfg, db_cfg)