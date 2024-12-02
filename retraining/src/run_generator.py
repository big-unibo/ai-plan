import os
import re
import copy
import shutil
from collections import OrderedDict

from utils.parameter_parsing import load_conf
from utils.data_acquisition import (
    load_agro_data_from_db,
    create_rolling_window,
    create_train_val_test_sets,
    get_data_labels,
)

run_version = "v.5.0.0.1.1.6.0.21"

algorithms = [
    "FeedForward",
]

re_training_offset = 0 # 0, 25, 50, 75

case_studies = [
    # Synthetic vs Synthetic
    {
        "field_names": {
            "train_field_name": "Synthetic field ERRANO v.2.0",
            "val_field_name": "Synthetic field ERRANO v.2.0",
            "test_field_name": "Synthetic field SANDY v.21.0",
        },
        "scenario_names": {
            "train_scenario_name": "Synthetic Martorano v.5.0",
            "val_scenario_name": "Synthetic Bologna v.5.0.1",
            "test_scenario_name": "Synthetic Bologna v.6.21.2",
        },
    },
]

rolling_window_parameters_values = [48, 96, 168]
rolling_window_parameters = [
    {
        "n_hours_ahead": value,
        "n_hours_past": value,
        "stride_ahead": value,
        "stride_past": value,
    }
    for value in rolling_window_parameters_values
]

dict_template = OrderedDict(
    [
        (
            "field_names",
            {
                "train_field_name": "",
                "val_field_name": "",
                "test_field_name": "",
            },
        ),
        (
            "scenario_names",
            {
                "train_scenario_name": "",
                "val_scenario_name": "",
                "test_scenario_name": "",
            },
        ),
        ("arrangement", "Fondo ERRANO"),
        ("run_version", run_version),
        (
            "window_parameters",
            {
                "n_hours_ahead": None,
                "n_hours_past": None,
                "stride_ahead": None,
                "stride_past": None,
            },
        ),
        (
            "tuning_parameters",
            {
                "value_type_name": "GROUND_WATER_POTENTIAL",  # GROUND_SATURATION_DEGREE
                "algorithm_name": "",
                "kind": "",
                "metric": "RMSE",  # LogRMSE
                "case_study": "",
                "description": "",
                "batch_size": -1,
            },
        ),
        (
            "re_training_parameters",
            {
                "re_training_offset": re_training_offset,
            },
        ),
    ]
)


def create_directory(directory_path):
    """Create a directory.

    Args:
        directory_path (str, bytes or os.PathLike object): path to the directory to create.
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)


def __write_run(path, run):
    try:
        print("   -> {}".format(path))
        with open(path, "w") as f:
            for k, v in run.items():
                if isinstance(v, str):
                    f.write("{}: {}\n".format(k, v))
                else:
                    f.write("{}:\n".format(k))
                    for i, j in v.items():
                        f.write("  {}: {}\n".format(i, j))
    except Exception as e:
        print(e)


def generate_runs(db_credentials_file_path):
    run_paths = []
    src_input_path = os.path.join("resources", "automl_inputs")

    for case_study in case_studies:
        print("# CASE STUDY: {}".format(case_study))
        fields_scenarios_dict = {}
        if case_study["field_names"][
            [*case_study["field_names"].keys()][-1]
        ].startswith("Synthetic"):
            case = "Synthetic vs Synthetic"
        elif case_study["field_names"][
            [*case_study["field_names"].keys()][0]
        ].startswith("Synthetic"):
            case = "Synthetic vs Real"
        else:
            case = "Real vs Real"
        for algorithm in algorithms:
            print("## ALGORITHM: {}".format(algorithm))
            for parameter in rolling_window_parameters:
                run = copy.deepcopy(dict_template)
                run["field_names"][
                    "train_field_name"
                ] = f"""{case_study[
                    'field_names'
                ]['train_field_name']}"""
                run["field_names"][
                    "val_field_name"
                ] = f"""{case_study[
                    "field_names"
                ]["val_field_name"]}"""
                run["field_names"][
                    "test_field_name"
                ] = f"""{case_study[
                    "field_names"
                ]["test_field_name"]}"""
                run["scenario_names"][
                    "train_scenario_name"
                ] = f"""{case_study[
                    "scenario_names"
                ]["train_scenario_name"]}"""
                run["scenario_names"][
                    "val_scenario_name"
                ] = f"""{case_study[
                    "scenario_names"
                ]["val_scenario_name"]}"""
                run["scenario_names"][
                    "test_scenario_name"
                ] = f"""{case_study[
                    "scenario_names"
                ]["test_scenario_name"]}"""
                run["window_parameters"]["n_hours_ahead"] = parameter["n_hours_ahead"]
                run["window_parameters"]["n_hours_past"] = parameter["n_hours_past"]
                run["window_parameters"]["stride_ahead"] = parameter["stride_ahead"]
                run["window_parameters"]["stride_past"] = parameter["stride_past"]
                run["tuning_parameters"][
                    "algorithm_name"
                ] = f"""{algorithm} {run_version} HA {parameter["n_hours_ahead"]} HP {parameter["n_hours_past"]} SA {parameter["stride_ahead"]} SP {parameter["stride_past"]} {case}"""
                run["tuning_parameters"]["kind"] = algorithm
                run["tuning_parameters"]["case_study"] = case
                sensors = "all sensors"
                if "single sensor" in run["arrangement"]:
                    sensors = f"""one sensor [{run["arrangement"][-1]}]"""
                normalization = ""
                if algorithm == "SVR" or algorithm == "FeedForward":
                    normalization = " - Normalization: standard"
                run["tuning_parameters"][
                    "description"
                ] = f'''"First attempt of {run["tuning_parameters"]["kind"]} with {sensors} - Train field: {run["field_names"]["train_field_name"]}, Val field: {run["field_names"]["val_field_name"]}, Test field: {run["field_names"]["test_field_name"]} - Train scenario: {run["scenario_names"]["train_scenario_name"]}, Val scenario: {run["scenario_names"]["val_scenario_name"]}, Test scenario: {run["scenario_names"]["test_scenario_name"]} - Imputation: linear interpolation{normalization} - Metric: {run["tuning_parameters"]["metric"]} - Output type: {run["tuning_parameters"]["value_type_name"]}"'''

                algo_name = re.sub(
                    " |\.", "_", run["tuning_parameters"]["algorithm_name"]
                )
                case_name = re.sub(" |\.", "_", case)
                common_path = os.path.join(
                    "outcomes",
                    re.sub(" |\.", "_", run_version) + "_retrain_0_v4", # + "_retrain_1"
                    f"""HA_{run["window_parameters"]["n_hours_ahead"]}_HP_{run["window_parameters"]["n_hours_past"]}_SA_{run["window_parameters"]["stride_ahead"]}_SP_{run["window_parameters"]["stride_past"]}_{case_name}""",
                )
                run_path = os.path.join(
                    common_path,
                    "runs",
                    f"run_{algo_name}",
                )
                create_directory(os.path.join(run_path, "logs"))
                create_directory(os.path.join(run_path, "predictions"))
                __write_run(
                    os.path.join(
                        run_path,
                        "config_{}.yaml".format(algo_name),
                    ),
                    run,
                )
                data_path = os.path.join(common_path, "data")
                create_directory(data_path)
                if not fields_scenarios_dict:
                    print("Load data from DB")  #
                    fields_scenarios_dict = load_agro_data_from_db(
                        run, load_conf(db_credentials_file_path)
                    )
                    print(fields_scenarios_dict)
                if not os.listdir(data_path):
                    print("Create rolling window")  #
                    rolled_df_dict = {}
                    for field in fields_scenarios_dict:
                        for field_name in fields_scenarios_dict[field]:
                            for scenario in fields_scenarios_dict[field][field_name]:
                                for year_scenario in fields_scenarios_dict[field][
                                    field_name
                                ][scenario]:
                                    # Create rolling windows
                                    rolled_df_dict[
                                        re.sub(
                                            " |\.",
                                            "_",
                                            (
                                                field.split("_", 1)[0]
                                                + "_"
                                                + year_scenario
                                            ).lower(),
                                        )
                                        + "_df_rolled"
                                    ] = create_rolling_window(
                                        fields_scenarios_dict[field][field_name][
                                            scenario
                                        ][year_scenario],
                                        run,
                                    )
                    # Create train, validation and test sets
                    train_data, val_data, test_data = create_train_val_test_sets(
                        rolled_df_dict, fields_scenarios_dict, run
                    )
                    # Get data labels
                    X_train, y_train, X_val, y_val, X_test, y_test = get_data_labels(
                        train_data,
                        val_data,
                        test_data,
                        run["window_parameters"]["n_hours_ahead"],
                    )
                    # Generate samples and ground truth CSV files
                    X_train.to_csv(os.path.join(data_path, "X_train.csv"))
                    y_train.to_csv(os.path.join(data_path, "y_train.csv"))
                    X_val.to_csv(os.path.join(data_path, "X_val.csv"))
                    y_val.to_csv(os.path.join(data_path, "y_val.csv"))
                    X_test.to_csv(os.path.join(data_path, "X_test.csv"))
                    y_test.to_csv(os.path.join(data_path, "y_test.csv"))

                # Generate AutoML input files
                if run["tuning_parameters"]["kind"] == "PersistentSystem":
                    shutil.copy2(
                        os.path.join(src_input_path, "persistent_system_input.json"),
                        os.path.join(run_path, "automl_input.json"),
                    )
                elif run["tuning_parameters"]["kind"] == "LinearRegression":
                    shutil.copy2(
                        os.path.join(src_input_path, "linear_regressor_input.json"),
                        os.path.join(run_path, "automl_input.json"),
                    )
                elif run["tuning_parameters"]["kind"] == "RandomForest":
                    shutil.copy2(
                        os.path.join(src_input_path, "random_forest_input.json"),
                        os.path.join(run_path, "automl_input.json"),
                    )
                elif run["tuning_parameters"]["kind"] == "SVR":
                    shutil.copy2(
                        os.path.join(src_input_path, "svr_input.json"),
                        os.path.join(run_path, "automl_input.json"),
                    )
                else:
                    shutil.copy2(
                        os.path.join(src_input_path, "feed_forward_input.json"),
                        os.path.join(run_path, "automl_input.json"),
                    )

                run_paths.append(run_path)

    return run_paths


