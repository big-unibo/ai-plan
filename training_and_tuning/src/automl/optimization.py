import os

seed = int(os.environ["PYTHONHASHSEED"])
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
os.environ["TF_DETERMINISTIC_OPS"] = "1"

import random as python_random

python_random.seed(seed)

import traceback
import numpy as np

np.random.seed(seed)

import pandas as pd
import re

from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_squared_log_error

## Feature Engineering operators
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer
from sklearn.impute import IterativeImputer

## Normalization operators
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler

from utils.parameters import Parameters

## Regression algorithms
from utils.persistent_system import PersistentSystem

from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor

from sklearn.multioutput import MultiOutputRegressor

from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
)  # this estimator is much faster than GradientBoostingRegressor for big datasets (n_samples >= 10 000)
from sklearn.ensemble import VotingRegressor
from sklearn.ensemble import StackingRegressor

from sklearn.svm import SVR

# NNs
import tensorflow as tf

tf.random.set_seed(seed)
tf.experimental.numpy.random.seed(seed)
tf.keras.utils.set_random_seed(seed)

from tensorflow import keras

# Keras optimizers
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import Adagrad
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers import Adadelta
from tensorflow.keras.optimizers import Adam

from utils.data_acquisition import (
    normalize_data,
    denormalize_data,
    normalization,
    inverse_normalization,
)

conf = 0  # configuration counter

### RE-TRAINING VARIABLES ###
global test_window
test_window = 33 # number of days considered for test
###

def get_prototype(config):
    """Create the prototype (i.e., the order of the pre-processing transformations + the ml algorithm)
    Args:
        config (_type_): the current config to visit
    Raises:
        NameError: if the hyper-parameter "prototype" is not in the config
    Returns:
        _type_: a list of string, each string is a step
    """
    ml_pipeline = config["prototype"]
    if ml_pipeline is None:
        raise NameError("No prototype specified")
    else:
        ml_pipeline = ml_pipeline.split("_")
    return ml_pipeline


def instantiate_pipeline(prototype, seed, config, columns, stride_past):
    """Create the pipeline instantiation from the prototype
    Args:
        prototype (_type_): sequence of pre-processing steps
        seed (_type_): seed for reproducibility
        config (_type_): the current config to visit
    Returns:
        _type_: the pipeline instantiation
    """
    pipeline = []
    # Iterate each step in the prototype
    for step in prototype:
        # Considering the chosen operator of the step at hand,
        # remove the "type" hyper-parameter and build a dict
        # with just the hyper-parameters that operator expects
        operator_parameters = {
            param_name: globals()[config[step][param_name]]()
            if param_name == "base_estimator"  #
            else config[step][param_name]  #
            for param_name in config[step]
            if param_name != "meta_estimator"
            and param_name != "super_type"
            and param_name != "type"
        }

        if config[step]["type"] == "MLPRegressor":
            operator_parameters["hidden_layer_sizes"] = eval(
                operator_parameters["hidden_layer_sizes"]
            )
        if config[step]["type"] == "PersistentSystem":
            operator_parameters["columns"] = columns
            operator_parameters["stride_past"] = stride_past

        # Instantiate the operator/algorithm, if random_state is in the hyper-parameters
        # of the operator, add it
        if (
            "random_state"
            in globals()[config[step]["type"]](**operator_parameters).get_params()
        ):
            operator = globals()[config[step]["type"]](
                random_state=seed, **operator_parameters
            )
        else:
            operator = globals()[config[step]["type"]](**operator_parameters)
        # If a super type or a meta estimator is defined, add it
        if "super_type" in config[step]:
            operator = globals()[config[step]["super_type"]](operator)
        if "meta_estimator" in config[step]:
            operator = globals()[config[step]["meta_estimator"]](
                regressor=operator,
                transformer=globals()[config["normalization"]["type"]](),
            )

        # Add the operator to the array
        pipeline.append([step, operator])

    # Instantiate the pipeline from the array
    return Pipeline(pipeline)


def get_neuron_count_per_hidden_layer(config):
    # If encoding
    if config["regression"]["encoder"]:
        # we iteratively divide the num_neurons until the half of the network, specifically:
        neuron_count_per_hidden_layer = [
            # the list is built of integers
            int(
                # the actual num_neurons is equal to 2 power to num_neurons
                (2 ** config["regression"]["num_neurons"])
                / (
                    # then we divide it by 2 power to
                    2
                    ** (
                        # the current position (index) in the list (network) IF IT IS IN THE FIRST HALF
                        index
                        if index < (config["regression"]["num_hidden_layers"] / 2)
                        # the inverted position (index) in the list (network) IF IT IS IN THE SECOND HALF
                        else config["regression"]["num_hidden_layers"] - 1 - index
                    )
                )
            )
            for index in range(config["regression"]["num_hidden_layers"])
        ]
    # If not encoding
    else:
        # we repete num_neurons for each num_hidden_layers
        neuron_count_per_hidden_layer = [
            2 ** config["regression"]["num_neurons"]
        ] * config["regression"]["num_hidden_layers"]

    print(neuron_count_per_hidden_layer)
    return neuron_count_per_hidden_layer


def my_config_constraint(config):
    # IF not encoding THEN max 3 hidden layers
    # IF encoding THEN num_hidden_layers is an odd number AND at least 128 neurons in the larger layer
    return (
        config["regression"]["type"] == "FeedForward"
        and (
            (
                not (config["regression"]["encoder"])
                and config["regression"]["num_hidden_layers"] < 4
            )
            or (
                config["regression"]["encoder"]
                and config["regression"]["num_hidden_layers"] % 2 == 1
                and config["regression"]["num_neurons"] > 6
            )
        )
    ) or (not (config["regression"]["type"] == "FeedForward"))


def scikitlearn_objective(
    X_train, y_train, X_val, y_val, X_test, y_test, stride_past, seed, config
):
    """Objective function to optimize when Scikit-learn regressors are used.
    Args:
        X_train (_type_): the training set
        y_train (_type_): the training label set
        X_val (_type_): the validation set
        y_val (_type_): the validation label set
        X_test (_type_): the test set
        y_test (_type_): the test label set
        stride_past (_type_): the specified stride for past observations
        seed (_type_): the seed for reproducibility
        config (_type_): the configuration to explore
    Returns:
        _type_: the predicted y_train, y_val and y_test
    """
    # Get the prototype from the config
    # (i.e., the order of the pre-processing transformations + the ML algorithm)
    prototype = get_prototype(config)

    # Instantiate the pipeline according to the current config
    # (i.e., at each step we put an operator with specific hyper-parameters)
    pipeline = instantiate_pipeline(
        prototype, seed, config, X_train.columns, stride_past
    )

    """
    # Normalization
    (
        X_train,
        y_train,
        X_val,
        X_test,
        y_scaler,
    ) = normalize_data(X_train, y_train, X_val, y_val, X_test, y_test)
    """

    # Fit and prediction
    estimator = pipeline.fit(X_train.to_numpy(), y_train.to_numpy())
    y_train_pred = estimator.predict(X_train.to_numpy())
    y_val_pred = estimator.predict(X_val.to_numpy())
    y_test_pred = estimator.predict(X_test.tail(test_window * 24).to_numpy()) #

    """
    # Inverse normalization
    y_train_pred, y_val_pred, y_test_pred = denormalize_data(
        y_train_pred,
        y_val_pred,
        y_test_pred,
        y_scaler,
    )
    """

    return y_train_pred, y_val_pred, y_test_pred


def build_dnn(
    input_count,
    output_count,
    neuron_count_per_hidden_layer,
    activation,
    last_activation,
    dropout,
    X_scaler,
    y_scaler,
):
    """Create the neural network architecture.
    Args:
        input_count (_type_): number of input neurons
        output_count (_type_): number of output neurons
        neuron_count_per_hidden_layer (_type_): number of neurons for each hidden layer
        activation (_type_): activation function for the hidden layers
        last_activation (_type_): activation funcion for the output layer
        dropout (_type_): dropout rate
        X_scaler (_type_): the scaler used to normalize training samples
        y_scaler (_type_): the scaler used to normalize training labels
    Returns:
        _type_: neural network model
    """
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(input_count)))

    # Normalization
    model.add(keras.layers.Lambda(normalization, arguments={"X_scaler": X_scaler}))
    """
    model.add(
        tf.keras.layers.Normalization(
            axis=1, mean=X_scaler.mean_, variance=X_scaler.var_, invert=False
        )
    )
    """

    for n in neuron_count_per_hidden_layer:
        model.add(keras.layers.Dense(n, activation=activation))
        # model.add(keras.layers.Dropout(rate=dropout))

    model.add(keras.layers.Dropout(rate=dropout))
    model.add(keras.layers.Dense(output_count, activation=last_activation))

    # Inverse normalization
    model.add(
        keras.layers.Lambda(inverse_normalization, arguments={"y_scaler": y_scaler})
    )
    """
    model.add(
        tf.keras.layers.Normalization(
            axis=1, mean=y_scaler.mean_, variance=y_scaler.var_, invert=True
        )
    )
    """

    return model


def root_mean_squared_error(y_true, y_pred):
    return tf.keras.backend.sqrt(
        tf.keras.backend.mean(tf.keras.backend.square(y_pred - y_true))
    )


def log_root_mean_squared_error(y_true, y_pred):
    return tf.keras.backend.sqrt(
        tf.keras.backend.mean(
            tf.keras.backend.square(
                tf.keras.backend.log(tf.keras.backend.abs(y_pred))
                - tf.keras.backend.log(tf.keras.backend.abs(y_true))
            )
        )
    )


def keras_objective(X_train, y_train, X_val, y_val, X_test, y_test, metric, seed, config, re_training_offset):
    """Objective function to optimize when Keras NNs are used.
    Args:
        X_train (_type_): the training set
        y_train (_type_): the training label set
        X_val (_type_): the validation set
        y_val (_type_): the validation label set
        X_test (_type_): the test set
        seed (_type_): the seed for reproducibility
        config (_type_): the configuration to explore
    Returns:
        _type_: the predicted y_train, y_val and y_test
                and the parameters of the chosen optimizer
    """
    os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
    os.environ["TF_DETERMINISTIC_OPS"] = "1"

    python_random.seed(seed)
    np.random.seed(seed)

    tf.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()

    tf.random.set_seed(seed)
    tf.experimental.numpy.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)

    # Fit X and y scalers
    X_scaler = globals()[config["normalization"]["type"]]().fit(X_train.to_numpy())
    y_scaler = globals()[config["normalization"]["type"]]().fit(y_train.to_numpy())

    # Create the model
    dnn = build_dnn(
        X_train.to_numpy().shape[1],
        y_train.to_numpy().shape[1],
        get_neuron_count_per_hidden_layer(
            config
        ),  # eval(config["regression"]["neuron_count_per_hidden_layer"]),
        config["regression"]["activation_function"],
        config["regression"]["last_activation_function"],
        config["regression"]["dropout"],
        X_scaler,
        y_scaler,
    )

    # Instantiate optimizer and callbacks
    optimizer = None
    callbacks = []
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=(2 ** config["regression"]["num_epochs"]) // 5,
        restore_best_weights=True,
    )
    callbacks.append(early_stop)
    if config["regression"]["optimizer"] == "SGD":  # non-adaptive optimizer
        optimizer = globals()[config["regression"]["optimizer"]](
            0.01 #learning_rate=config["regression"]["learning_rate"]
        )
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.8,
            patience=(2 ** config["regression"]["num_epochs"]) // 10,
            mode="min",
            min_lr=0.000001,
        )
        callbacks.append(reduce_lr)
    else:  # adaptive optimizers
        optimizer = globals()[config["regression"]["optimizer"]]()

    # Use the specified metric
    metric_function = (
        root_mean_squared_error if metric == "RMSE" else log_root_mean_squared_error
    )

    # Compile the model
    dnn.compile(
        loss=metric_function,
        optimizer=optimizer,
        metrics=[metric_function],
    )

    # Fit the model
    dnn.fit(
        X_train.to_numpy(),
        y_train.to_numpy(),
        validation_data=(X_val.to_numpy(), y_val.to_numpy()),
        epochs=2 ** config["regression"]["num_epochs"],
        batch_size=2 ** config["regression"]["batch_size"],
        shuffle=False,
        callbacks=callbacks,
    )

    # Prediciton
    y_train_pred = dnn.predict(X_train.to_numpy())
    y_val_pred = dnn.predict(X_val.to_numpy())
    y_test_pred = dnn.predict(X_test.tail(test_window * 24).to_numpy())

    # Get the hyperparameters of the optimizer and convert float32 values to float values for compatibility reasons
    optimizer_params = dnn.optimizer.get_config()
    optimizer_params.update(
        {
            key: float(value)
            for key, value in dnn.optimizer.get_config().items()
            if type(value) == np.float32
        }
    )

    return y_train_pred, y_val_pred, y_test_pred, optimizer_params


def objective(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    stride_past,
    metric,
    seed,
    run_path,
    sensors_df,
    re_training_offset,
    config,
):
    """Function to optimize (i.e., the order of the pre-processing transformations + the ML algorithm)
    Args:
        X_train (_type_): training data
        y_train (_type_): training data labels (ground truth)
        X_val (_type_): validation data
        y_val (_type_): validation data labels (ground truth)
        X_test (_type_): test data
        y_test (_type_): test data labels (ground truth)
        stride_past (_type_): the specified stride for past observations
        seed (_type_): seed for reproducibility
        run_path (_type_): the current run path
        sensors_df (_type_): real sensor names and coordinates
        config (_type_): the current config to visit
    Raises:
        Exception: It tells if something went wrong during the optimization
    Returns:
        _type_: dictionary with the result
    """
    global conf

    # Set the result if the config failed to be evaluated
    sensors_name_list = sensors_df["sensor_name"].values.flatten()
    result = {
        "train_raw_scores": {
            f"train_raw_score_{name}": float("-inf") for name in sensors_name_list
        },
        "val_raw_scores": {
            f"val_raw_score_{name}": float("-inf") for name in sensors_name_list
        },
        "test_raw_scores": {
            f"test_raw_score_{name}": float("-inf") for name in sensors_name_list
        },
        "train_score": float("-inf"),
        "val_score": float("-inf"),
        "test_score": float("-inf"),
        "status": "fail",
    }

    try:
        if config["regression"]["type"] == "FeedForward":
            y_train_pred, y_val_pred, y_test_pred, optimizer_params = keras_objective(
                X_train,
                y_train,
                X_val,
                y_val,
                X_test,
                y_test,
                metric,
                seed,
                config,
                re_training_offset,
            )
        else:
            y_train_pred, y_val_pred, y_test_pred = scikitlearn_objective(
                X_train,
                y_train,
                X_val,
                y_val,
                X_test,
                y_test,
                stride_past,
                seed,
                config,
            )

        # Export predictions to CSV files
        y_train_pred_df = pd.DataFrame(
            y_train_pred, columns=y_train.columns, index=y_train.index
        )
        y_train_pred_df.to_csv(
            os.path.join(run_path, "predictions", "conf_{}_train.csv".format(conf))
        )
        y_val_pred_df = pd.DataFrame(
            y_val_pred, columns=y_val.columns, index=y_val.index
        )
        y_val_pred_df.to_csv(
            os.path.join(run_path, "predictions", "conf_{}_val.csv".format(conf))
        )
        y_test_pred_df = pd.DataFrame(
            y_test_pred, columns=y_test.tail(test_window * 24).columns, index=y_test.tail(test_window * 24).index
        )
        y_test_pred_df.to_csv(
            os.path.join(run_path, "predictions", "conf_{}_test.csv".format(conf))
        )

        # Use the specified metric
        if metric == "LogRMSE":
            metric_function = mean_squared_log_error
            # Apply absolute value to predicted and target values to deal with LogRMSE
            y_train = y_train.abs()
            y_train_pred_df = y_train_pred_df.abs()
            y_val = y_val.abs()
            y_val_pred_df = y_val_pred_df.abs()
            y_test = y_test.abs()
            y_test_pred_df = y_test_pred_df.abs()
        else:
            metric_function = mean_squared_error

        # Compute raw RMSE (one value for each sensor)
        train_raw_rmse = metric_function(
            y_train, y_train_pred_df, multioutput="raw_values", squared=False
        )
        for idx, col in enumerate(y_train.columns):
            z = re.sub("z", "", col.split("_")[0])
            y = re.sub("y", "", col.split("_")[1])
            x = re.sub("x", "", col.split("_")[2])

            sensor_name = int(
                sensors_df.loc[
                    (sensors_df["z"] == float(z))
                    & (sensors_df["y"] == float(y))
                    & (sensors_df["x"] == float(x)),
                    "sensor_name",
                ]
            )

            result["train_raw_scores"][
                f"train_raw_score_{sensor_name}"
            ] = train_raw_rmse[idx]

        val_raw_rmse = metric_function(
            y_val, y_val_pred_df, multioutput="raw_values", squared=False
        )
        for idx, col in enumerate(y_val.columns):
            z = re.sub("z", "", col.split("_")[0])
            y = re.sub("y", "", col.split("_")[1])
            x = re.sub("x", "", col.split("_")[2])

            sensor_name = int(
                sensors_df.loc[
                    (sensors_df["z"] == float(z))
                    & (sensors_df["y"] == float(y))
                    & (sensors_df["x"] == float(x)),
                    "sensor_name",
                ]
            )

            result["val_raw_scores"][f"val_raw_score_{sensor_name}"] = val_raw_rmse[idx]

        test_raw_rmse = metric_function(
            y_test.tail(test_window * 24), y_test_pred_df, multioutput="raw_values", squared=False
        )
        for idx, col in enumerate(y_test.columns):
            z = re.sub("z", "", col.split("_")[0])
            y = re.sub("y", "", col.split("_")[1])
            x = re.sub("x", "", col.split("_")[2])

            sensor_name = int(
                sensors_df.loc[
                    (sensors_df["z"] == float(z))
                    & (sensors_df["y"] == float(y))
                    & (sensors_df["x"] == float(x)),
                    "sensor_name",
                ]
            )

            result["test_raw_scores"][f"test_raw_score_{sensor_name}"] = test_raw_rmse[
                idx
            ]

        # Compute average RMSE
        result["train_score"] = np.around(
            metric_function(y_train, y_train_pred_df, squared=False), 2
        )
        result["val_score"] = np.around(
            metric_function(y_val, y_val_pred_df, squared=False), 2
        )
        result["test_score"] = np.around(
            metric_function(y_test.tail(test_window * 24), y_test_pred_df, squared=False), 2
        )

        # If something is NaN, raise an exception
        for metric in result:
            if metric != "status" and metric != "conf":
                if metric in ["train_raw_scores", "val_raw_scores", "test_raw_scores"]:
                    for elem in result[metric]:
                        if np.isnan(result[metric][elem]):
                            result[metric][elem] = float("-inf")
                            raise Exception(f"The result for {config} was")
                else:
                    if np.isnan(result[metric]):
                        result[metric] = float("-inf")
                        raise Exception(f"The result for {config} was")
        result["status"] = "success"
        result["conf"] = conf
        if config["regression"]["type"] == "FeedForward":
            result["optimizer_params"] = optimizer_params

        conf += 1

    except Exception as e:
        print(
            f"""MyException: {e}
            {traceback.print_exc()}"""
        )

    return result
