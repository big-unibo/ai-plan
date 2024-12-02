import argparse
import yaml


def parse_args():

    parser = argparse.ArgumentParser(description="HAMLET")

    parser.add_argument(
        "-run_directory_path",
        "--run_directory_path",
        nargs="?",
        type=str,
        required=True,
        help="path to the current run directory",
    )
    parser.add_argument(
        "-config_file_path",
        "--config_file_path",
        nargs="?",
        type=str,
        required=True,
        help="path to the configuration file",
    )
    parser.add_argument(
        "-db_credentials_file_path",
        "--db_credentials_file_path",
        nargs="?",
        type=str,
        required=True,
        help="path to the DB credentials file",
    )

    args = parser.parse_args()

    return args


def load_conf(path):
    with open(path, "r") as ymlfile:
        cfg = yaml.safe_load(ymlfile)

    return cfg
