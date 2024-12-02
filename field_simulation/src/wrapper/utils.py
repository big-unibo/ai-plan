import argparse
import os


def parse_args_wrapper():
    parser = argparse.ArgumentParser(description="CRITERIA")

    parser.add_argument(
        "-p",
        "--path",
        type=str,
        required=False,
        default=os.path.join("data", "errano_tuning"),
        help="path to working directory",
    )
    parser.add_argument(
        "-nit",
        "--num_iterations",
        type=int,
        required=False,
        default=-1,
        help="number of tuning iterations",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        required=False,
        default=42,
        help="seed for reproducibility",
    )
    args = parser.parse_args()
    return args

def create_directory(result_path, directory):
    result_path = os.path.join(result_path, directory)

    if not os.path.exists(result_path):
        os.makedirs(result_path)

    return result_path
