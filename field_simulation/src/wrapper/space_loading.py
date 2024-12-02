import json
from flaml import tune


def get_space(path):
    return _get_space(load_file(path))


def load_file(path):
    with open(path) as f:
        data = json.load(f)
    return data


def _get_space(input_space):
    output_space = {}
    if type(input_space) is not dict:
        return input_space
    for key, value in input_space.items():
        if key == "choice":
            return tune.choice([_get_space(elem) for elem in value])
        if key == "randint":
            return tune.randint(lower=value[0], upper=value[1])
        if key == "uniform":
            return tune.uniform(lower=value[0], upper=value[1])
        if key == "loguniform":
            return tune.loguniform(lower=value[0], upper=value[1])
        if type(value) is dict:
            output_space[key] = _get_space(value)
        elif type(value) is list:
            raise Exception("You put an array without the 'choice' key")
        else:
            output_space[key] = value
    return output_space
