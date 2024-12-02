import pandas as pd

from utils.parameters import Parameters

"""Class implementing the "Persistent System" behaviour.
"""


class PersistentSystem:
    def __init__(self, columns, stride_past):
        self.columns = columns
        self.stride_past = stride_past

    def fit(self, X_train, y_train):
        return self

    def predict(self, dataset):
        df = pd.DataFrame(data=dataset, columns=self.columns)

        predictions = df[
            [
                sensor + f"_p{self.stride_past}"
                for sensor in Parameters().get_real_sensor_columns()
            ]
        ]

        return predictions.to_numpy()

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def get_params(self, deep=True):
        return {"columns": self.columns, "stride_past": self.stride_past}
