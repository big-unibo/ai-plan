import numpy as np

"""Singleton class used to store meaningful information.
"""


class Parameters:
    _instance = None

    # Real sensors coordinates
    real_sensors = []
    real_x_coords = []
    real_y_coords = []
    real_z_coords = []

    # (Filtered) synthetic sensors coordinates
    synthetic_sensors = []
    x_coords = []
    y_coords = []
    z_coords = []

    # Columns names
    weather_columns = []
    watering_columns = []
    real_sensor_columns = []
    sensor_columns = []

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Parameters, cls).__new__(cls)
            cls.weather_columns = [
                "air_temperature",
                "air_humidity",
                "wind_speed",
                "solar_radiation",
            ]
            cls.watering_columns = ["precipitation", "irrigation"]
        return cls._instance

    def set_real_sensors_coords(cls, real_sensors_coords):
        cls.real_sensors = real_sensors_coords
        cls.real_x_coords = list(set(np.array(cls.real_sensors)[:, 0]))
        cls.real_x_coords.sort()
        cls.real_y_coords = list(set(np.array(cls.real_sensors)[:, 1]))
        cls.real_y_coords.sort()
        cls.real_z_coords = list(set(np.array(cls.real_sensors)[:, 2]))
        cls.real_z_coords.sort(reverse=True)

        cls.real_sensor_columns = [
            f"z{z}_y{y}_x{x}"
            for z in cls.real_z_coords
            for y in cls.real_y_coords
            for x in cls.real_x_coords
        ]

    def set_synthetic_sensors_coords(cls, synthetic_sensors_coords):
        cls.synthetic_sensors = synthetic_sensors_coords
        cls.synthetic_x_coords = list(set(np.array(cls.synthetic_sensors)[:, 0]))
        cls.synthetic_x_coords.sort()
        cls.synthetic_y_coords = list(set(np.array(cls.synthetic_sensors)[:, 1]))
        cls.synthetic_y_coords.sort()
        cls.synthetic_z_coords = list(set(np.array(cls.synthetic_sensors)[:, 2]))
        cls.synthetic_z_coords.sort(reverse=True)

        cls.synthetic_sensor_columns = [
            f"z{z}_y{y}_x{x}"
            for z in cls.synthetic_z_coords
            for y in cls.synthetic_y_coords
            for x in cls.synthetic_x_coords
        ]

    def get_real_sensors(cls):
        return cls.real_sensors

    def get_real_x_coords(cls):
        return cls.real_x_coords

    def get_real_y_coords(cls):
        return cls.real_y_coords

    def get_real_z_coords(cls):
        return cls.real_z_coords

    def get_synthetic_sensors(cls):
        return cls.synthetic_sensors

    def get_synthetic_x_coords(cls):
        return cls.synthetic_x_coords

    def get_synthetic_y_coords(cls):
        return cls.synthetic_y_coords

    def get_synthetic_z_coords(cls):
        return cls.synthetic_z_coords

    def get_weather_columns(cls):
        return cls.weather_columns

    def get_watering_columns(cls):
        return cls.watering_columns

    def get_real_sensor_columns(cls):
        return cls.real_sensor_columns

    def get_synthetic_sensor_columns(cls):
        return cls.synthetic_sensor_columns
