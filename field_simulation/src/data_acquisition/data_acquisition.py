import os
import numpy as np
import pandas as pd
import yaml
import re
import math
from scipy.spatial import distance

from sqlalchemy import create_engine

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

def load_conf(path):
    with open(path, "r") as ymlfile:
        cfg = yaml.safe_load(ymlfile)

    return cfg

def importData(db_cfg, configDict, isTuning, path):
    check_field_arrangement_consistency_query = "SELECT * FROM public.synthetic_field_arrangement \
        WHERE field_name = '{}' \
        AND arrangement_name = '{}'"

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
        AND value_type_name = 'GROUND_WATER_POTENTIAL' \
        ORDER BY unix_timestamp ASC, x ASC, y ASC, z DESC"

    sensors_query = "SELECT x, y, z \
        FROM synthetic_sensor_arrangement ssa \
        INNER JOIN synthetic_sensor ss \
        ON ssa.sensor_name = ss.name \
        WHERE arrangement_name = '{}' \
        ORDER BY x ASC, y ASC, z DESC"

    frequent_sensors_query = "SELECT x, y, z, COUNT(*) AS freq \
        FROM synthetic_data \
        WHERE field_name = '{field}' \
        AND scenario_name = '{scenario}' \
        AND value_type_name = 'GROUND_WATER_POTENTIAL' \
        GROUP BY x, y, z \
        HAVING COUNT(*) > ( \
            SELECT COUNT(DISTINCT unix_timestamp) / 4 \
            FROM synthetic_data \
            WHERE field_name = '{field}' \
            AND scenario_name = '{scenario}' \
            AND value_type_name = 'GROUND_WATER_POTENTIAL')"

    connection = open_db_connection(db_cfg)

    field_arrangement_consistency = pd.read_sql(
                check_field_arrangement_consistency_query.format(
                    configDict["data"]["field_name"], configDict["data"]["arrangement"]
                ),
                connection,
            )
    if field_arrangement_consistency.empty:
        raise Exception(
            f"""The specified sensors arrangement [{configDict["data"]["arrangement"]}] is not compatible with the given field [{configDict["data"]["field_name"]}]"""
        )

    # Get the coordinates of the real sensors
    sensors_coordinates_df = pd.read_sql(
        sensors_query.format(configDict["data"]["arrangement"]), connection
    )
    sensors_coordinates_df.sort_values(
        ["x", "y", "z"], ascending=[True, True, False], inplace=True
    )  # sort sensors in ascending order by x and y values and in descending order by z value
    real_sensors = np.around(sensors_coordinates_df.to_numpy(), 2)

    # Check sensors coordinates correctness
    output_points = pd.read_csv(os.path.join(path, "settings", f"output_points.csv"))
    print("OUTPUT POINTS")
    print(output_points)
    real_sensors_df = pd.DataFrame(data=real_sensors.copy(), columns=["x", "y", "z"])
    real_sensors_df["z"] *= -1
    if len(real_sensors_df.merge(output_points)) != len(real_sensors_df):
        raise Exception("The given sensors are not included in the output points")

    print("REAL SENSORS")
    print(real_sensors)

    """
    frequent_sensors_df = pd.read_sql(
        frequent_sensors_query.format(
            field=configDict["data"]["field_name"],
            scenario=configDict["data"]["scenario_name"],
        ),
        connection,
    )  # take the first batch of the training scenario samples
    frequent_sensors_df = frequent_sensors_df.loc[:, "x":"z"]
    frequent_sensors_df.sort_values(
        ["x", "y", "z"], ascending=[True, True, False], inplace=True
    )  # sort sensors in ascending order by x and y values and in descending order by z value
    synthetic_sensors = np.around(frequent_sensors_df.to_numpy(), 2)

    print("SYNTHETIC SENSORS")
    print(synthetic_sensors)

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

    print("FILTERED SYNTHETIC SENSORS")
    print(filtered_synthetic_sensors)
    """

    dfs_to_concat = []
    scenario = configDict["data"]["scenario_name"]
    scenario_name = re.sub(" |\.", "_", scenario.lower())
    globals()[scenario_name + "_arpae"] = pd.read_sql(arpae_query.format(scenario), connection)
    globals()[scenario_name + "_water"] = pd.read_sql(water_query.format(scenario), connection)
    if configDict["data"]["field_name"].startswith("Real"): # isTuning or configDict["simulation_type"]["isFirstAssimilation"] or configDict["simulation_type"]["isPeriodicAssimilation"] or configDict["simulation_type"]["isForecast"]:
        globals()[scenario_name + "_gp"] = pd.read_sql(gp_query.format(configDict["data"]["field_name"], scenario), connection)
        # print("GP")
        # print(globals()[scenario_name + "_gp"])

    close_db_connection(connection)

    # Pivot weather and irrigation data
    globals()[scenario_name + "_arpae"] = globals()[scenario_name + "_arpae"].pivot(index="unix_timestamp", columns="value_type_name", values="value")
    dfs_to_concat.append(globals()[scenario_name + "_arpae"])
    globals()[scenario_name + "_water"] = globals()[scenario_name + "_water"].pivot(index="unix_timestamp", columns="value_type_name", values="value")
    dfs_to_concat.append(globals()[scenario_name + "_water"])

    if configDict["data"]["field_name"].startswith("Real"): #isTuning or configDict["simulation_type"]["isFirstAssimilation"] or configDict["simulation_type"]["isPeriodicAssimilation"] or configDict["simulation_type"]["isForecast"]:
        # Filter ground potential data
        globals()[scenario_name + "_gp"] = globals()[
                        scenario_name + "_gp"
                    ][
                        [
                            i in real_sensors #filtered_synthetic_sensors
                            for i in zip(
                                globals()[scenario_name + "_gp"].x,
                                globals()[scenario_name + "_gp"].y,
                                globals()[scenario_name + "_gp"].z,
                            )
                        ]
                    ]
        # Pivot ground potential data
        globals()[scenario_name + "_gp"] = globals()[
            scenario_name + "_gp"
        ].pivot(
            index="unix_timestamp",
            columns=["z", "y", "x"],
            values="value",
        )
        globals()[scenario_name + "_gp"].columns = [
            f"z{column[0]}_y{column[1]}_x{column[2]}"
            for column in globals()[
                scenario_name + "_gp"
            ].columns.values
        ]
        globals()[scenario_name + "_gp"] = globals()[
            scenario_name + "_gp"
        ].reindex(
            sorted(
                globals()[scenario_name + "_gp"].columns,
                reverse=True,
            ),
            axis=1,
        )
        dfs_to_concat.append(globals()[scenario_name + "_gp"])

    # Find timestamp intersection boundaries
    top_boundary = max(df.index.min() for df in dfs_to_concat)
    bottom_boundary = min(df.index.max() for df in dfs_to_concat)
    # Concatenate weather, watering and ground potential data
    globals()[scenario_name + "_full"] = pd.concat(
        dfs_to_concat,
        axis=1, join="outer"
    )
    # Keep only common timestamps
    globals()[scenario_name + "_full"] = globals()[scenario_name + "_full"].loc[top_boundary:bottom_boundary]

    # Rename columns
    globals()[scenario_name + "_full"].index.names = ["timestamp"]
    globals()[scenario_name + "_full"].rename(
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
    ]
    if configDict["simulation_type"]["isIrrigation"]:
        cols.append("irrigation")
    if configDict["data"]["field_name"].startswith("Real"): #isTuning or configDict["simulation_type"]["isFirstAssimilation"] or configDict["simulation_type"]["isPeriodicAssimilation"] or configDict["simulation_type"]["isForecast"]:
        cols += list(globals()[scenario_name + "_full"].columns[6:].values)
    globals()[scenario_name + "_full"] = globals()[
        scenario_name + "_full"
    ][cols]

    # Add and impute missing timestamps
    timestamps_range = list(
        range(top_boundary, bottom_boundary + 1, 3600)
    )
    missing_timestamps = [
        timestamp
        for timestamp in timestamps_range
        if timestamp not in globals()[scenario_name + "_full"].index
    ]
    for timestamp in missing_timestamps:
        globals()[scenario_name + "_full"].loc[
            timestamp
        ] = pd.Series(
            data=np.nan,
            index=globals()[scenario_name + "_full"].columns,
        )
    globals()[scenario_name + "_full"].sort_index(inplace=True)
    globals()[scenario_name + "_full"].interpolate(limit_direction="both", inplace=True)


    # Rename sensors columns to real values
    globals()[scenario_name + "_full"].rename(
        columns={
            f"z{col[2]}_y{col[1]}_x{col[0]}": f"z{int(col[2] * -100)}_y{int(col[1] * 100)}_x{int(col[0] * 100)}"
            for col in real_sensors
        },
        inplace=True,
    )

    # Save data in CSV files
    air_temperature_data = pd.DataFrame(globals()[scenario_name + "_full"]["air_temperature"])
    air_temperature_data.to_csv(os.path.join(path, "meteo", "air_temperature.csv"))
    air_humidity_data = pd.DataFrame(globals()[scenario_name + "_full"]["air_humidity"])
    air_humidity_data.to_csv(os.path.join(path, "meteo", "air_humidity.csv"))
    wind_speed_data = pd.DataFrame(globals()[scenario_name + "_full"]["wind_speed"])
    wind_speed_data.to_csv(os.path.join(path, "meteo", "wind_speed.csv"))
    solar_radiation_data = pd.DataFrame(globals()[scenario_name + "_full"]["solar_radiation"])
    solar_radiation_data.to_csv(os.path.join(path, "meteo", "solar_radiation.csv"))
    precipitation_data = pd.DataFrame(globals()[scenario_name + "_full"]["precipitation"])
    precipitation_data.to_csv(os.path.join(path, "water", "precipitation.csv"))
    if configDict["simulation_type"]["isIrrigation"]:
        irrigation_data = pd.DataFrame(globals()[scenario_name + "_full"]["irrigation"])
    else:
        # empty irrigation file (to be filled with the Moreno's Rule)
        irrigation_data = pd.DataFrame(data=np.nan, index=globals()[scenario_name + "_full"].index, columns=["irrigation"])
    irrigation_data.to_csv(os.path.join(path, "water", "irrigation.csv"))
    if configDict["data"]["field_name"].startswith("Real"): #isTuning or configDict["simulation_type"]["isFirstAssimilation"] or configDict["simulation_type"]["isPeriodicAssimilation"] or configDict["simulation_type"]["isForecast"]:
        gp_data = pd.DataFrame(globals()[scenario_name + "_full"].iloc[:, 6:])
        gp_data.to_csv(os.path.join(path, "obs_data", "waterPotential.csv"))
        print("GP DATA") #
        print(gp_data) #


def exportData(db_cfg, configDict, path):
    connection = open_db_connection(db_cfg)

    potential_output_data = pd.read_csv(os.path.join(path, "output", "output.csv"), index_col="timestamp")
    volumetric_output_data = pd.read_csv(os.path.join(path, "output", "outputWaterContent.csv"), index_col="timestamp")
    output_points = pd.read_csv(os.path.join(path, "settings", "output_points.csv"))

    if not configDict["simulation_type"]["isIrrigation"]:
        irrigation_data = pd.read_csv(os.path.join(path, "water", "irrigation.csv"), index_col="timestamp")
        # Populate 'synthetic_scenario_water_data' with irrigation data
        syn_irr_data_rows_list = []
        common_col_dict = {
            "scenario_water_name": configDict["data"]["scenario_name"],
            "value_type_name": "IRRIGATIONS"
        }

        for timestamp, _ in irrigation_data.iterrows():
            syn_irr_data_col_dict = dict(common_col_dict)
            syn_irr_data_col_dict["unix_timestamp"] = timestamp
            syn_irr_data_col_dict["value"] = irrigation_data.loc[timestamp, "irrigation"]
            syn_irr_data_rows_list.append(syn_irr_data_col_dict)

        syn_irr_data_df = pd.DataFrame(syn_irr_data_rows_list)
        print("syn_irr_data_df") #
        print(syn_irr_data_df) #
        syn_irr_data_df.to_sql(name="synthetic_scenario_water_data", con=connection, index=False, if_exists="append")

    # Populate 'synthetic_data' table with potential data
    syn_data_rows_list = []
    common_col_dict = {
        "field_name": configDict["data"]["field_name"],
        "scenario_name": configDict["data"]["scenario_name"],
        "value_type_name": "GROUND_WATER_POTENTIAL"
    }

    for _, sensor in output_points.iterrows():
        for timestamp, _ in potential_output_data.iterrows():
            syn_data_col_dict = dict(common_col_dict)
            syn_data_col_dict["unix_timestamp"] = timestamp
            syn_data_col_dict["x"] = sensor["x"]
            syn_data_col_dict["y"] = sensor["y"]
            syn_data_col_dict["z"] = -sensor["z"] ###
            syn_data_col_dict["value"] = potential_output_data.loc[timestamp, f"""z{int(sensor["z"] * 100)}_y{int(sensor["y"] * 100)}_x{int(sensor["x"] * 100)}"""]
            syn_data_rows_list.append(syn_data_col_dict)

    syn_data_df = pd.DataFrame(syn_data_rows_list)
    print("syn_data_df (potential)") #
    print(syn_data_df) #
    syn_data_df.to_sql(name="synthetic_data", con=connection, index=False, if_exists="append")

    # Populate 'synthetic_data' table with volumetric data
    syn_data_rows_list = []
    common_col_dict["value_type_name"] = "GROUND_SATURATION_DEGREE"
    for _, sensor in output_points.iterrows():
        for timestamp, _ in volumetric_output_data.iterrows():
            syn_data_col_dict = dict(common_col_dict)
            syn_data_col_dict["unix_timestamp"] = timestamp
            syn_data_col_dict["x"] = sensor["x"]
            syn_data_col_dict["y"] = sensor["y"]
            syn_data_col_dict["z"] = -sensor["z"] ###
            syn_data_col_dict["value"] = volumetric_output_data.loc[timestamp, f"""z{int(sensor["z"] * 100)}_y{int(sensor["y"] * 100)}_x{int(sensor["x"] * 100)}"""]
            syn_data_rows_list.append(syn_data_col_dict)

    syn_data_df = pd.DataFrame(syn_data_rows_list)
    print("syn_data_df (volumetric)") #
    print(syn_data_df) #
    syn_data_df.to_sql(name="synthetic_data", con=connection, index=False, if_exists="append")

    # Populate 'synthetic_data_humidity_bins' table
    humidity_bins_query = "SELECT * FROM synthetic_humidity_bins \
            ORDER BY min ASC"
    humidity_bins_df = pd.read_sql(
        humidity_bins_query, connection, index_col="humidity_bin"
    )

    syn_data_hum_bins_rows_list = []
    humidity_bins_count_dict = {bin: 0 for bin in humidity_bins_df.index}
    del common_col_dict["value_type_name"]
    for idx, _ in potential_output_data.iterrows():
        bins_count_dict = dict(humidity_bins_count_dict)
        syn_data_hum_bins_col_dict = dict(common_col_dict)
        syn_data_hum_bins_col_dict["unix_timestamp"] = idx
        # Compute bins count
        for _, sensor in output_points.iterrows():
            for index, col in humidity_bins_df.iterrows():
                if (
                    potential_output_data.loc[
                        idx,
                        f"""z{int(sensor["z"] * 100)}_y{int(sensor["y"] * 100)}_x{int(sensor["x"] * 100)}""",
                    ]
                    >= col["min"]
                    and potential_output_data.loc[
                        idx,
                        f"""z{int(sensor["z"] * 100)}_y{int(sensor["y"] * 100)}_x{int(sensor["x"] * 100)}""",
                    ]
                    < col["max"]
                    ):
                        bins_count_dict[index] += 1
        for bin in bins_count_dict:
            new_syn_data_hum_bins_col_dict = dict(syn_data_hum_bins_col_dict)
            new_syn_data_hum_bins_col_dict["humidity_bin"] = bin
            new_syn_data_hum_bins_col_dict["count"] = bins_count_dict[bin]
            syn_data_hum_bins_rows_list.append(new_syn_data_hum_bins_col_dict)

    syn_data_hum_bins_df = pd.DataFrame(syn_data_hum_bins_rows_list)
    print("syn_data_hum_bins_df") #
    print(syn_data_hum_bins_df) #
    syn_data_hum_bins_df.to_sql(name="synthetic_data_humidity_bins", con=connection, index=False, if_exists="append")

    close_db_connection(connection)
