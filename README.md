This is the repository for the paper ***AIPlan: Adaptive Soil Moisture Forecasting for Long-Term Irrigation Planning*** submitted to EDBT 2025.



# Structure
As our approach consists of three different phases, this repository has been structured accordingly, with each folder in the root representing a specific approach phase:
1. **field_simulation**,
2. **retraining**, 
3. **training_and_tuning**.

Each folder contains files and folders with the following semantics.

*Folders*:
- ```.devcontainer```, configuration to instantiate a dev container with vscode, so that the user can debug;
- ```.github```, configuration for the github actions and deploy;
- ```data```, raw agro data for the analysis at hand (e.g., weather data);
- ```resources```, configuration for the analysis at hand (e.g., the automl space);
- ```scripts```, starting point for reproducibility;
- ```src```, source code.

*Files*:
- ```.gitattributes``` and ```.gitignore``` are configuration git files,
- ```Dockerfile``` is the configuration file to build the Docker container,
- ```requirements``` lists the required python packages,
- ```LICENSE```, the copyright GNU GENERAL PUBLIC LICENSE.

# Reproducibility

A Docker file is present to build the needed container.
There are two options:
- [JUST RUN] run either ```scripts/start.sh``` (unix) or ```scripts/start.bat``` (windows) to build the container and launch ```scripts/run_experiments.sh```, which already contains a configuration for the file ```src/main.py```. If you want to run something different to the ```src/main.py```, you should either: (i) modify ```scripts/run_experiments.sh``` to do that or (ii) once the container is running, run ```docker exec watering_forecasting [your command]```, in this case ```watering_forecasting``` is the name of the container and ```[your command]``` could be something like ```python file_name.py --param_name param_value``` or ```bash another_script.sh```. This option is usually use in deployment, not in dev.
- [RUN & DEBUG] open vscode, which should suggest you to open the project in the devcontainer. Here, you can both run and debug each file through the vscode interface. In the container, I installed some plugins that both help to maintain a "good-quality" code (e.g., black formatter) and are useful to develop/see results (csv reader). (To notice: if you open a terminal, you will find yourself inside the container, it is like your world is just the container.)


# Database Dump

The whole end-to-end process is extremely time- and resource-consuming. Hence, we provide the results obtained during our experiments via a [Zenodo repository](https://zenodo.org/records/14277535)
