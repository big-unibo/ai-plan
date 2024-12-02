docker stop watering_forecasting
docker rm watering_forecasting
docker build -t watering_forecasting .
docker run --name watering_forecasting --volume %cd%:/home/${USER} --detach -t watering_forecasting
docker exec watering_forecasting python src/run_experiments.py