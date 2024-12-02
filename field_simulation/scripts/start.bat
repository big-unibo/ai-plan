docker stop watering_forecasting
docker rm watering_forecasting
docker build -t watering_forecasting .
docker run --name watering_forecasting --volume %cd%:/home --detach -t watering_forecasting
docker exec watering_forecasting bash ./scripts/wrapper_experiments.sh %1 %2