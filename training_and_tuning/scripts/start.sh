#!/bin/bash
docker stop watering_forecasting
docker rm watering_forecasting
docker run -u $(id -u):$(id -g) --name watering_forecasting --volume $(pwd):/home/watering_forecasting --detach -t ghcr.io/josephgiovanelli/watering_forecasting:0.1.0
docker exec watering_forecasting /bin/bash -c "export PYTHONHASHSEED=42 && python src/run_experiments.py"