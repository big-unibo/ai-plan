#!/bin/bash
docker stop watering_simulator
docker rm watering_simulator
docker run -u $(id -u):$(id -g) --name watering_simulator --volume $(pwd):/home --detach -t ghcr.io/josephgiovanelli/synthetic-soil-simulator:0.2.0
docker exec watering_simulator bash ./scripts/wrapper_experiments.sh
