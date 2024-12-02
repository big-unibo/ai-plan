#!/bin/bash
cd src
python cythonSetup.py build_ext --inplace
cd ..
python src/wrapper.py --path ./data/progetto_tuning --num_iterations 150 --seed 42