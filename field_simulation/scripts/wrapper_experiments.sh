#!/bin/bash
cd src
python cythonSetup.py build_ext --inplace
cd ..
python src/wrapper.py --path ./data/errano_evaluation --iteration -1 --seed 42
