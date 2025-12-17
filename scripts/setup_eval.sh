#!/bin/bash
# Setup script for evaluation environment

# Install requirements
pip install -r requirements.txt

# Install additional dependencies for evaluation
pip install tensorflow
pip install numpy==1.26.2
pip install scipy