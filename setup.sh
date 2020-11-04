#!/bin/bash

set -e # exit script if an error occurs



echo ""
echo "########################################"
echo "Setup Virtual Environment"
echo "########################################"
echo ""

python3 -m venv myenv            # Create a new virtual environment (venv) using native python3.7 venv
source myenv/bin/activate        # This replaces the python/pip command with the ones from the venv
which python                     # shoud output: ./myenv/bin/python

pip install --upgrade pip        # Update pip
pip install -r requirements.txt  # Install the exact same packages that we used

# Alternatively you can install globally using pip
# pip install jupyter torch natsort pyyaml opencv-python torchvision skimage scikit-image tqdm lpips pandas environment_kernels 



echo ""
echo "########################################"
echo "Download models, data"
echo "########################################"
echo ""

wget  http://data.vision.ee.ethz.ch/alugmayr/SRFlow_Data_wohThi7Tae0eNoo7ahcu/datasets.zip
unzip datasets.zip
rm datasets.zip

wget  http://data.vision.ee.ethz.ch/alugmayr/SRFlow_Data_wohThi7Tae0eNoo7ahcu/trained_models.zip
unzip trained_models.zip
rm trained_models.zip


echo ""
echo "########################################"
echo "Start Demo"
echo "########################################"
echo ""

./run_jupyter.sh
