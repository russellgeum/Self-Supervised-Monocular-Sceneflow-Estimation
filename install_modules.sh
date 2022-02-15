#!/bin/bash
cd ./model_layer/correlation_package
python setup.py install
cd ../forwardwarp_package
python setup.py install
cd ../..

pip install pytz
pip install tqdm==4.40.0
pip install future
pip install natsort
pip install colorama
pip install pypng
pip install numpy
pip install scipy
pip install pandas
pip install Pillow
pip install scikit-image
pip install opencv-python-headless==4.1.2.30
pip install einops
pip install albumentations==0.5.2
pip install jupyter
pip install jupyterthemes
pip install torchsummary
pip install tensorboard
pip install open3d --ignore-installed PyYAML