#!/bin/sh

# Install Tkinter
# apt-get update --assume-yes
# apt-get install python3-tk --assume-yes

# Install any needed packages specified in requirements.txt
pip install --trusted-host pypi.python.org -r requirements.txt
python setup.py install

# Compile cocoapi
cd coco/PythonAPI/
make
make install
python setup.py install