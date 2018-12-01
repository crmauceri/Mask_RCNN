#!/bin/sh

# Install any needed packages specified in requirements.txt
if [ ${ENV} = "GPU" ]; then
    pip install --trusted-host pypi.python.org -r requirements_gpu.txt
else
    pip install --trusted-host pypi.python.org -r requirements.txt
fi

python setup.py install

# Compile cocoapi
cd coco/PythonAPI/
make
make install
python setup.py install