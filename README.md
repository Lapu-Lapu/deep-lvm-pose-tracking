# Deep LVM Pose Tracking

Implementation of deep latent variable models for pose estimation.


## Description

This is the repository containing everything related to "Pose Tracking" project
for my research stay in Gunnar Blohm's Lab in Kingston.


## Setup

Please report errors immediately, so I can help and update the docs/code.

1. Use conda to reproduce a tested environment:

`conda env create -f environment.lock.yaml --force`

2. Activate environment:

`conda activate deepLVM`

3. If you have a CUDA (for example version 10) available, install the gpu
   pytorch version:

`conda install pytorch torchvision cudatoolkit=10.0 -c pytorch`

4. Register environments' ipykernel to your Jupyter Lab/Notebook (maybe not
   necessary):

`python -m ipykernel install --user --name deepLVM --display-name "deepLVM"`

5. Install deep_lvm_pose_tracking package (develop sets up symbolic links to
   the package, modifications to the modules therefore have immediate effect).

`python setup.py develop`

6. Pip install github dependencies:

`pip install -r requirements.txt`

7. Run jupyter notebooks with kernel deepLVM.


## Note

This project has been set up using PyScaffold 3.1. For details and usage
information on PyScaffold see https://pyscaffold.org/.
