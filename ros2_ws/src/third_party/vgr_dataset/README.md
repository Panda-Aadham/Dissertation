# VGR Olfaction Dataset

This directory is a placeholder for the VGR olfaction dataset used with GADEN.

## Download

The dataset is not included in this repository due to its size.

You can download it from the official source:
https://mapir.isa.uma.es/mapirwebsite/?p=1708

## Setup

1. Download and extract the dataset
2. Place the contents into this folder:

```
ros2_ws/src/third_party/VGR_dataset/
```

The expected file structure should be:

```
vgr_dataset/
├── README.md
├── simulations.csv
├── GADEN_files/
│   ├── package.xml
│   ├── CMakeLists.txt
│   └── scenarios/
│       ├── House01/
│       │   ├── cad_models/
│       │   ├── gas_simulations/
│       │   ├── launch/
│       │   ├── wind_simulations/
│       │   ├── BasicSimScene.yaml
│       │   ├── occupancy.pgm
│       │   ├── occupancy.yaml
│       │   └── OccupancyGrid3D.csv
│       ├── House02/
│       ├── House03/
│       ├── ...
│       ├── House29/
│       └── House30/
│
├── models_FBX/
│   ├── House01(Clone).fbx
│   ├── House02(Clone).fbx
│   ├── ...
│   └── House30(Clone).fbx
│
├── OpenFOAM_files/
│   ├── House01.zip
│   ├── House02.zip
│   ├── ...
│   └── House30.zip
│
└── utils/
    ├── batch_compress.sh
    ├── batch_rename.sh
    └── src/
        ├── createLaunchFiles.cpp
        ├── rename.cpp
        └── writeConcentrations.cpp
```

## Notes

* The dataset contains precomputed gas simulations for use with GADEN
* No compilation is required, but it must be accessible from your ROS workspace
