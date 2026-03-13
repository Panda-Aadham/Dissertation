# VGR GSL Evaluation

This package provides a standalone ROS 2 launch wrapper for running the non-semantic GSL gas source localisation methods on VGR dataset scenarios.

It reuses the existing project stack rather than replacing it:

- prerecorded VGR gas playback data
- the existing GSL action server
- simulated gas and wind sensors
- the existing `BasicSim` and `Nav2` launch pattern

## Supported methods

- `PMFS`
- `GrGSL`
- `ParticleFilter`
- `Spiral`
- `SurgeCast`
- `SurgeSpiral`

Semantic methods are not supported by this wrapper because they require extra ontology, room-mask, and detection inputs that are not present in the plain VGR dataset setup used here.

## Main launch

Run a single scenario with:

```bash
ros2 launch gsl_evaluation main_simbot_launch.py simulation:=1,3-2,4_fast
```

Choose a specific method explicitly with:

```bash
ros2 launch gsl_evaluation main_simbot_launch.py simulation:=1,3-2,4_fast method:=GrGSL
```

Typical useful arguments:

- `scenario:=House01`
- `simulation:=1,3-2,4_fast`
- `method:=PMFS`
- `use_rviz:=True`
- `robot_radius:=0.01`

## Repeated runs

Run the same scenario multiple times without restarting manually:

```bash
ros2 launch gsl_evaluation series_simbot_launch.py runs:=10 scenario:=House02 simulation:=3,5-1_fast method:=PMFS use_rviz:=False
```

The series launch starts one full `main_simbot_launch.py` child launch at a time and waits for it to exit before starting the next run.

By default it uses `run_index:=auto`, so result and variance CSV rows continue from the next available run index. Use `start_run_index:=1` if you want a fixed sequence.

## Batch scenario execution

Run scenarios from `vgr_dataset/simulations.csv` without typing each house manually:

```bash
ros2 launch gsl_evaluation series_simbot_launch.py scenario_set:=first simulation_speed:=fast runs:=3 method:=PMFS use_rviz:=False
```

Useful batch options:

- `scenario_set:=single` keeps the original one-scenario mode
- `scenario_set:=first` runs the first block of 30 rows in `simulations.csv`
- `scenario_set:=second` runs the second block of 30 rows in `simulations.csv`
- `scenario_set:=all` runs both blocks
- `simulation_speed:=fast`, `slow`, or `both`
- `houses:=House01,House02` or `houses:=1-5` limits the batch to selected houses
- `house_start:=1 house_end:=10` limits the batch by numeric house range

## Available House01 simulations

- `1,3-2,4_fast`
- `1,3-2,4_slow`
- `2,4-1_fast`
- `2,4-1_slow`
