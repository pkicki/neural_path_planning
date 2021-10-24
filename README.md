# Speeding up DNN-based planning of local maneuvers via efficient B-spline path construction

This repository contains code associated with the paper "Speeding up DNN-based planning of local maneuvers via efficient B-spline path construction".

```
This paper demonstrates how an efficient representation of the planned
path using B-splines, and a construction procedure that takes advantage
of the neural network's inductive bias speed up both the inference and
training of a DNN-based motion planner.
We build upon our recent work on learning local car maneuvers from past
experience using a DNN architecture, but we introduce a novel B-spline
path construction method, making it possible to generate local maneuvers in
almost constant time of about 11 ms, respecting a number of constraints
imposed by the environment map and the kinematics of a car-like vehicle.
We evaluate thoroughly the new planner employing the recent Bench-MR framework 
to obtain quantitative results showing that our DNN-based
procedure outperforms state-of-the-art planners by a large margin in the considered task.
```

## Dependencies

* Tensorflow 2.1+ (Eager Execution)
* NumPy
* Matplotlib
* tqdm

## How to run

1. Download data:
- download zip from https://chmura.put.poznan.pl/s/FH6Edd39nw98y70/download and put in `./data`
- `mv data.zip data && cd ./data && unzip data.zip && mv ./data/* . && rm -r data`
- download zip from https://chmura.put.poznan.pl/s/KdmYPDjTnnCYiDe and put in `./experiments`
- `unzip trained_models.zip && mv trained_models ./experiments`

2. Go to experiments
    ```bash
    cd experiments
    ```
3. Run pretrained planner on the test set (inside the script you can choose different models, what allows to generate Fig. 4)
    ```bash
    python planner_test.py
    ```
4. Generate Fig. 6 from the paper
    ```bash
    python exemplary_paths.py
    ```
5. Tou can train your own model (some configuration variables can be set in ```./config_files/icra.conf```)
    ```bash
    python planner.py --config-file ./config_files/icra.conf
    ```
