# Speeding up DNN-based planning of local maneuvers via efficient B-spline path construction

This repository contains code associated with the paper "Speeding up DNN-based planning of local maneuvers via efficient B-spline path construction".

```
abstract
```

## Dependencies

* Tensorflow 2.1+ (Eager Execution)
* NumPy
* Matplotlib

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
3. Run pretrained planner on the test set
    ```bash
    python planner_test.py
    ```
4. Generate Fig. 4 from the paper
    ```bash
    python exemplary_paths.py
    ```
5. Tou can train your own model (some configuration variables can be set in ```./config_files/eaai.conf```)
    ```bash
    python planner.py --config-file ./config_files/clamp.conf
    ```