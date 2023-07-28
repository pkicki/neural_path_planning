# Learning from Experience for Rapid Generationof Local Car Maneuvers

This repository contains code associated with the paper ["Learning from Experience for Rapid Generation of Local Car Maneuvers"](https://www.sciencedirect.com/science/article/pii/S0952197621002475) ([arXiv version](https://arxiv.org/abs/2012.03707)).

```
Being able to rapidly respond to the changing scene and traffic situations by generating feasible local paths is of pivotal importance for car autonomy.
We propose to train a deep neural network to plan feasible, nearly-optimal paths for kinematically constrained vehicles in small constant time. Our model is trained with a novel weakly supervised approach with a gradient-based policy search.
We demonstrate on real and simulated scenes, and a large set of local planning problems, that
our approach outperforms the existing planners with respect to the number of successfully completed tasks. While the path generation time is only about 40 ms, the generated paths are smooth and comparable to those obtained from conventional path planners.
```

## Dependencies

* Python 3.6+
* Tensorflow 2.1+ (Eager Execution)
* OMPL
* NumPy
* Matplotlib
* tqdm

## How to run

1. Download data:
- download zip from https://chmura.put.poznan.pl/s/FH6Edd39nw98y70/download and put in `./data`
- `mv data.zip data && cd ./data && unzip data.zip && mv ./data/* . && rm -r data`
- download zip from https://chmura.put.poznan.pl/s/wy1Whhk4qDTnBx0/download and put in `./experiments`
- `unzip trained_models.zip && mv trained_models ./experiments`

2. Go to experiments
    ```bash
    cd experiments
    ```
3. Run pretrained planner on the test set
    ```bash
    python planner_test.py
    ```
4. Generate Fig. 8, 9, 11 from the paper (without heatmaps)
    ```bash
    python exemplary_paths.py
    python geometry_change.py
    python ablation.py
    ```
5. Tou can train your own model (some configuration variables can be set in ```./config_files/eaai.conf```)
    ```bash
    python planner.py --config-file ./config_files/eaai.conf
    ```
   
### Contributions
* `An approach for rapid path generation under differential constraints by approximating the oracle planning function by a neural network`
    - neural network architecture is in class `PlanningNetworkMP` in `./models/planner.py` 
    - training pipeline is in `./experiments/planner.py` 
* `A novel differentiable loss function which penalizes infeasible paths, because they violate constraints imposed either by the vehicle kinematics or the environment map.`
    - function `plan_loss` in `./models/planner.py`
* `Dataset of urban environment local maps (based on real sensory data) and motion planning scenarios that can be used for training and evaluation of local planners for self-driving cars.`
    - data in `./data/`
 
### Citation
```
@article{kicki2022neuralplanning,
title = {Learning from experience for rapid generation of local car maneuvers},
journal = {Engineering Applications of Artificial Intelligence},
volume = {105},
pages = {104399},
year = {2021},
issn = {0952-1976},
doi = {https://doi.org/10.1016/j.engappai.2021.104399},
url = {https://www.sciencedirect.com/science/article/pii/S0952197621002475},
author = {Piotr Kicki and Tomasz Gawron and Krzysztof Ćwian and Mete Ozay and Piotr Skrzypczyński},
keywords = {Motion planning, Neural networks, Robotics, Autonomous driving, Reinforcement learning, Autonomous vehicle navigation},
}
``` 
