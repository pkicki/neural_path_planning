# Learning from Experience for Rapid Generationof Local Car Maneuvers

This repository contains code associated with the paper "Learning from Experience for Rapid Generationof Local Car Maneuvers".

```
Being able to rapidly respond to the changing scene and traffic situations by generating feasible local paths is of pivotal importance for car autonomy.
We propose to train a deep neural network to plan feasible, nearly-optimal paths for kinematically constrained vehicles in small constant time. Our model is trained with a novel weakly supervised approach with a gradient-based policy search.
We demonstrate on real and simulated scenes, and a large set of local planning problems, that
our approach outperforms the existing planners with respect to the number of successfully completed tasks. While the path generation time is only about 40 ms, the generated paths are smooth and comparable to those obtained from conventional path planners.
```

## Dependencies

* Tensorflow 2.1+ (Eager Execution)
* OMPL
* NumPy
* Matplotlib

## How to run

1. Download data:
- download zip from https://anonfiles.com/72rbk4K2o0 and put in `./data`
- `unzip data.zip`
- download zip from https://anonfiles.com/RaJ0h2Kbo8 and put in `./experiments`
- `unzip trained_models.zip`

2. Go to experiments
    ```bash
    cd experiments
    ```
3. Run pretrained planner on the test set
    ```bash
    python planner_test.py
    ```
4. Generate Fig. 8, 9, 11 from the paper
    ```bash
    python exemplary_paths.py
    python geometry_change.py
    python ablation.py
    ```
5. Tou can train your own model (some configuration variables can be set in ```./config_files/eaai.conf```)
    ```bash
    python planner.py --config-file ./config_files/eaai.conf
    ```

6. You can check some results for other planners from OMPL library.
Change the ```time```, ```ALG``` and ```TYPE``` variables to check the accuracy of different planner.
    ```bash
    cd ../ompl_planners
    python aggregate.py
    ```
6. You can also generate the results by your own (but you need to change the "SSL" and "DUBINS" in the code in order to get different tag for your results).
    ```bash
    python dubins.py
    ```
8. Lengths and curvature statistics form Table 1. can be generated with the use of ```len_curv.py```.
    ```bash
    python len_curv.py
    ```
9. Use ```plot.py``` to generate Fig. 10. (values are obtained with multiple runs of ```aggregate.py``` with different tags).
    ```bash
    python plot.py
    ```
   
### Contributions
* `An approach for rapid path generation under differential constraints by approximating the oracle planning function by a neural network`
    - neural network architecture is in class `PlanningNetworkMP` in `./models/planner.py` 
    - training pipeline is in `./experiments/planner.py` 
* `A novel differentiable loss function which penalizes infeasible paths, because they violate constraints imposed either by the vehicle kinematics or the environment map.`
    - function `plan_loss` in `./models/planner.py`
* `Dataset of urban environment local maps (based on real sensory data) and motion planning scenarios that can be used for training and evaluation of local planners for self-driving cars.`
    - data in `./data/`
