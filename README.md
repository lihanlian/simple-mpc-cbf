
# Overview

This repository is about using model predictive control (MPC) and control barrier function (CBF) for robot motion planning problem with obstacle avoidance. Two methods are used: one is using eulidean distance (MPC-DC) and the other is using CBF for safety guarantee. Python and CasADi are used for implementation. It is initiated by the final project of **AEROSP 740 - Model Predictive Control (2024 Winter)** at University of Michigan - Ann Arbor.

Disclaimer: This is **NOT** an research project. Some part might not be rigorous and suggestions are welcomed.

**MPC-DC Reuslts**
<p align="center">
  <img alt="Image 1" src="https://github.com/lihanlian/simple-mpc-cbf/figs/animation_mpc_dc_static.gif" width="45%" />
  <img alt="Image 2" src="https://github.com/lihanlian/simple-mpc-cbf/figs/animation_mpc_dc_static_and_dynamic.gif" width="45%" />
</p>
**MPC-CBF ResultS**
<p align="center">
  <img alt="Image 1" src="https://github.com/lihanlian/simple-mpc-cbf/figs/animation_mpc_cbf_static.gif" width="45%" />
  <img alt="Image 2" src="https://github.com/lihanlian/simple-mpc-cbf/figs/animation_mpc_cbf_static_and_dynamic.gif" width="45%" />
</p>


## Run Locally

Clone the project

```bash
  git clone https://github.com/lihanlian/simple-mpc-cbf.git
```

Go to project directory
```bash
  python3 -m venv env && source env/bin/activate 
```
```bash
  pip install -r requirements.txt
```

 - run _mpc_dc_static.py_ and _mpc_dc_static_and_dynamic.py_ to generate pickle files that store robot state and control input information for MPC-DC algorithm.
 - run _mpc_cbf_static.py_ and  _mpc_cbf_static_and_dynamic.py_ to generate pickle files that store robot state and control input information for MPC-CBF algorithm. 
 - run _plot.ipynb_ to load data and visualize results.
 - Adjust hyperparameters (preidction horizon N and cbf parameter γ) if necessary.
 - Some hyperparameters might result in bad result such as moving from one place to another instantaneously in a unreasonable way. This might be due to the nonlinearity and feasibility properties of optimization problem, and the optimization solver fail to get a good solution. 
## Acknowledgements
The author would like to appreciate the course instructor Professor Ilya Kolmanovsky for his help throughout the semester and discussion during office hours. Below are some good resources that might be helpful on related topics.
 - [106B Discussion: Control Barrier Functions](https://www.youtube.com/watch?v=G7OiBjlO07k&t=1506s)
 - [Jason Choi -- Introduction to Control Lyapunov Functions and Control Barrier Functions](https://www.youtube.com/watch?v=_Tkn_Hzo4AA&t=2392s)

## License

[MIT](https://choosealicense.com/licenses/mit/)

