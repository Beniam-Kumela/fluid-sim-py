# Real-Time Fluid Dynamics Renderer in Python
This repository contains a Python implementation of modelling real-time fluid dynamics based on the following [paper](https://pages.cs.wisc.edu/~chaol/data/cs777/stam-stable_fluids.pdf).

## Installation
### Windows
To install, press the green "Code" button, download, and extract the zip. Then, navigate to the "dist" folder and run the main.exe file.
### MacOS, Linux
The pre-compiled main.exe file will not work for these systems. Ensure [python](https://www.python.org/downloads/) is installed before proceeding with the steps below.

We need to ensure we have the right dependencies required for the project to run. 
1. First navigate into the root directory of the project 

```
  cd ./fluid-sim-py
```
2. Prepare the virtual environment and install dependencies.

```
    pip install virtualenv 
    virtualenv venv
    source venv/bin/activate
    pip install -r requirements.txt
```
3. (Important) To ensure no errors occur when pygame tries to load shared object files run For more information follow the conversation [here](https://stackoverflow.com/a/72200748).

```
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
```

4. Finally, navigate to the "src" folder and run the main.py file.

```
    cd src/
    python main.py
```

## Examples
The simulation accurately predicts real world phenomena such as the [Kármán vortex street](https://en.wikipedia.org/wiki/K%C3%A1rm%C3%A1n_vortex_street),

| This work | Experimental |
|---------|---------|
| <img width="424" alt="vortex street" src="https://github.com/user-attachments/assets/f3239c8e-b90b-4a94-b4eb-8e1daea9d0f0"> | <img width="424" alt="experimental" src="https://github.com/user-attachments/assets/d08e3e42-d945-4cda-8e25-7ca8cf37504d"> |

why aerodynamic [airfoils](https://en.wikipedia.org/wiki/NACA_airfoil) generate lift for aircraft and birds,

| This work | Experimental |
|---------|---------|
| <img width="395" alt="image" src="https://github.com/user-attachments/assets/0c9f1cf7-23e8-4971-bb4a-057de31113c6"> | <img width="395" alt="Untitled" src="https://github.com/user-attachments/assets/c2336677-3369-482a-aa0c-1150f10a6e4e">

and the transition from laminar to turbulent one might find from burning [incense](https://en.wikipedia.org/wiki/Incense) (actually the [motivation](https://en.wikipedia.org/wiki/Boswellia_papyrifera) for this project).

| This work | Experimental |
|---------|---------|
| <img width="214" alt="smoke" src="https://github.com/user-attachments/assets/8c57c418-2179-417d-8feb-e6bd3eed7257"> | <img width="214" alt="incense" src="https://github.com/user-attachments/assets/83367697-b284-46d0-aa54-1997219328a3">

## Future
I would like to port this code to C++ using OpenGL or SFML to get higher performance, explore different CFD schemes ([LBM](https://en.wikipedia.org/wiki/Lattice_Boltzmann_methods), [DNS](https://en.wikipedia.org/wiki/Direct_numerical_simulation), and [RANS](https://en.wikipedia.org/wiki/Reynolds-averaged_Navier%E2%80%93Stokes_equations)), and add more user customization (colors, solution stepping, drawing boundaries, density sources).
