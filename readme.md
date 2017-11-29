# Wildpyre

A Semi-Continuous Wildfire Model using the Heat Equation.

## Prerequisites

For this project you need python3, as well as numpy, matplotlib and PIL

These can be installed with pip

```
pip install numpy
```

```
pip install matplotlib
```

```
pip install pillow
```

## How to use

Create an Simulator object with the parameters you see fit and then run it with .Run() (for simulation and possible visualization) or .CreateGIF() (to create a video... uses FuncAnimation form matplotlib and so the possible formats will depend on the writers available).
