# Memristive Neural Networks Simulation Code

This repository contains code that can be used to run simulations with memristors and memristive neural networks. It was created for the first-year research project of my master's degree.

## Installation

It is recommended to install conda and then create an environment for the simulation software using the ```environment.yaml``` file. A suggestion on how to install the simulation software and activate the environment is provided below.

```bash
git clone https://github.com/anpenta/memristive-neural-networks-simulation-code.git
cd memristive-neural-networks-simulation-code
conda env create -f environment.yaml
conda activate memristive-neural-networks
```

## Running the simulations

To run the simulations you can provide commands through the terminal using the ```simulate``` module. An example is given below.

```bash
python3 simulate.py solve_xor 1000 1000 --learn_hidden_synapses --output_to_csv
```
This will run the ```solve_xor``` function with 1000 learning pulses in each update step for 1000 epochs, the model will learn the hidden layer to output layer connections and you will get the training data as a csv file. An example of how to see the arguments for each simulation function is provided below.

```bash
python3 simulate.py solve_xor_snn --help
```

## Sources
* Goossens, A. S., A. Das, and T. Banerjee. "Electric field driven memristive behavior at the Schottky interface of Nb-doped SrTiO3." Journal of Applied Physics 124.15 (2018): 152102.
* Izhikevich, Eugene M. "Simple model of spiking neurons." IEEE Transactions on neural networks 14.6 (2003): 1569-1572.
