# Copyright (C) 2019 Andreas Pentaliotis
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

# Utility Module
# Utility functions to run simulations with memristors.

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.size": 12})


def parse_input_arguments():
  parser = argparse.ArgumentParser()
  subparsers = parser.add_subparsers(dest="simulation_function")
  subparsers.required = True

  pulsing_experiment_parser = subparsers.add_parser("pulsing_experiment")
  model_xor_parser = subparsers.add_parser("model_xor")
  add_solve_xor_parser(subparsers)
  add_solve_xor_complex_parser(subparsers)
  model_xor_snn_parser = subparsers.add_parser("model_xor_snn")
  add_solve_xor_snn_parser(subparsers)
  add_solve_xor_complex_snn_parser(subparsers)

  return parser.parse_args()


def add_solve_xor_parser(subparsers):
  solve_xor_parser = subparsers.add_parser("solve_xor")
  solve_xor_parser.add_argument("learning_pulses", type=int,
                                help="Number of learning pulses to apply in each update step")
  solve_xor_parser.add_argument("epochs", type=int, help="Number of training epochs")
  solve_xor_parser.add_argument("--learn_hidden_synapses", dest="learn_hidden_synapses", action="store_true",
                                help="Whether to learn the hidden to output connections or not")
  solve_xor_parser.set_defaults(learn_hidden_synapses=False)
  solve_xor_parser.add_argument("--output_to_csv", dest="output_to_csv", action="store_true",
                                help="Whether to output the training data in a csv file or not")
  solve_xor_parser.set_defaults(output_to_csv=False)


def add_solve_xor_complex_parser(subparsers):
  solve_xor_complex_parser = subparsers.add_parser("solve_xor_complex")
  solve_xor_complex_parser.add_argument("learning_pulses", type=int,
                                        help="Number of learning pulses to apply in each update step")
  solve_xor_complex_parser.add_argument("epochs", type=int, help="Number of training epochs")
  solve_xor_complex_parser.add_argument("--output_to_csv", dest="output_to_csv", action="store_true",
                                        help="Whether to output the training data in a csv file or not")
  solve_xor_complex_parser.set_defaults(output_to_csv=False)


def add_solve_xor_snn_parser(subparsers):
  solve_xor_snn_parser = subparsers.add_parser("solve_xor_snn")
  solve_xor_snn_parser.add_argument("learning_pulses", type=int,
                                    help="Number of learning pulses to apply in each update step")
  solve_xor_snn_parser.add_argument("epochs", type=int, help="Number of training epochs")
  solve_xor_snn_parser.add_argument("--learn_hidden_synapses", dest="learn_hidden_synapses", action="store_true",
                                    help="Whether to learn the hidden to output connections or not")
  solve_xor_snn_parser.set_defaults(learn_hidden_synapses=False)
  solve_xor_snn_parser.add_argument("--output_to_csv", dest="output_to_csv", action="store_true",
                                    help="Whether to output the training data in a csv file or not")
  solve_xor_snn_parser.set_defaults(output_to_csv=False)


def add_solve_xor_complex_snn_parser(subparsers):
  solve_xor_complex_snn_parser = subparsers.add_parser("solve_xor_complex_snn")
  solve_xor_complex_snn_parser.add_argument("learning_pulses", type=int,
                                            help="Number of learning pulses to apply in each update step")
  solve_xor_complex_snn_parser.add_argument("epochs", type=int, help="Number of training epochs")
  solve_xor_complex_snn_parser.add_argument("--output_to_csv", dest="output_to_csv", action="store_true",
                                            help="Whether to output the training data in a csv file or not")
  solve_xor_complex_snn_parser.set_defaults(output_to_csv=False)


def save_data(dataframe, directory_path, filename):
  if not os.path.isdir(directory_path):
    print("Output directory does not exist | Creating directories along directory path")
    os.makedirs(directory_path)

  filename = filename + ".csv"
  print("Saving data | Filename: {} | Directory path: {}".format(filename, directory_path))
  dataframe.to_csv("{}/{}".format(directory_path, filename), index=False)


def plot_mse(dataframe):
  # Plot the MSE as a function of epoch.
  mse = dataframe.groupby("epoch")["squared_error"].mean()
  plt.plot(mse)
  plt.xlabel("Epoch")
  plt.ylabel("MSE")
  plt.title("Learning pulses = " + str(dataframe["learning_pulses"].unique()[0]))
  plt.show()


def invert(x):
  # Add zero to avoid returning a negative zero.
  return -x + 0


def normalize(current):
  return current * 10 ** 6


def burst(spikes, threshold):
  return spikes >= threshold


def is_false_negative(error):
  return error > 0


def is_false_positive(error):
  return error < 0


def add_noise(current):
  if current != 0:
    noise = np.random.normal(0, 0.1, 1)
    current += noise
  return current
