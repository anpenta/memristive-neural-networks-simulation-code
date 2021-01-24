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

# XOR SNN Module
# Functions to simulate the xor problem with memristive spiking
# neural networks.

import pandas as pd

import memristor
import spiking_neuron
import utility


# model_xor_snn: Models the XOR problem using a memristive spiking neural network with
# pre-fixed resistance values in the synapses.
def model_xor_snn():
  hrs_set_voltage = -4
  positive_bias = 1
  fixing_pulses = 37 * 10 ** 4
  current_application_time = 500
  spike = 1
  true = 20
  false = 0
  input_spikes_1 = [false, false, true, true]
  input_spikes_2 = [false, true, false, true]
  targets = [false, true, true, false]

  # Create a network  with 2 input neurons, 2 hidden layer neurons
  # and 1 output neuron. The input neurons are simulated by their spikes.
  hidden_neuron_1 = spiking_neuron.SpikingNeuron()
  hidden_neuron_2 = spiking_neuron.SpikingNeuron()
  output_neuron = spiking_neuron.SpikingNeuron()

  # Create 6 synapses with input_synapse_ij representing the connection
  # between the input neuron i and the hidden neuron j, and hidden_synapse_k
  # representing the connection between the hidden neuron k and the output neuron.
  # The inhibitory synapses are input_synapse_12 and input_synapse_21.
  input_synapse_11 = memristor.Memristor()
  input_synapse_12 = memristor.Memristor()
  input_synapse_21 = memristor.Memristor()
  input_synapse_22 = memristor.Memristor()
  hidden_synapse_1 = memristor.Memristor()
  hidden_synapse_2 = memristor.Memristor()

  # Set all the synapses to a high resistance state.
  input_synapse_11.set_resistance(hrs_set_voltage)
  input_synapse_12.set_resistance(hrs_set_voltage)
  input_synapse_21.set_resistance(hrs_set_voltage)
  input_synapse_22.set_resistance(hrs_set_voltage)
  hidden_synapse_1.set_resistance(hrs_set_voltage)
  hidden_synapse_2.set_resistance(hrs_set_voltage)

  # Fix the resistance values in the synapses by applying consecutive
  # positive bias voltage pulses.
  for _ in range(fixing_pulses):
    input_synapse_11.apply_voltage(positive_bias)
    input_synapse_12.apply_voltage(positive_bias)
    input_synapse_21.apply_voltage(positive_bias)
    input_synapse_22.apply_voltage(positive_bias)
    hidden_synapse_1.apply_voltage(positive_bias)
    hidden_synapse_2.apply_voltage(positive_bias)

  c_11 = []
  c_12 = []
  c_21 = []
  c_22 = []
  hidden_spikes_1 = []
  hidden_spikes_2 = []
  z_1 = []
  z_2 = []
  output_spikes = []
  for spikes_1, spikes_2 in zip(input_spikes_1, input_spikes_2):
    # The normalized current c_ij passes through the input_synapse_ij.
    # The current in the inhibitory synapses is inverted. The input voltage
    # is 1 V if the input neuron fired as many pulses as the encoding of TRUE
    # or more, else it is 0 V.
    c_11.append(utility.normalize(utility.burst(spikes_1, true) / input_synapse_11.read_resistance()))
    c_12.append(utility.invert(utility.normalize(utility.burst(spikes_1, true) / input_synapse_12.read_resistance())))
    c_21.append(utility.invert(utility.normalize(utility.burst(spikes_2, true) / input_synapse_21.read_resistance())))
    c_22.append(utility.normalize(utility.burst(spikes_2, true) / input_synapse_22.read_resistance()))

    hidden_spikes_1.append(hidden_neuron_1.apply_current(c_11[-1] + c_21[-1],
                                                         current_application_time)[0].count(spike))
    hidden_spikes_2.append(hidden_neuron_2.apply_current(c_12[-1] + c_22[-1],
                                                         current_application_time)[0].count(spike))

    # The normalized curent z_i passes through the hidden_synapse_i. The
    # input voltage is 1 V if the hidden neuron fired as many pulses as
    # the encoding of TRUE or more, else it is 0 V.
    z_1.append(utility.normalize(utility.burst(hidden_spikes_1[-1], true) / hidden_synapse_1.read_resistance()))
    z_2.append(utility.normalize(utility.burst(hidden_spikes_2[-1], true) / hidden_synapse_2.read_resistance()))

    output_spikes.append(output_neuron.apply_current(z_1[-1] + z_2[-1],
                                                     current_application_time)[0].count(spike))

    # The neurons are resting until the next input.
    hidden_neuron_1.rest()
    hidden_neuron_2.rest()
    output_neuron.rest()

  # Store the data into a data frame.
  data = pd.DataFrame()
  data["input_spikes_1"] = input_spikes_1
  data["input_spikes_2"] = input_spikes_2
  data["c_11"] = c_11
  data["c_12"] = c_12
  data["c_21"] = c_21
  data["c_22"] = c_22
  data["hidden_spikes_1"] = hidden_spikes_1
  data["hidden_spikes_2"] = hidden_spikes_2
  data["z_1"] = z_1
  data["z_2"] = z_2
  data["output_spikes"] = output_spikes
  data["target"] = targets
  data["current_application_time"] = [current_application_time] * 4

  print(data)


# solve_xor_snn: Solves the XOR problem using a memristive spiking neural network
# that learns the resistance values in the synapses using online learning. If
# learn_hidden_synapses is set to False then only the resistance values between
# the input layer and the hidden layer are learned.
def solve_xor_snn(learning_pulses, epochs, learn_hidden_synapses=False, output_to_csv=False):
  hrs_set_voltage = -4
  positive_bias = 1
  fixing_pulses = 37 * 10 ** 4
  current_application_time = 500
  spike = 1
  true = 20
  false = 0
  training_voltage = 1
  input_spikes_1 = [false, false, true, true]
  input_spikes_2 = [false, true, false, true]
  targets = [false, true, true, false]

  # Create a network with 2 input neurons, 2 hidden layer neurons
  # and 1 output neuron. The input neurons are simulated by their spikes.
  hidden_neuron_1 = spiking_neuron.SpikingNeuron()
  hidden_neuron_2 = spiking_neuron.SpikingNeuron()
  output_neuron = spiking_neuron.SpikingNeuron()

  # Create 6 synapses with input_synapse_ij representing the connection
  # between the input neuron i and the hidden neuron j, and hidden_synapse_k
  # representing the connection between the hidden neuron k and the output neuron.
  # The inhibitory synapse is hidden_synapse_2.
  input_synapse_11 = memristor.Memristor()
  input_synapse_12 = memristor.Memristor()
  input_synapse_21 = memristor.Memristor()
  input_synapse_22 = memristor.Memristor()
  hidden_synapse_1 = memristor.Memristor()
  hidden_synapse_2 = memristor.Memristor()

  # Set all the synapses to a high resistance state.
  input_synapse_11.set_resistance(hrs_set_voltage)
  input_synapse_12.set_resistance(hrs_set_voltage)
  input_synapse_21.set_resistance(hrs_set_voltage)
  input_synapse_22.set_resistance(hrs_set_voltage)
  hidden_synapse_1.set_resistance(hrs_set_voltage)
  hidden_synapse_2.set_resistance(hrs_set_voltage)

  # If the resistance values at the hidden synapses are not going
  # to be learned, then fix them by applying consecutive positive
  # bias voltage pulses.
  if not learn_hidden_synapses:
    for _ in range(fixing_pulses):
      hidden_synapse_1.apply_voltage(positive_bias)
      hidden_synapse_2.apply_voltage(positive_bias)

  c_11 = []
  c_12 = []
  c_21 = []
  c_22 = []
  hidden_spikes_1 = []
  hidden_spikes_2 = []
  z_1 = []
  z_2 = []
  output_spikes = []
  errors = []
  squared_errors = []
  epoch_numbers = []
  for epoch_number in range(1, epochs + 1):
    for spikes_1, spikes_2, target in zip(input_spikes_1, input_spikes_2, targets):
      epoch_numbers.append(epoch_number)

      # The normalized current c_ij passes through the input_synapse_ij.
      # The input voltage is 1 V if the input neuron fired as many pulses
      # as the encoding of TRUE or more, else it is 0 V.
      c_11.append(utility.normalize(utility.burst(spikes_1, true) / input_synapse_11.read_resistance()))
      c_12.append(utility.normalize(utility.burst(spikes_1, true) / input_synapse_12.read_resistance()))
      c_21.append(utility.normalize(utility.burst(spikes_2, true) / input_synapse_21.read_resistance()))
      c_22.append(utility.normalize(utility.burst(spikes_2, true) / input_synapse_22.read_resistance()))

      hidden_spikes_1.append(hidden_neuron_1.apply_current(c_11[-1] + c_21[-1],
                                                           current_application_time)[0].count(spike))
      hidden_spikes_2.append(hidden_neuron_2.apply_current(c_12[-1] + c_22[-1],
                                                           current_application_time)[0].count(spike))

      # The normalized current z_i passes through the hidden_synapse_i.
      # The current in the inhibitory synapse is inverted. The input
      # voltage is 1 V if the hidden neuron fired as many pulses as
      # the encoding of TRUE or more, else it is 0 V.
      z_1.append(utility.normalize(utility.burst(hidden_spikes_1[-1], true) / hidden_synapse_1.read_resistance()))
      z_2.append(utility.invert(utility.normalize(utility.burst(hidden_spikes_2[-1], true)
                                                  / hidden_synapse_2.read_resistance())))

      output_spikes.append(output_neuron.apply_current(z_1[-1] + z_2[-1],
                                                       current_application_time)[0].count(spike))

      errors.append(target - output_spikes[-1])
      squared_errors.append(errors[-1] ** 2)

      # Update the synapses based on the error.
      # If the output neuron should have fired more times and it did not,
      # then strengthen hidden synapse 1 if hidden neuron 1 fired as many
      # pulses as the encoding of TRUE or more and if the resistance values
      # at the hidden synapses are to be learned. Otherwise strengthen the
      # input synapses of the input neurons that fired as many pulses as the
      # encoding of TRUE or more and that are connected to hidden neuron 1.
      if utility.is_false_negative(errors[-1]):
        if utility.burst(hidden_spikes_1[-1], true) and learn_hidden_synapses:
          for _ in range(learning_pulses):
            hidden_synapse_1.apply_voltage(training_voltage)
        else:
          if utility.burst(spikes_1, true):
            for _ in range(learning_pulses):
              input_synapse_11.apply_voltage(training_voltage)
          if utility.burst(spikes_2, true):
            for _ in range(learning_pulses):
              input_synapse_21.apply_voltage(training_voltage)
      # If the output neuron should not have fired and it did, then do the
      # equivalent update for hidden synapse 2 or for the input synapses that
      # are connected to hidden neuron 2.
      elif utility.is_false_positive(errors[-1]):
        if utility.burst(hidden_spikes_2[-1], true) and learn_hidden_synapses:
          for _ in range(learning_pulses):
            hidden_synapse_2.apply_voltage(training_voltage)
        else:
          if utility.burst(spikes_1, true):
            for _ in range(learning_pulses):
              input_synapse_12.apply_voltage(training_voltage)
          if utility.burst(spikes_2, true):
            for _ in range(learning_pulses):
              input_synapse_22.apply_voltage(training_voltage)

      # The neurons are resting until the next input.
      hidden_neuron_1.rest()
      hidden_neuron_2.rest()
      output_neuron.rest()

  # Store the data into a data frame.
  data = pd.DataFrame()
  data["input_spikes_1"] = input_spikes_1 * epochs
  data["input_spikes_2"] = input_spikes_2 * epochs
  data["c_11"] = c_11
  data["c_12"] = c_12
  data["c_21"] = c_21
  data["c_22"] = c_22
  data["hidden_spikes_1"] = hidden_spikes_1
  data["hidden_spikes_2"] = hidden_spikes_2
  data["z_1"] = z_1
  data["z_2"] = z_2
  data["output_spikes"] = output_spikes
  data["target"] = targets * epochs
  data["error"] = errors
  data["squared_error"] = squared_errors
  data["epoch"] = epoch_numbers
  data["learning_pulses"] = [learning_pulses] * epochs * 4
  data["training_voltage"] = [training_voltage] * epochs * 4
  data["learn_hidden_synapses"] = [learn_hidden_synapses] * epochs * 4
  data["current_application_time"] = [current_application_time] * epochs * 4

  if output_to_csv:
    utility.save_data(data, "./output", "solve-xor-snn")

  # Plot the MSE as a function of epoch.
  utility.plot_mse(data)


# solve_xor_complex_snn: Solves the XOR problem using a complex memristive
# spiking neural network that learns the resistance values in the synapses
# using online learning.
def solve_xor_complex_snn(learning_pulses, epochs, output_to_csv=False):
  hrs_set_voltage = -4
  current_application_time = 500
  spike = 1
  training_voltage = 1
  true = 20
  false = 0
  input_spikes_1 = [false, false, true, true]
  input_spikes_2 = [false, true, false, true]
  targets = [false, true, true, false]

  # Create a network  with 2 input neurons, 3 hidden layer neurons
  # and 1 output neuron. The input neurons are simulated by their spikes.
  hidden_neuron_1 = spiking_neuron.SpikingNeuron()
  hidden_neuron_2 = spiking_neuron.SpikingNeuron()
  hidden_neuron_3 = spiking_neuron.SpikingNeuron()
  output_neuron = spiking_neuron.SpikingNeuron()

  # Create 7 synapses with pos_input_synapse_ij representing the excitatory
  # connection between the input neuron i and the hidden neuron j, and
  # pos_hidden_synapse_k representing the excitatory connection between
  # the hidden neuron k and the output neuron.
  pos_input_synapse_11 = memristor.Memristor()
  pos_input_synapse_13 = memristor.Memristor()
  pos_input_synapse_22 = memristor.Memristor()
  pos_input_synapse_23 = memristor.Memristor()
  pos_hidden_synapse_1 = memristor.Memristor()
  pos_hidden_synapse_2 = memristor.Memristor()
  pos_hidden_synapse_3 = memristor.Memristor()

  # Create 7 synapses with neg_input_synapse_ij representing the inhibitory
  # connection between the input neuron i and the hidden neuron j, and
  # neg_hidden_synapse_k representing the inhibitory connection between
  # the hidden neuron k and the output neuron.
  neg_input_synapse_11 = memristor.Memristor()
  neg_input_synapse_13 = memristor.Memristor()
  neg_input_synapse_22 = memristor.Memristor()
  neg_input_synapse_23 = memristor.Memristor()
  neg_hidden_synapse_1 = memristor.Memristor()
  neg_hidden_synapse_2 = memristor.Memristor()
  neg_hidden_synapse_3 = memristor.Memristor()

  # Set all the synapses to a high resistance state.
  pos_input_synapse_11.set_resistance(hrs_set_voltage)
  pos_input_synapse_13.set_resistance(hrs_set_voltage)
  pos_input_synapse_22.set_resistance(hrs_set_voltage)
  pos_input_synapse_23.set_resistance(hrs_set_voltage)
  pos_hidden_synapse_1.set_resistance(hrs_set_voltage)
  pos_hidden_synapse_2.set_resistance(hrs_set_voltage)
  pos_hidden_synapse_3.set_resistance(hrs_set_voltage)

  neg_input_synapse_11.set_resistance(hrs_set_voltage)
  neg_input_synapse_13.set_resistance(hrs_set_voltage)
  neg_input_synapse_22.set_resistance(hrs_set_voltage)
  neg_input_synapse_23.set_resistance(hrs_set_voltage)
  neg_hidden_synapse_1.set_resistance(hrs_set_voltage)
  neg_hidden_synapse_2.set_resistance(hrs_set_voltage)
  neg_hidden_synapse_3.set_resistance(hrs_set_voltage)

  c_11 = []
  c_13 = []
  c_22 = []
  c_23 = []
  hidden_spikes_1 = []
  hidden_spikes_2 = []
  hidden_spikes_3 = []
  z_1 = []
  z_2 = []
  z_3 = []
  output_spikes = []
  errors = []
  squared_errors = []
  epoch_numbers = []
  for epoch_number in range(1, epochs + 1):
    for spikes_1, spikes_2, target in zip(input_spikes_1, input_spikes_2, targets):
      epoch_numbers.append(epoch_number)

      # The normalized current c_ij passes through the input_synapse_ij. The input
      # voltage is 1 V if the input neuron fired as many pulses as the encoding of
      # TRUE or more, else it is 0 V.
      pos_c_11 = utility.normalize(utility.burst(spikes_1, true) / pos_input_synapse_11.read_resistance())
      pos_c_13 = utility.normalize(utility.burst(spikes_1, true) / pos_input_synapse_13.read_resistance())
      pos_c_22 = utility.normalize(utility.burst(spikes_2, true) / pos_input_synapse_22.read_resistance())
      pos_c_23 = utility.normalize(utility.burst(spikes_2, true) / pos_input_synapse_23.read_resistance())

      # The normalized current neg_c_ij passes through the neg_input_synapse_ij.
      # The current is inverted to represent the inhibitory connection. The input
      # voltage is 1 V if the input neuron fired as many pulses as the encoding
      # of TRUE or more, else it is 0 V.
      neg_c_11 = utility.invert(utility.normalize(utility.burst(spikes_1, true)
                                                  / neg_input_synapse_11.read_resistance()))
      neg_c_13 = utility.invert(utility.normalize(utility.burst(spikes_1, true)
                                                  / neg_input_synapse_13.read_resistance()))
      neg_c_22 = utility.invert(utility.normalize(utility.burst(spikes_2, true)
                                                  / neg_input_synapse_22.read_resistance()))
      neg_c_23 = utility.invert(utility.normalize(utility.burst(spikes_2, true)
                                                  / neg_input_synapse_23.read_resistance()))

      c_11.append(pos_c_11 + neg_c_11)
      c_13.append(pos_c_13 + neg_c_13)
      c_22.append(pos_c_22 + neg_c_22)
      c_23.append(pos_c_23 + neg_c_23)

      hidden_spikes_1.append(hidden_neuron_1.apply_current(c_11[-1],
                                                           current_application_time)[0].count(spike))
      hidden_spikes_2.append(hidden_neuron_2.apply_current(c_22[-1],
                                                           current_application_time)[0].count(spike))
      hidden_spikes_3.append(hidden_neuron_3.apply_current(c_13[-1] + c_23[-1],
                                                           current_application_time)[0].count(spike))

      # The normalized curent pos_z_i passes through the pos_hidden_synapse_i.
      # The input voltage is 1 V if the hidden neuron fired as many pulses as
      # the encoding of TRUE or more, else it is 0 V.
      pos_z_1 = utility.normalize(utility.burst(hidden_spikes_1[-1], true) / pos_hidden_synapse_1.read_resistance())
      pos_z_2 = utility.normalize(utility.burst(hidden_spikes_2[-1], true) / pos_hidden_synapse_2.read_resistance())
      pos_z_3 = utility.normalize(utility.burst(hidden_spikes_3[-1], true) / pos_hidden_synapse_3.read_resistance())

      # The normalized curent neg_z_i passes through the neg_hidden_synapse_i.
      # The current is inverted to represent the inhibitory connection. The
      # input voltage is 1 V if the hidden neuron fired as many pulses as the
      # encoding of TRUE or more, else it is 0 V.
      neg_z_1 = utility.invert(utility.normalize(utility.burst(hidden_spikes_1[-1], true)
                                                 / neg_hidden_synapse_1.read_resistance()))
      neg_z_2 = utility.invert(utility.normalize(utility.burst(hidden_spikes_2[-1], true)
                                                 / neg_hidden_synapse_2.read_resistance()))
      neg_z_3 = utility.invert(utility.normalize(utility.burst(hidden_spikes_3[-1], true)
                                                 / neg_hidden_synapse_3.read_resistance()))

      z_1.append(pos_z_1 + neg_z_1)
      z_2.append(pos_z_2 + neg_z_2)
      z_3.append(pos_z_3 + neg_z_3)

      output_spikes.append(output_neuron.apply_current(z_1[-1] + z_2[-1] + z_3[-1],
                                                       current_application_time)[0].count(spike))

      errors.append(target - output_spikes[-1])
      squared_errors.append(errors[-1] ** 2)

      # Update the synapses based on the error.
      # If the output neuron should have fired more times and it did not,
      # then strengthen the excitatory hidden synapses of the hidden neurons
      # that fired as many pulses as the encoding of TRUE or more. Otherwise,
      # do the equivalent update for the input synapses of the input neurons
      # that fired as many pulses as the encoding of TRUE or more.
      if utility.is_false_negative(errors[-1]):
        if (utility.burst(hidden_spikes_1[-1], true) or utility.burst(hidden_spikes_2[-1], true)
           or utility.burst(hidden_spikes_3[-1], true)):
          if utility.burst(hidden_spikes_1[-1], true):
            for _ in range(learning_pulses):
              pos_hidden_synapse_1.apply_voltage(training_voltage)
          if utility.burst(hidden_spikes_2[-1], true):
            for _ in range(learning_pulses):
              pos_hidden_synapse_2.apply_voltage(training_voltage)
          if utility.burst(hidden_spikes_3[-1], true):
            for _ in range(learning_pulses):
              pos_hidden_synapse_3.apply_voltage(training_voltage)
        else:
          if utility.burst(spikes_1, true):
            for _ in range(learning_pulses):
              pos_input_synapse_11.apply_voltage(training_voltage)
              pos_input_synapse_13.apply_voltage(training_voltage)
          if utility.burst(spikes_2, true):
            for _ in range(learning_pulses):
              pos_input_synapse_22.apply_voltage(training_voltage)
              pos_input_synapse_23.apply_voltage(training_voltage)
      # If the output neuron should not have fired and it did, then strengthen
      # the inhibitory hidden synapses of the hidden neurons that fired as many
      # pulses as the encoding of TRUE or more.
      elif utility.is_false_positive(errors[-1]):
        if utility.burst(hidden_spikes_1[-1], true):
          for _ in range(learning_pulses):
            neg_hidden_synapse_1.apply_voltage(training_voltage)
        if utility.burst(hidden_spikes_2[-1], true):
          for _ in range(learning_pulses):
            neg_hidden_synapse_2.apply_voltage(training_voltage)
        if utility.burst(hidden_spikes_3[-1], true):
          for _ in range(learning_pulses):
            neg_hidden_synapse_3.apply_voltage(training_voltage)

      # The neurons are resting until the next input.
      hidden_neuron_1.rest()
      hidden_neuron_2.rest()
      hidden_neuron_3.rest()
      output_neuron.rest()

  # Store the data into a data frame.
  data = pd.DataFrame()
  data["input_spikes_1"] = input_spikes_1 * epochs
  data["input_spikes_2"] = input_spikes_2 * epochs
  data["c_11"] = c_11
  data["c_13"] = c_13
  data["c_22"] = c_22
  data["c_23"] = c_23
  data["hidden_spikes_1"] = hidden_spikes_1
  data["hidden_spikes_2"] = hidden_spikes_2
  data["hidden_spikes_3"] = hidden_spikes_3
  data["z_1"] = z_1
  data["z_2"] = z_2
  data["z_3"] = z_3
  data["output_spikes"] = output_spikes
  data["target"] = targets * epochs
  data["error"] = errors
  data["squared_error"] = squared_errors
  data["epoch"] = epoch_numbers
  data["learning_pulses"] = [learning_pulses] * epochs * 4
  data["training_voltage"] = [training_voltage] * epochs * 4
  data["current_application_time"] = [current_application_time] * epochs * 4

  if output_to_csv:
    utility.save_data(data, "./output", "solve-xor-complex-snn")

  # Plot the MSE as a function of epoch.
  utility.plot_mse(data)


# solve_xor_complex_noisy_snn: Solves the XOR problem using a complex memristive
# spiking neural network that learns the resistance values in the synapses
# using online learning and has noisy input current to the nodes.
def solve_xor_complex_noisy_snn(learning_pulses, epochs, output_to_csv=False):
  hrs_set_voltage = -4
  current_application_time = 500
  spike = 1
  training_voltage = 1
  true = 20
  false = 0
  input_spikes_1 = [false, false, true, true]
  input_spikes_2 = [false, true, false, true]
  targets = [false, true, true, false]

  # Create a network  with 2 input neurons, 2 hidden layer neurons
  # and 1 output neuron. The input neurons are simulated by their spikes.
  hidden_neuron_1 = spiking_neuron.SpikingNeuron()
  hidden_neuron_2 = spiking_neuron.SpikingNeuron()
  output_neuron = spiking_neuron.SpikingNeuron()

  # Create 6 synapses with pos_input_synapse_ij representing the excitatory
  # connection between the input neuron i and the hidden neuron j, and
  # pos_hidden_synapse_k representing the excitatory connection between
  # the hidden neuron k and the output neuron.
  pos_input_synapse_11 = memristor.Memristor()
  pos_input_synapse_12 = memristor.Memristor()
  pos_input_synapse_21 = memristor.Memristor()
  pos_input_synapse_22 = memristor.Memristor()
  pos_hidden_synapse_1 = memristor.Memristor()
  pos_hidden_synapse_2 = memristor.Memristor()

  # Create 6 synapses with neg_input_synapse_ij representing the inhibitory
  # connection between the input neuron i and the hidden neuron j, and
  # neg_hidden_synapse_k representing the inhibitory connection between
  # the hidden neuron k and the output neuron.
  neg_input_synapse_11 = memristor.Memristor()
  neg_input_synapse_12 = memristor.Memristor()
  neg_input_synapse_21 = memristor.Memristor()
  neg_input_synapse_22 = memristor.Memristor()
  neg_hidden_synapse_1 = memristor.Memristor()
  neg_hidden_synapse_2 = memristor.Memristor()

  # Set all the synapses to a high resistance state.
  pos_input_synapse_11.set_resistance(hrs_set_voltage)
  pos_input_synapse_12.set_resistance(hrs_set_voltage)
  pos_input_synapse_21.set_resistance(hrs_set_voltage)
  pos_input_synapse_22.set_resistance(hrs_set_voltage)
  pos_hidden_synapse_1.set_resistance(hrs_set_voltage)
  pos_hidden_synapse_2.set_resistance(hrs_set_voltage)

  neg_input_synapse_11.set_resistance(hrs_set_voltage)
  neg_input_synapse_12.set_resistance(hrs_set_voltage)
  neg_input_synapse_21.set_resistance(hrs_set_voltage)
  neg_input_synapse_22.set_resistance(hrs_set_voltage)
  neg_hidden_synapse_1.set_resistance(hrs_set_voltage)
  neg_hidden_synapse_2.set_resistance(hrs_set_voltage)

  c_11 = []
  c_12 = []
  c_21 = []
  c_22 = []
  hidden_spikes_1 = []
  hidden_spikes_2 = []
  z_1 = []
  z_2 = []
  output_spikes = []
  errors = []
  squared_errors = []
  epoch_numbers = []
  for epoch_number in range(1, epochs + 1):
    for spikes_1, spikes_2, target in zip(input_spikes_1, input_spikes_2, targets):
      epoch_numbers.append(epoch_number)

      # The normalized noisy current c_ij passes through the input_synapse_ij. The input
      # voltage is 1 V if the input neuron fired as many pulses as the encoding of
      # TRUE or more, else it is 0 V.
      pos_c_11 = utility.add_noise(utility.normalize(utility.burst(spikes_1, true)
                                                     / pos_input_synapse_11.read_resistance()))
      pos_c_12 = utility.add_noise(utility.normalize(utility.burst(spikes_1, true)
                                                     / pos_input_synapse_12.read_resistance()))
      pos_c_21 = utility.add_noise(utility.normalize(utility.burst(spikes_2, true)
                                                     / pos_input_synapse_21.read_resistance()))
      pos_c_22 = utility.add_noise(utility.normalize(utility.burst(spikes_2, true)
                                                     / pos_input_synapse_22.read_resistance()))

      # The normalized noisy current neg_c_ij passes through the neg_input_synapse_ij.
      # The current is inverted to represent the inhibitory connection. The input
      # voltage is 1 V if the input neuron fired as many pulses as the encoding
      # of TRUE or more, else it is 0 V.
      neg_c_11 = utility.invert(utility.add_noise(utility.normalize(utility.burst(spikes_1, true)
                                                                    / neg_input_synapse_11.read_resistance())))
      neg_c_12 = utility.invert(utility.add_noise(utility.normalize(utility.burst(spikes_1, true)
                                                                    / neg_input_synapse_12.read_resistance())))
      neg_c_21 = utility.invert(utility.add_noise(utility.normalize(utility.burst(spikes_2, true)
                                                                    / neg_input_synapse_21.read_resistance())))
      neg_c_22 = utility.invert(utility.add_noise(utility.normalize(utility.burst(spikes_2, true)
                                                                    / neg_input_synapse_22.read_resistance())))

      c_11.append(pos_c_11 + neg_c_11)
      c_12.append(pos_c_12 + neg_c_12)
      c_21.append(pos_c_21 + neg_c_21)
      c_22.append(pos_c_22 + neg_c_22)

      hidden_spikes_1.append(hidden_neuron_1.apply_current(c_11[-1] + c_21[-1],
                                                           current_application_time)[0].count(spike))
      hidden_spikes_2.append(hidden_neuron_2.apply_current(c_12[-1] + c_22[-1],
                                                           current_application_time)[0].count(spike))

      # The normalized noisy curent pos_z_i passes through the pos_hidden_synapse_i.
      # The input voltage is 1 V if the hidden neuron fired as many pulses as
      # the encoding of TRUE or more, else it is 0 V.
      pos_z_1 = utility.add_noise(utility.normalize(utility.burst(hidden_spikes_1[-1], true)
                                                    / pos_hidden_synapse_1.read_resistance()))
      pos_z_2 = utility.add_noise(utility.normalize(utility.burst(hidden_spikes_2[-1], true)
                                                    / pos_hidden_synapse_2.read_resistance()))

      # The normalized noisy curent neg_z_i passes through the neg_hidden_synapse_i.
      # The current is inverted to represent the inhibitory connection. The
      # input voltage is 1 V if the hidden neuron fired as many pulses as the
      # encoding of TRUE or more, else it is 0 V.
      neg_z_1 = utility.invert(utility.add_noise(utility.normalize(utility.burst(hidden_spikes_1[-1], true)
                                                                   / neg_hidden_synapse_1.read_resistance())))
      neg_z_2 = utility.invert(utility.add_noise(utility.normalize(utility.burst(hidden_spikes_2[-1], true)
                                                                   / neg_hidden_synapse_2.read_resistance())))

      z_1.append(pos_z_1 + neg_z_1)
      z_2.append(pos_z_2 + neg_z_2)

      output_spikes.append(output_neuron.apply_current(z_1[-1] + z_2[-1],
                                                       current_application_time)[0].count(spike))

      errors.append(target - output_spikes[-1])
      squared_errors.append(errors[-1] ** 2)

      # Update the synapses based on the error.
      # If the output neuron should have fired more times and it did not,
      # then strengthen the excitatory hidden synapses of the hidden neurons
      # that fired as many pulses as the encoding of TRUE or more. Otherwise,
      # do the equivalent update for the input synapses of the input neurons
      # that fired as many pulses as the encoding of TRUE or more.
      if utility.is_false_negative(errors[-1]):
        if utility.burst(hidden_spikes_1[-1], true) or utility.burst(hidden_spikes_2[-1], true):
          if utility.burst(hidden_spikes_1[-1], true):
            for _ in range(learning_pulses):
              pos_hidden_synapse_1.apply_voltage(training_voltage)
          if utility.burst(hidden_spikes_2[-1], true):
            for _ in range(learning_pulses):
              pos_hidden_synapse_2.apply_voltage(training_voltage)
        else:
          if utility.burst(spikes_1, true):
            for _ in range(learning_pulses):
              pos_input_synapse_11.apply_voltage(training_voltage)
              pos_input_synapse_12.apply_voltage(training_voltage)
          if utility.burst(spikes_2, true):
            for _ in range(learning_pulses):
              pos_input_synapse_21.apply_voltage(training_voltage)
              pos_input_synapse_22.apply_voltage(training_voltage)
      # If the output neuron should not have fired and it did, then strengthen
      # the inhibitory hidden synapses of the hidden neurons that fired as many
      # pulses as the encoding of TRUE or more.
      elif utility.is_false_positive(errors[-1]):
        if utility.burst(hidden_spikes_1[-1], true):
          for _ in range(learning_pulses):
            neg_hidden_synapse_1.apply_voltage(training_voltage)
        if utility.burst(hidden_spikes_2[-1], true):
          for _ in range(learning_pulses):
            neg_hidden_synapse_2.apply_voltage(training_voltage)

      # The neurons are resting until the next input.
      hidden_neuron_1.rest()
      hidden_neuron_2.rest()
      output_neuron.rest()

  # Store the data into a data frame.
  data = pd.DataFrame()
  data["input_spikes_1"] = input_spikes_1 * epochs
  data["input_spikes_2"] = input_spikes_2 * epochs
  data["c_11"] = c_11
  data["c_12"] = c_12
  data["c_21"] = c_21
  data["c_22"] = c_22
  data["hidden_spikes_1"] = hidden_spikes_1
  data["hidden_spikes_2"] = hidden_spikes_2
  data["z_1"] = z_1
  data["z_2"] = z_2
  data["output_spikes"] = output_spikes
  data["target"] = targets * epochs
  data["error"] = errors
  data["squared_error"] = squared_errors
  data["epoch"] = epoch_numbers
  data["learning_pulses"] = [learning_pulses] * epochs * 4
  data["training_voltage"] = [training_voltage] * epochs * 4
  data["current_application_time"] = [current_application_time] * epochs * 4

  if output_to_csv:
    utility.save_data(data, "./output", "solve-xor-complex-noisy-snn")

  # Plot the MSE as a function of epoch.
  utility.plot_mse(data)
