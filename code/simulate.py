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

# Simulate Module
# Module to run simulations with memristors.

import pulses
import utility
import xor
import xor_snn

input_arguments = utility.parse_input_arguments()

if input_arguments.simulation_function == "pulsing_experiment":
  pulses.simulate_pulsing_experiment()
elif input_arguments.simulation_function == "model_xor":
  xor.model_xor()
elif input_arguments.simulation_function == "solve_xor":
  xor.solve_xor(input_arguments.learning_pulses, input_arguments.epochs,
                input_arguments.learn_hidden_synapses, input_arguments.output_to_csv)
elif input_arguments.simulation_function == "solve_xor_complex":
  xor.solve_xor_complex(input_arguments.learning_pulses, input_arguments.epochs,
                        input_arguments.output_to_csv)
elif input_arguments.simulation_function == "model_xor_snn":
  xor_snn.model_xor_snn()
elif input_arguments.simulation_function == "solve_xor_snn":
  xor_snn.solve_xor_snn(input_arguments.learning_pulses, input_arguments.epochs,
                        input_arguments.learn_hidden_synapses, input_arguments.output_to_csv)
elif input_arguments.simulation_function == "solve_xor_complex_snn":
  xor_snn.solve_xor_complex_snn(input_arguments.learning_pulses, input_arguments.epochs,
                                input_arguments.output_to_csv)
