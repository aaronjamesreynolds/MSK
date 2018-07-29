#!/usr/bin/env python

# ToDo: (3) form linear system (4) solve linear system

import numpy
import matplotlib.pyplot as plt
import input.read as read
from numba import jitclass, int64, float64
import moc_transport.step_characteristic as moc


class QuasiDiffusionPrecursorConcentration:

    def __init__(self, input_file_name):

        # Quadrature data
        self.ab = numpy.array([-0.9739065285171717, -0.8650633666889845, -0.6794095682990244, -0.4333953941292472,
                               -0.1488743389816312, 0.1488743389816312, 0.4333953941292472, 0.6794095682990244,
                               0.8650633666889845, 0.9739065285171717], dtype=numpy.float64)
        self.weights = numpy.array([0.0666713443086881, 0.1494513491505806, 0.2190863625159820, 0.2692667193099963,
                                    0.2955242247147529, 0.2955242247147529, 0.2692667193099963, 0.2190863625159820,
                                    0.1494513491505806, 0.0666713443086881], dtype=numpy.float64)

        # Import from YAML input file
        input_data = read.Input(input_file_name)

        # Nuclear data
        self.sig_t = input_data.data.sig_t  # total cross section
        self.sig_s = input_data.data.sig_s  # scatter cross section
        self.sig_f = input_data.data.sig_f  # fission cross section
        self.nu = input_data.data.nu  # number of neutrons produced per fission
        self.material = input_data.data.material  # material map

        # Problem geometry parameters
        self.groups = 1  # energy groups in problem
        self.core_mesh_length = input_data.data.cells  # number of intervals
        self.dx = 0.5  # discretization in length
        self.dmu = 2 / len(self.ab)  # discretization in angle
        self.dt = 0.5  # discretization in time

        # Alpha approximation parameters
        self.alpha = 0.0 * numpy.ones(self.core_mesh_length, dtype=numpy.float64) # describes change in scalar flux between time steps
        self.v = 1000 # neutron velocity
        self.beta = 0.007 # delayed neutron fraction
        self.lambda_eff = 0.08 # delayed neutron precursor decay constant
        self.delayed_neutron_precursor_concentration = 0.0*numpy.ones((self.core_mesh_length, 2), dtype=numpy.float64)
        self.dnpc_velocity = 0.0*numpy.ones(self.core_mesh_length, dtype=numpy.float64)

        # Set initial values
        self.flux = numpy.zeros((self.core_mesh_length, 2), dtype=numpy.float64)  # initialize flux. (position, 0:new, 1:old)
        self.current = numpy.zeros((self.core_mesh_length + 1, 2), dtype=numpy.float64)
        self.eddington_factors = numpy.zeros(self.core_mesh_length, dtype=numpy.float64)
        self.coefficient_matrix = numpy.empty([2, 2])
        self.rhs = numpy.empty(2)

        # Solver metrics
        self.exit1 = 0  # initialize exit condition
        self.exit2 = 0  # initialize exit condition
        self.flux_iterations = 0  # iteration counter
        self.source_iterations = 0  # iteration counter


    """ Update neutron flux, neutron current, Eddington factors, and delayed neutron precursor concentration variables. """
    def update_variables(self, _flux, _current, _eddington_factors, _delayed_neutron_precursor_concentration):

        self.flux[:, 1] = _flux
        self.current[:, 1] = _current
        self.eddington_factors = _eddington_factors
        self.delayed_neutron_precursor_concentration[:, 1] = _delayed_neutron_precursor_concentration

    def explicit_time_solve(self):

        for position in xrange(self.core_mesh_length):

            sig_a = self.sig_t[self.material[position]] - self.sig_s[self.material[position]]
            self.coefficient_matrix[0, 0] = 1 + self.v * self.dt * (sig_a - (1 - self.beta) * self.nu[self.material[position]] *
                                                                    self.sig_f[self.material[position]])
            self.coefficient_matrix[0, 1] = -self.v * self.dt * self.lambda_eff
            self.coefficient_matrix[1, 0] = -self.dt * self.beta * self.nu[self.material[position]] * self.sig_f[self.material[position]]
            self.coefficient_matrix[1, 1] = 1 + self.dt * self.lambda_eff

            self.rhs[0] = self.flux[position, 1] + self.v * self.dt * \
                          (self.current[position, 1] - self.current[position + 1 , 1]) / self.dx
            self.rhs[1] = self.dt * (self.dnpc_velocity[position - 1] *
                                     self.delayed_neutron_precursor_concentration[position - 1, 1] - self.dnpc_velocity[position] *
                                     self.delayed_neutron_precursor_concentration[position, 1]) / self.dx + \
                                     self.delayed_neutron_precursor_concentration[position, 1]
            solutions = numpy.linalg.solve(self.coefficient_matrix, self.rhs)
            self.flux[position, 0] = solutions[0]
            self.delayed_neutron_precursor_concentration[position, 0] = solutions[1]

    def implicit_time_solve(self):

        for position in xrange(self.core_mesh_length):

            sig_a = self.sig_t[self.material[position]] - self.sig_s[self.material[position]]
            self.coefficient_matrix[0, 0] = 1 + self.v * self.dt * (sig_a - (1 - self.beta) * self.nu[self.material[position]] *
                                                                    self.sig_f[self.material[position]])
            self.coefficient_matrix[0, 1] = -self.v * self.dt * self.lambda_eff
            self.coefficient_matrix[1, 0] = -self.dt * self.beta * self.nu[self.material[position]] * self.sig_f[self.material[position]]
            self.coefficient_matrix[1, 1] = 1 + self.dt * self.lambda_eff

            self.rhs[0] = self.flux[position, 1] + self.v * self.dt * \
                          (self.current[position, 1] - self.current[position + 1 , 1]) / self.dx
            self.rhs[1] = self.dt * (self.dnpc_velocity[position - 1] *
                                     self.delayed_neutron_precursor_concentration[position - 1, 1] - self.dnpc_velocity[position] *
                                     self.delayed_neutron_precursor_concentration[position, 1]) / self.dx + \
                                     self.delayed_neutron_precursor_concentration[position, 1]
            solutions = numpy.linalg.solve(self.coefficient_matrix, self.rhs)
            self.flux[position, 0] = solutions[0]
            self.delayed_neutron_precursor_concentration[position, 0] = solutions[1]




if __name__ == "__main__":

    test_gray = QuasiDiffusionPrecursorConcentration("test_input.yaml") # test for initialization
    test_moc = moc.StepCharacteristic("test_input.yaml")
    test_moc.solve(True)

    x = numpy.arange(0, test_gray.core_mesh_length)

    for iteration in xrange(20):
        test_gray.update_variables(test_moc.flux[:, 0], test_moc.current, test_moc.eddington_factors,
                                   test_moc.delayed_neutron_precursor_concentration) # test to update variables
        test_gray.explicit_time_solve() # test if linear system can be solved
        test_moc.update_variables(test_gray.flux[:, 0], test_gray.delayed_neutron_precursor_concentration[:, 0])
        test_moc.exit1 = 0
        test_moc.solve(True)


    # plot scalar flux
    x = numpy.arange(0, test_gray.core_mesh_length)
    plt.plot(x, test_gray.flux[:, 0])
    plt.grid(True)
    plt.xlabel('Position [cm]')
    plt.ylabel('Flux [s^-1 cm^-2]')
    plt.title('Neutron Flux')
    plt.show()

    # plot delayed neutron precursor concentration
    x = numpy.arange(0, test_gray.core_mesh_length)
    plt.plot(x, test_gray.delayed_neutron_precursor_concentration[:, 0])
    plt.grid(True)
    plt.xlabel('Position [cm]')
    plt.ylabel('Flux [s^-1 cm^-2]')
    plt.title('Precursor Concentration')
    plt.show()
