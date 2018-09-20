#!/usr/bin/env python

# ToDo: (3) form linear system (4) solve linear system | DONE
# ToDo: (5) fix time stepping

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
        self.dx = input_data.data.dx
        self.dt = input_data.data.dt


        # Problem geometry parameters
        self.groups = 1  # energy groups in problem
        self.core_mesh_length = input_data.data.cells  # number of intervals
        #self.dx = 1.0  # discretization in length
        self.dmu = 2 / len(self.ab)  # discretization in angle
        #self.dt = 0.0001  # discretization in time

        # Alpha approximation parameters
        self.alpha = 0.0 * numpy.ones(self.core_mesh_length, dtype=numpy.float64) # describes change in scalar flux between time steps
        self.v = 1000 # neutron velocity
        self.beta = 0.007 # delayed neutron fraction
        self.lambda_eff = 0.08 # delayed neutron precursor decay constant
        self.delayed_neutron_precursor_concentration = 1*numpy.ones((self.core_mesh_length, 2), dtype=numpy.float64)
        self.dnpc_velocity = 0.0*numpy.ones(self.core_mesh_length, dtype=numpy.float64)
        self.dnpc_velocity = xrange(9000, 0, -100)

        # Set initial values
        self.flux = numpy.zeros((self.core_mesh_length, 2), dtype=numpy.float64)  # initialize flux. (position, 0:new, 1:old)
        self.current = numpy.zeros((self.core_mesh_length + 1, 2), dtype=numpy.float64)
        self.eddington_factors = numpy.zeros(self.core_mesh_length, dtype=numpy.float64)
        self.coefficient_matrix = numpy.empty([2, 2])
        self.coefficient_matrix_implicit = numpy.empty([3, 3])
        self.rhs = numpy.empty(2)
        self.rhs_implicit = numpy.empty(3)


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

    def update_eddington(self, _eddington_factors):

        self.eddington_factors = _eddington_factors

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

        self.set_initial_conditions()

        for position in xrange(1, self.core_mesh_length):

            sig_a = self.sig_t[self.material[position]] - self.sig_s[self.material[position]]
            ave_sig_t = (self.sig_t[self.material[position - 1]] + self.sig_t[self.material[position]])/2
            self.coefficient_matrix_implicit[0, 0] = 1 + self.v * self.dt * (sig_a - (1 - self.beta) * self.nu[self.material[position]] *
                                                                    self.sig_f[self.material[position]])
            self.coefficient_matrix_implicit[0, 1] = self.v * self.dt / self.dx
            self.coefficient_matrix_implicit[0, 2] = -self.v * self.dt * self.lambda_eff
            self.coefficient_matrix_implicit[1, 0] = self.v * self.dt * self.eddington_factors[position] / self.dx
            self.coefficient_matrix_implicit[1, 1] = 1 + self.v * self.dt * ave_sig_t
            self.coefficient_matrix_implicit[1, 2] = 0
            self.coefficient_matrix_implicit[2, 0] = -self.dt * self.beta * self.nu[self.material[position]] * self.sig_f[self.material[position]]
            self.coefficient_matrix_implicit[2, 1] = 0
            self.coefficient_matrix_implicit[2, 2] = 1 + self.dt * self.lambda_eff + self.dt * self.dnpc_velocity[position] / self.dx

            self.rhs_implicit[0] = self.flux[position, 1] + self.v * self.dt \
                                   * (self.current[position - 1, 0]) / self.dx
            self.rhs_implicit[1] = self.current[position, 1] + self.v * self.dt * (self.eddington_factors[position - 1]
                                                                                   * self.flux[position - 1, 0]) / self.dx
            self.rhs_implicit[2] = self.dt * (self.dnpc_velocity[position - 1] *
                                     self.delayed_neutron_precursor_concentration[position - 1, 0]) / self.dx + \
                                     self.delayed_neutron_precursor_concentration[position, 1]

            solutions = numpy.linalg.solve(self.coefficient_matrix_implicit, self.rhs_implicit)
            self.flux[position, 0] = solutions[0]
            self.current[position, 0] = solutions[1]
            self.delayed_neutron_precursor_concentration[position, 0] = solutions[2]

        self.delayed_neutron_precursor_concentration[:, 1] = self.delayed_neutron_precursor_concentration[:, 0]
        #make first and last cell equal
        self.delayed_neutron_precursor_concentration[0, 0] = self.delayed_neutron_precursor_concentration[-1, 0]


    def set_initial_conditions(self):

        self.flux[0, 0] = self.flux[0, 1]
        self.current[0, 0] = 0
        #self.delayed_neutron_precursor_concentration[0, 0] = 0


if __name__ == "__main__":

    steps = 10
    flux_t = numpy.zeros([90, steps+1])
    precursor_t = numpy.zeros([90, steps+1])
    test_gray = QuasiDiffusionPrecursorConcentration("test_input.yaml") # test for initialization
    test_moc = moc.StepCharacteristic("test_input.yaml")
    test_moc.solve(False, True)
    flux_t[:, 0] = test_moc.flux[:, 1]
    precursor_t[:, 0] = test_gray.delayed_neutron_precursor_concentration[:, 0]
    test_moc.converged = False

    test_gray.update_variables(test_moc.flux[:, 1], test_moc.current, test_moc.eddington_factors,
                               test_moc.delayed_neutron_precursor_concentration)

    for iteration in xrange(steps):
        test_gray.update_variables(test_moc.flux[:, 1], test_moc.current, test_moc.eddington_factors,
                                   test_moc.delayed_neutron_precursor_concentration)
        test_gray.implicit_time_solve() # test if linear system can be solved
        test_moc.update_variables(test_moc.flux[:, 1], test_gray.delayed_neutron_precursor_concentration[:, 0])
        test_moc.exit1 = 0
        test_moc.solve(False, False)
        test_moc.converged = False
        flux_t[:, iteration+1] = test_moc.flux[:, 1]
        precursor_t[:, iteration+1] = test_gray.delayed_neutron_precursor_concentration[:, 0]


    # test_moc.results()
    x = numpy.arange(0, test_gray.core_mesh_length)
    for iteration in xrange(steps+1):
        # plt.plot(x, test_gray.flux[:, 0])
        plt.plot(x, flux_t[:, iteration], label = iteration)
    plt.grid(True)
    plt.xlabel('Position [cm]')
    plt.ylabel('Flux [s^-1 cm^-2]')
    plt.title('Grey Group: Neutron Flux')
    plt.legend()
    plt.show()

    # plot scalar flux
    # x = numpy.arange(0, test_gray.core_mesh_length)
    # plt.plot(x, test_gray.flux[:, 0])
    # plt.plot(x, test_moc.flux[:, 1])
    # plt.grid(True)
    # plt.xlabel('Position [cm]')
    # plt.ylabel('Flux [s^-1 cm^-2]')
    # plt.title('Grey Group: Neutron Flux')
    # plt.show()

    # plot delayed neutron precursor concentration
    x = numpy.arange(0, test_gray.core_mesh_length)
    for iteration in xrange(steps+1):
        # plt.plot(x, test_gray.flux[:, 0])
        plt.plot(x, precursor_t[:, iteration], label = iteration)
    plt.grid(True)
    plt.legend()
    plt.xlabel('Position [cm]')
    plt.ylabel('Concentration [cm^-3]')
    plt.title('Grey Group: Precursor Concentration')
    plt.show()
