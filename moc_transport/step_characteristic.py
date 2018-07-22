#!/usr/bin/env python

# ToDo: (1) normalize to allow for time dynamics (2) make a test

import numpy
import matplotlib.pyplot as plt
import input.read as read
from numba import jitclass, int64, float64

# Note: an initializing class may be necessary to accomodate the use of numba. Might be clumsy, but worth it.

spec = [
    ('sig_t', float64[:]),
    ('sig_s_in', float64[:]),
    ('sig_s_out', float64[:]),
    ('sig_s', float64[:]),
    ('sig_f', float64[:]),
    ('nu', float64[:]),
    ('chi', float64[:]),
    ('ab', float64[:]),
    ('weights', float64[:]),
    ('groups', int64),
    ('core_mesh_length', int64),
    ('dx', float64),
    ('dmu', float64),
    ('flux_new', float64[:]),
    ('flux_old', float64[:]),
    ('edge_flux', float64[:]),
    ('phi_L_old', float64[:]),
    ('phi_R_old', float64[:]),
    ('angular_flux_edge', float64[:, :]),
    ('angular_flux_center', float64[:, :]),
    ('eddington_factors', float64[:]),
    ('k_old', float64),
    ('k_new', float64),
    ('spatial_fission_old', float64[:]),
    ('spatial_fission_new', float64[:]),
    ('material', int64[:]),
    ('exit1', int64),
    ('exit2', int64),
    ('flux_iterations', int64),
    ('source_iterations', int64),
    ('Q', float64[:]),
    ('fission_source_dx', float64),
    ('spatial_sig_s_out', float64[:])

]


#@jitclass(spec)
class StepCharacteristic(object):

    # Initialize and assign variables.
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
        self.alpha = 0.01 * numpy.ones(self.core_mesh_length, dtype=numpy.float64) # describes change in scalar flux between time steps
        self.v = 1000 # neutron velocity
        self.beta = 0.007 # delayed neutron fraction
        self.lambda_eff = 0.08 # delayed neutron precursor decay constant
        self.delayed_neutron_precursor_concentration = 0.05*numpy.ones(self.core_mesh_length, dtype=numpy.float64)

        # Set initial values
        self.flux = numpy.zeros((self.core_mesh_length, 2), dtype=numpy.float64)  # initialize flux. (position, 0:new, 1:old)
        self.flux_t = numpy.zeros((self.core_mesh_length, 100), dtype=numpy.float64) # assume ten time steps to start
        self.edge_flux = numpy.ones(self.core_mesh_length + 1, dtype=numpy.float64)
        self.angular_flux_edge = numpy.zeros((self.core_mesh_length + 1, len(self.ab)),
                                             dtype=numpy.float64)  # initialize edge flux
        self.angular_flux_center = numpy.zeros((self.core_mesh_length, len(self.ab)),
                                               dtype=numpy.float64)  # initialize edge flux
        self.current = numpy.zeros(self.core_mesh_length + 1, dtype=numpy.float64)
        self.eddington_factors = numpy.zeros(self.core_mesh_length, dtype=numpy.float64)

        # Solver metrics
        self.exit1 = 0  # initialize exit condition
        self.exit2 = 0  # initialize exit condition
        self.flux_iterations = 0  # iteration counter
        self.source_iterations = 0  # iteration counter

        # Implement initial angular fluxes conditions of 1 at cell edges.
        for k in xrange(self.groups):
            for j in [0, self.core_mesh_length]:
                for i in xrange(10):
                    if i + 1 <= len(self.ab) / 2 and j == self.core_mesh_length:
                        self.angular_flux_edge[j, i] = 1
                    elif i + 1 > len(self.ab) / 2 and j == 0:
                        self.angular_flux_edge[j, i] = 1

    """ With given angular fluxes at the center, calculate the scalar flux using a quadrature set. """
    def calculate_scalar_flux(self):

        for i in xrange(self.core_mesh_length):
            for x in xrange(len(self.ab)):
                self.flux[i, 0] = self.flux[i, 0] + self.weights[x] * self.angular_flux_center[i, x]

    """ With given angular fluxes the edge, calculate the scalar flux using a quadrature set. """
    def calculate_scalar_edge_flux(self):

        for i in xrange(self.core_mesh_length+1):
            for x in xrange(len(self.ab)):
                self.edge_flux[i] = self.edge_flux[i] + self.weights[x] * self.angular_flux_edge[i, x]

    """ With given angular fluxes at the edge, calculate the current using a quadrature set. """

    def calculate_current(self):

        self.current = numpy.zeros(self.core_mesh_length + 1, dtype=numpy.float64)

        for i in xrange(self.core_mesh_length + 1):
            for x in xrange(len(self.ab)):
                self.current[i] = self.current[i] + self.ab[x] * self.weights[x] * self.angular_flux_edge[i, x]

    """ Calculates eddington factors (done a single time after problem is converged) """
    def calculate_eddington_factors(self):

        self.eddington_factors = numpy.zeros(self.core_mesh_length, dtype=numpy.float64)

        for i in xrange(self.core_mesh_length):
            for x in xrange(len(self.ab)):
                self.eddington_factors[i] = self.eddington_factors[i] + self.ab[x] ** 2 *\
                                               self.angular_flux_center[i, x] * self.weights[x]\
                                               / self.flux[i, 0]

    """ Reflects given angular fluxes at the boundary across the mu = 0 axis. """
    def iterate_boundary_condition(self):
        # Redefine angular flux at boundaries.
        for j in xrange(0, self.core_mesh_length+1):
            for i in xrange(10):
                if i + 1 <= len(self.ab) / 2 and j == self.core_mesh_length:
                    self.angular_flux_edge[j, i] = self.angular_flux_edge[j, len(self.ab) - i - 1]
                elif i + 1 > len(self.ab) / 2 and j == 0:
                    self.angular_flux_edge[j, i] = self.angular_flux_edge[j, len(self.ab) - i - 1]

    """ Iterate on alpha based on old and new scalar flux """
    def iterate_alpha(self, last_time_step):

        for i in xrange(self.core_mesh_length):
            self.alpha[i] = numpy.log(self.flux[i, 0] / self.flux_t[i, last_time_step]) / self.dt

    # Propagate angular flux boundary conditions across the problem.
    def flux_iteration(self):
        for z in xrange(5, 10):
            for i in xrange(self.core_mesh_length):
                xi = (self.sig_t[self.material[i]] + self.alpha[i] / self.v) / self.ab[z]  # integrating factor

                if xi < 10**-4:
                    xi = 10**-4 / self.ab[z]

                q = (self.sig_s[self.material[i]] * self.flux[i, 1]
                     + (1 - self.beta) * self.nu[self.material[i]] * self.sig_f[self.material[i]] * self.flux[i, 1]
                     + self.lambda_eff * self.delayed_neutron_precursor_concentration[i])  # source term

                self.angular_flux_edge[i + 1, z] = self.angular_flux_edge[i, z] * numpy.exp(-xi * self.dx) \
                    + (q / (2 * self.ab[z] * xi)) * (1 - numpy.exp(-xi * self.dx))

                self.angular_flux_center[i, z] = (1 / (self.dx * xi)) * (q * self.dx / (2 * self.ab[z])
                                                                         + self.angular_flux_edge[i, z]
                                                                         - self.angular_flux_edge[i + 1, z])
        for z in xrange(0, 5):
            for i in range(self.core_mesh_length, 0, -1):
                xi = (self.sig_t[self.material[i-1]] + self.alpha[i-i] / self.v) / numpy.abs(self.ab[z])  # integrating factor

                if xi < 10**-4:
                    xi = 10**-4 / numpy.abs(self.ab[z])

                q = (self.sig_s[self.material[i-1]] * self.flux[i-1, 1]
                     + (1 - self.beta) * self.nu[self.material[i-1]] * self.sig_f[self.material[i-1]] * self.flux[i - 1, 1]
                     + self.lambda_eff * self.delayed_neutron_precursor_concentration[i-1])  # source term

                self.angular_flux_edge[i - 1, z] = self.angular_flux_edge[i, z] * numpy.exp(-xi * self.dx) \
                    + (q / (2 * numpy.abs(self.ab[z]) * xi)) * (1 - numpy.exp(-xi * self.dx))

                self.angular_flux_center[i - 1, z] = (1 / (self.dx * xi)) * (q * self.dx / (2 * numpy.abs(self.ab[z]))
                                                                             + self.angular_flux_edge[i, z]
                                                                              - self.angular_flux_edge[i - 1, z])
        self.calculate_scalar_flux()

    def solve(self, single_step = False):
        print "Performing method of characterisitics solve..."

        while self.exit1 == 0:  # flux convergence

            if single_step:
                self.exit1 = 1

            self.flux_iterations += 1

            print "----------------DEBUG----------------------"
            print "Iteration: {}".format(self.flux_iterations)
            print "Alpha: {}".format(self.alpha)
            print "Flux: {}".format(self.flux[:, 1])
            print "-------------------------------------------"

            self.flux_iteration()  # do a flux
            self.calculate_eddington_factors()
            self.flux[:, 0] = self.flux[:, 0] / numpy.sum(self.flux[:, 0])
            # Check for convergence
            if abs(numpy.max(((self.flux[:, 0] - self.flux[:, 1]) / self.flux[:, 0]))) < 1E-6:
                self.exit1 = 1  # exit flux iteration
                self.flux[:, 1] = self.flux[:, 0] # reassign flux
                self.flux_t[:, 0] = self.flux[:, 0]
                print self.flux
                self.calculate_current()
                self.results()
                print ''

            else:
                self.flux[:, 1] = self.flux[:, 0]  # assign flux
                self.flux[:, 0] = numpy.zeros(self.core_mesh_length, dtype=numpy.float64)  # reset new_flux
                self.iterate_boundary_condition()



    """Solve forward in time. [NOT USED]"""
    def solve_forward(self):
        print "First time step solution found, solving for subsequent steps..."

        for time_step in xrange(1, 100):

            self.flux_iterations = 0

            self.flux = numpy.ones((self.core_mesh_length, 2), dtype=numpy.float64)
            self.iterate_alpha(time_step-1)

            self.exit2 = 0

            while self.exit2 == 0:  # flux convergence

                self.flux_iterations += 1

                # print "----------------DEBUG----------------------"
                # print "Iteration: {}".format(self.flux_iterations)
                # print "Alpha: {}".format(self.alpha)
                # print "Flux_old: {}".format(self.flux_old)
                # print "-------------------------------------------"

                self.flux_iteration()  # do a flux
                self.calculate_eddington_factors()
                self.flux[:, 0] = self.flux[:, 0] / numpy.sum(self.flux[:, 0])
                # Check for convergence
                if abs(numpy.max(((self.flux[:, 0] - self.flux[:, 1]) / self.flux[:, 0]))) < 1E-6:
                    self.exit2 = 1  # exit flux iteration
                    self.flux[:, 1] = self.flux[:, 0] # reassign flux
                    self.flux_t[:, time_step] = self.flux[:, 0]
                    print self.flux
                    print "Solution at step {} found".format(time_step)
                   # print "Alpha: {}".format(self.alpha)
                    self.results()

                else:
                    self.iterate_alpha(time_step-1)
                    #t "Alpha: {}".format(self.alpha)
                    self.flux[:, 1] = self.flux[:, 0]  # assign flux
                    self.flux[:, 0] = numpy.zeros(self.core_mesh_length, dtype=numpy.float64)  # reset new_flux
                    self.iterate_boundary_condition()

    """ Update neutron flux, neutron current, Eddington factors, and delayed neutron precursor concentration variables. """
    def update_variables(self, _flux, _delayed_neutron_precursor_concentration):

        self.flux[:, 1] = _flux
        self.delayed_neutron_precursor_concentration = _delayed_neutron_precursor_concentration

    """Plot scalar and angular fluxes."""
    def results(self):

        print 'Flux iterations: {0}'.format(self.flux_iterations)
        # plot scalar flux
        x = numpy.arange(0, self.core_mesh_length)
        plt.plot(x, self.flux[:, 0])
        plt.grid(True)
        plt.xlabel('Position [cm]')
        plt.ylabel('Flux [s^-1 cm^-2]')
        plt.title('Neutron Flux')
        plt.show()

        x = numpy.arange(0, self.core_mesh_length + 1)
        plt.plot(x, self.current[:])
        plt.grid(True)
        plt.xlabel('Position [cm]')
        plt.ylabel('Current [s^-1 cm^-2]')
        plt.title('Neutron Current')
        plt.show()

        # # plot angular flux
        # for i in xrange(10):
        #     plt.plot(x, self.angular_flux_center[:, i])
        # plt.grid(True)
        # plt.xlabel('Position [cm]')
        # plt.ylabel('Flux [s^-1 cm^-2]')
        # plt.title('Angular Neutron Flux')
        # plt.show()
        # # plot eddington factors
        # plt.plot(x, self.eddington_factors[:])
        # plt.grid(True)
        # plt.xlabel('Position [cm]')
        # plt.ylabel('Eddington Factor')
        # plt.title('Eddington Factors')
        # plt.show()

if __name__ == "__main__":

    test = StepCharacteristic("test_input.yaml")
    test.solve()









