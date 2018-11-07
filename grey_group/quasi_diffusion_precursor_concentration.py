#!/usr/bin/env python

# ToDo: comment
# ToDo: make convergence measurements relative

import numpy
import matplotlib.pyplot as plt
import input.read as read
from numba import jitclass, int64, float64
import moc_transport.step_characteristic as moc


class QuasiDiffusionPrecursorConcentration:

    """Initialize the object from an input file. """
    def __init__(self, input_file_name):

        self.input_file_name = input_file_name

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
        self.dmu = 2 / len(self.ab)  # discretization in angle

        # Alpha approximation parameters
        self.alpha = input_data.data.alpha * numpy.ones(self.core_mesh_length, dtype=numpy.float64) # describes change in scalar flux between time steps
        self.v = input_data.data.v # neutron velocity
        self.beta = input_data.data.beta # delayed neutron fraction
        self.lambda_eff = input_data.data.lambda_eff # delayed neutron precursor decay constant
        self.delayed_neutron_precursor_concentration = input_data.data.dnp_concentration*numpy.ones((self.core_mesh_length, 2), dtype=numpy.float64)
        self.delayed_neutron_precursor_concentration[:, 0] = numpy.ones(self.core_mesh_length)
        self.dnpc_velocity = 10 *numpy.ones(self.core_mesh_length, dtype=numpy.float64)
        self.dnpc_velocity = numpy.linspace(input_data.data.dnp_velocity_lhs, input_data.data.dnp_velocity_rhs, self.core_mesh_length)


        # Set initial values
        self.flux = numpy.ones((self.core_mesh_length, 2), dtype=numpy.float64)  # initialize flux. (position, 0:new, 1:old)
        self.current = numpy.zeros((self.core_mesh_length + 1, 2), dtype=numpy.float64)
        #self.current[:,0] = numpy.ones(self.core_mesh_length + 1)
        self.eddington_factors = 1*numpy.array(numpy.ones(self.core_mesh_length, dtype=numpy.float64))
        self.coefficient_matrix = numpy.empty([2, 2])
        self.coefficient_matrix_implicit = numpy.empty([3, 3])
        self.coefficient_matrix_stationary_implicit = numpy.empty([2, 2])
        self.rhs = numpy.empty(2)
        self.rhs_implicit = numpy.empty(3)
        self.rhs_stationary_implicit = numpy.empty(2)
        self.stationary_linear_system = numpy.zeros([2*self.core_mesh_length + 1, 2*self.core_mesh_length + 1])
        self.stationary_linear_system_solution = numpy.zeros([2*self.core_mesh_length + 1, 2])
        self.linear_system = numpy.zeros([3 * self.core_mesh_length + 1, 3 * self.core_mesh_length + 1])
        self.linear_system_solution = numpy.zeros([3 * self.core_mesh_length + 1, 2])

        # Method of manufactured solutions parameters
        self.psi_0_mms = 1  # constant flux coefficient
        self.C_0_mms = 1  # constant precursor coefficient
        self.q_z_mms = numpy.zeros((self.core_mesh_length, 1), dtype=numpy.float64)
        self.q_q_mms = numpy.zeros((self.core_mesh_length, 1), dtype=numpy.float64)
        self.q_p_mms = numpy.zeros((self.core_mesh_length, 1), dtype=numpy.float64)
        self.a = 100  # where a*pi is the velocity on the LHS

    """ Update neutron flux, neutron current, Eddington factors, and delayed neutron precursor concentration variables. """
    def update_variables(self, _flux, _current, _eddington_factors, _delayed_neutron_precursor_concentration):

        self.flux[:, 1] = _flux
        self.current[:, 1] = _current
        self.eddington_factors = _eddington_factors
        self.delayed_neutron_precursor_concentration[:, 1] = _delayed_neutron_precursor_concentration

    """ Updates the Eddington factors of the grey group. """
    def update_eddington(self, _eddington_factors):

        self.eddington_factors = _eddington_factors
        # Diffusion
        #self.eddington_factors = 1/3*numpy.ones(self.core_mesh_length, dtype=numpy.float64)

    """Deprecated method."""
    def explicit_time_solve(self):

        for position in xrange(self.core_mesh_length):

            ave_sig_t = (self.sig_t[self.material[position - 1]] + self.sig_t[self.material[position]])/2

            self.current[position + 1, 0] = (self.current[position + 1, 1] + self.v * self.dt \
                                            * (self.eddington_factors[position - 1]*self.flux[position - 1, 1]
                                               - self.eddington_factors[position]*self.flux[position, 1])/self.dx) \
                                             / (1 + self.v * self.dt * ave_sig_t)

            self.current[position + 1, 0] = 10 * (-1)**position

            sig_a = self.sig_t[self.material[position]] - self.sig_s[self.material[position]]
            self.coefficient_matrix[0, 0] = 1 + self.v * self.dt * (sig_a - (1 - self.beta) * self.nu[self.material[position]] *
                                                                    self.sig_f[self.material[position]])
            self.coefficient_matrix[0, 1] = -self.v * self.dt * self.lambda_eff
            self.coefficient_matrix[1, 0] = -self.dt * self.beta * self.nu[self.material[position]] * self.sig_f[self.material[position]]
            self.coefficient_matrix[1, 1] = 1 + self.dt * self.lambda_eff

            self.rhs[0] = self.flux[position, 1] + self.v * self.dt * \
                          (self.current[position, 1] - self.current[position + 1, 1]) / self.dx
            self.rhs[1] = self.dt * (self.dnpc_velocity[position - 1] *
                                     self.delayed_neutron_precursor_concentration[position - 1, 1] - self.dnpc_velocity[position] *
                                     self.delayed_neutron_precursor_concentration[position, 1]) / self.dx + \
                                     self.delayed_neutron_precursor_concentration[position, 1]
            solutions = numpy.linalg.solve(self.coefficient_matrix, self.rhs)
            self.flux[position, 0] = solutions[0]
            self.delayed_neutron_precursor_concentration[position, 0] = solutions[1]

    """ Solve the linear system of the zero moment neutron transport equation, quasi diffusion equation, and precursor
        concentration equation (with no precursor drift). """
    def solve_stationary_linear_system(self):


        # BUILD THE LINEAR SYSTEM AND THE SOLUTION WILL COME
        # LHS
        n = self.core_mesh_length

        for position in range(n):

            sig_a = self.sig_t[self.material[position]] - self.sig_s[self.material[position]]
            ave_sig_t = (self.sig_t[self.material[position - 1]] + self.sig_t[self.material[position]]) / 2
            zeta = 1 + self.v * self.dt * (sig_a - (1 - self.beta) * self.nu[self.material[position]] *
                                           self.sig_f[self.material[position]] \
                                           + self.dt * self.beta * self.nu[self.material[position]] \
                                           * self.sig_f[self.material[position]] * self.lambda_eff \
                                           / (1 + self.dt * self.lambda_eff))
            self.stationary_linear_system[position, position] = zeta

        courant = self.dt*self.v/self.dx

        column = 0
        for row in range(n + 1, 2*n):
            self.stationary_linear_system[row, column] = -courant*self.eddington_factors[column]
            self.stationary_linear_system[row, column + 1] = courant*self.eddington_factors[column]
            column += 1

        row = 0
        for column in range(n, 2*n):
            self.stationary_linear_system[row, column] = -courant
            self.stationary_linear_system[row, column + 1] = courant
            row += 1

        position = 0
        for index in xrange(n+1, 2*n):
            ave_sig_t = (self.sig_t[self.material[position - 1]] + self.sig_t[self.material[position]]) / 2
            self.stationary_linear_system[index, index] = 1 + self.dt*self.v*ave_sig_t
            position += 1

        self.stationary_linear_system[n , n] = 1
        self.stationary_linear_system[2*n, 2*n] = 1

        # RHS

        self.stationary_linear_system_solution[0:n, 1] = numpy.array(self.flux[:, 1] \
                                                + self.dt*self.v*self.lambda_eff/ (1+ self.dt*self.v*self.lambda_eff)\
                                              * self.delayed_neutron_precursor_concentration[:,1])
        self.stationary_linear_system_solution[n:, 1] = numpy.array(self.current[:, 1])

        #solve!
        self.stationary_linear_system_solution[:, 0] = numpy.linalg.solve(self.stationary_linear_system, self.stationary_linear_system_solution[:, 1])
        self.flux[:, 0] =  self.stationary_linear_system_solution[0:n, 0]
        self.current[:, 0] = self.stationary_linear_system_solution[n:, 0]

        #obtain delayed neutron precursor concentration
        for i in range(n):
            self.delayed_neutron_precursor_concentration[i, 0] = (self.delayed_neutron_precursor_concentration[i, 1] +\
                                                                  self.flux[i, 0]*self.dt*self.beta*\
                                                                  self.nu[self.material[i]] * \
                                                                  self.sig_f[self.material[i]]) / (1 + self.dt*self.lambda_eff)

    """ Solve the linear system of the zero moment neutron transport equation, quasi diffusion equation, and precursor
    concentration equation (with a drift term). """
    def solve_linear_system(self):

        # BUILD THE LINEAR SYSTEM AND THE SOLUTION WILL COME
        # LHS
        n = self.core_mesh_length

        # zeroth moment flux terms
        for position in range(n):

            sig_a = self.sig_t[self.material[position]] - self.sig_s[self.material[position]]
            ave_sig_t = (self.sig_t[self.material[position - 1]] + self.sig_t[self.material[position]]) / 2
            zeta = 1 + self.v * self.dt * (sig_a - (1 - self.beta) * self.nu[self.material[position]] *
                                           self.sig_f[self.material[position]])
            self.linear_system[position, position] = zeta

        courant = self.dt*self.v/self.dx

        # qd flux terms

        column = 0
        for row in range(n + 1, 2*n):
            self.linear_system[row, column] = -courant*self.eddington_factors[column]
            self.linear_system[row, column + 1] = courant*self.eddington_factors[column]
            column += 1

        # precursor equation flux terms
        column = 1
        for row in range(2*n + 2, 3*n+1):
            self.linear_system[row, column] = -self.dt*self.beta*self.nu[self.material[column]]\
                                              *self.sig_f[self.material[column]]
            column += 1

        # zeroth moment current terms
        row = 0
        for column in range(n, 2*n):
            self.linear_system[row, column] = -courant
            self.linear_system[row, column + 1] = courant
            row += 1

        # qd current terms
        position = 0
        for index in xrange(n+1, 2*n):
            ave_sig_t = (self.sig_t[self.material[position - 1]] + self.sig_t[self.material[position]]) / 2
            self.linear_system[index, index] = 1 + self.dt*self.v*ave_sig_t
            position += 1

        self.linear_system[n , n] = 1
        self.linear_system[2*n, 2*n] = 1

        # zeroth moment precursor terms

        row = 0
        for column in range(2 * n + 1, 3 * n + 1):
            self.linear_system[row, column] = -self.v * self.lambda_eff
            row += 1

        # precursor equation precursor terms

        position = 1
        self.linear_system[2*n + 1, 2*n + 1] = 1
        for index in range(2*n + 2, 3 * n + 1):
            self.linear_system[index, index - 1] = -self.dnpc_velocity[position - 1] * self.dt / self.dx
            self.linear_system[index, index] = 1 + self.dnpc_velocity[position] * self.dt / self.dx + self.dt \
                                               * self.lambda_eff
            position += 1



        # RHS

        self.linear_system_solution[0:n, 1] = numpy.array(self.flux[:, 1])
        self.linear_system_solution[n:2*n+1, 1] = numpy.array(self.current[:, 1])
        self.linear_system_solution[2*n+1:, 1] = numpy.array(self.delayed_neutron_precursor_concentration[:, 1])
        # periodic boundary condition on precursor concentration
        self.linear_system_solution[2*n+1, 1] = numpy.array(self.delayed_neutron_precursor_concentration[-1, 1])

        #solve!
        self.linear_system_solution[:, 0] = numpy.linalg.solve(self.linear_system, self.linear_system_solution[:, 1])

        #Assign fluxes!
        self.flux[:, 0] = self.linear_system_solution[0:n, 0]
        self.current[:, 0] = self.linear_system_solution[n:2*n +1, 0]
        self.delayed_neutron_precursor_concentration[:, 0] = self.linear_system_solution[2*n + 1:, 0]


    """Solver for method of manufactured solutions"""
    def solve_linear_system_mms(self):

        # BUILD THE LINEAR SYSTEM AND THE SOLUTION WILL COME
        # LHS
        n = self.core_mesh_length

        # zeroth moment flux terms
        for position in range(n):

            sig_a = self.sig_t[self.material[position]] - self.sig_s[self.material[position]]
            ave_sig_t = (self.sig_t[self.material[position - 1]] + self.sig_t[self.material[position]]) / 2
            zeta = 1 + self.v * self.dt * (sig_a - (1 - self.beta) * self.nu[self.material[position]] *
                                           self.sig_f[self.material[position]])
            self.linear_system[position, position] = zeta

        courant = self.dt*self.v/self.dx

        # qd flux terms

        column = 0
        for row in range(n + 1, 2*n):
            self.linear_system[row, column] = -courant*self.eddington_factors[column]
            self.linear_system[row, column + 1] = courant*self.eddington_factors[column]
            column += 1

        # precursor equation flux terms
        column = 1
        for row in range(2*n + 2, 3*n+1):
            self.linear_system[row, column] = -self.dt*self.beta*self.nu[self.material[column]]\
                                              *self.sig_f[self.material[column]]
            column += 1

        # zeroth moment current terms
        row = 0
        for column in range(n, 2*n):
            self.linear_system[row, column] = -courant
            self.linear_system[row, column + 1] = courant
            row += 1

        # qd current terms
        position = 0
        for index in xrange(n+1, 2*n):
            ave_sig_t = (self.sig_t[self.material[position - 1]] + self.sig_t[self.material[position]]) / 2
            self.linear_system[index, index] = 1 + self.dt*self.v*ave_sig_t
            position += 1

        self.linear_system[n , n] = 1
        self.linear_system[2*n, 2*n] = 1

        # zeroth moment precursor terms

        row = 0
        for column in range(2 * n + 1, 3 * n + 1):
            self.linear_system[row, column] = -self.v * self.lambda_eff
            row += 1

        # precursor equation precursor terms

        position = 1
        self.linear_system[2*n + 1, 2*n + 1] = 1
        for index in range(2*n + 2, 3 * n + 1):
            self.linear_system[index, index - 1] = -self.dnpc_velocity[position - 1] * self.dt / self.dx
            self.linear_system[index, index] = 1 + self.dnpc_velocity[position] * self.dt / self.dx + self.dt \
                                               * self.lambda_eff
            position += 1



        # RHS

        self.calc_q_z_mms()
        self.calc_q_q_mms()
        self.calc_q_p_mms()

        self.linear_system_solution[0:n, 1] = numpy.array(self.flux[:, 1] + self.calc_q_z_mms())
        self.linear_system_solution[n:2*n+1, 1] = numpy.array(self.current[:, 1] + self.calc_q_q_mms())
        self.linear_system_solution[2*n+1:, 1] = numpy.array(self.delayed_neutron_precursor_concentration[:, 1]) + self
        # periodic boundary condition on precursor concentration
        self.linear_system_solution[2*n+1, 1] = numpy.array(self.delayed_neutron_precursor_concentration[-1, 1])

        #solve!
        self.linear_system_solution[:, 0] = numpy.linalg.solve(self.linear_system, self.linear_system_solution[:, 1])

        #Assign fluxes!
        self.flux[:, 0] = self.linear_system_solution[0:n, 0]
        self.current[:, 0] = self.linear_system_solution[n:2*n +1, 0]
        self.delayed_neutron_precursor_concentration[:, 0] = self.linear_system_solution[2*n + 1:, 0]


    """ Solve the transient problem by taking the Eddington factors from a StepCharacteristic solve and putting them
     into a linear system of equations. """
    def solve_transient(self, steps):

        # Initialize arrays to store transient solutions
        flux_t = numpy.zeros([90, steps + 1])
        precursor_t = numpy.zeros([90, steps + 1])

        # Initialize a StepCharacteristic object
        test_moc = moc.StepCharacteristic(self.input_file_name)

        # Record initial conditions
        flux_t[:, 0] = test_moc.flux[:, 1]
        precursor_t[:, 0] = self.delayed_neutron_precursor_concentration[:, 1]

        self.update_variables(test_moc.flux[:, 1], test_moc.current, test_moc.eddington_factors,
                                   test_moc.delayed_neutron_precursor_concentration)

        for iteration in xrange(steps):
            converged = False
            while not converged:
                self.update_eddington(test_moc.eddington_factors)
                # Store previous solutions to evaluate convergence
                last_flux = numpy.array(self.flux[:, 0])
                last_current = numpy.array(self.current[:, 0])
                last_dnpc = numpy.array(self.delayed_neutron_precursor_concentration[:, 0])

                self.solve_linear_system()

                # Calculate difference between previous and present solutions
                flux_diff = abs(last_flux - self.flux[:, 0])
                current_diff = abs(last_current[1:-1] - self.current[1:-1, 0])
                dnpc_diff = abs(last_dnpc - self.delayed_neutron_precursor_concentration[:, 0])
                eddington_diff = abs(test_moc.eddington_factors - test_moc.eddington_factors_old)

                if numpy.max(flux_diff / abs(self.flux[:, 0])) < 1E-6 \
                        and numpy.max(current_diff / abs(self.current[1:-1, 0])) < 1E-6 \
                        and numpy.max(dnpc_diff) < 1E-10\
                        and numpy.max(eddington_diff / test_moc.eddington_factors) < 1E-6:

                    test_moc.iterate_alpha()

                    # Calculate difference between previous and present alpha
                    alpha_diff = abs(test_moc.alpha - test_moc.alpha_old)

                    if numpy.max(alpha_diff/abs(test_moc.alpha)) < 1E-4:
                        converged = True
                        test_moc.flux_t = self.flux[:, 0]

                else:
                    test_moc.update_variables(self.flux[:, 0],
                                              self.delayed_neutron_precursor_concentration[:, 0])
                    test_moc.iterate_alpha()
                    test_moc.solve(False, True)

            self.flux[:, 1] = self.flux[:, 0]
            self.current[:, 1] = self.current[:, 0]
            self.delayed_neutron_precursor_concentration[:, 1] = self.delayed_neutron_precursor_concentration[
                                                                      :, 0]

            flux_t[:, iteration + 1] = test_moc.flux[:, 1]
            precursor_t[:, iteration + 1] = self.delayed_neutron_precursor_concentration[:, 0]

        # plot flux at each time step
        x = numpy.arange(0, self.core_mesh_length)
        ax = plt.subplot(111)
        for iteration in xrange(steps + 1):
            ax.plot(x, flux_t[:, iteration], label= "t = " + str(self.dt * iteration))
        ax.grid(True)
        plt.xlabel('Position [cm]')
        plt.ylabel('Flux [s^-1 cm^-2]')
        plt.title('Grey Group: Neutron Flux')

        # Shrink current axis by 20%
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        # Put a legend to the right of the current axis
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.show()

        # plot precursor concentration at each time step
        ax = plt.subplot(111)
        for iteration in xrange(steps + 1):
            ax.plot(x, precursor_t[:, iteration], label="t = " + str(self.dt * iteration))
        ax.grid(True)
        plt.xlabel('Position [cm]')
        plt.ylabel('Concentration [cm^-3]')
        plt.title('Grey Group: Precursor Concentration')
        # Shrink current axis by 20%
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        # Put a legend to the right of the current axis
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.show()

    def solve_transient_mms(self, steps):

        # Initialize arrays to store transient solutions
        flux_t = numpy.zeros([90, steps + 1])
        precursor_t = numpy.zeros([90, steps + 1])

        # Initialize a StepCharacteristic object
        test_moc = moc.StepCharacteristic(self.input_file_name)

        # Initial flux and precursor concentration
        for position in xrange(self.core_mesh_length):
            self.flux[1, position] = self.psi_0_mms * numpy.sin(position*self.dx)
            self.delayed_neutron_precursor_concentration[1, position] = self.C_0_mms * numpy.sin(position * self.dx)



        # Record initial conditions
        flux_t[:, 0] = test_moc.flux[:, 1]
        precursor_t[:, 0] = self.delayed_neutron_precursor_concentration[:, 1]


        self.update_variables(test_moc.flux[:, 1], test_moc.current, test_moc.eddington_factors,
                                   test_moc.delayed_neutron_precursor_concentration)

        for iteration in xrange(steps):
            converged = False
            t = self.dt
            while not converged:
                self.update_eddington(test_moc.eddington_factors)
                # Store previous solutions to evaluate convergence
                last_flux = numpy.array(self.flux[:, 0])
                last_current = numpy.array(self.current[:, 0])
                last_dnpc = numpy.array(self.delayed_neutron_precursor_concentration[:, 0])

                self.solve_linear_system_mms(t)

                # Calculate difference between previous and present solutions
                flux_diff = abs(last_flux - self.flux[:, 0])
                current_diff = abs(last_current[1:-1] - self.current[1:-1, 0])
                dnpc_diff = abs(last_dnpc - self.delayed_neutron_precursor_concentration[:, 0])
                eddington_diff = abs(test_moc.eddington_factors - test_moc.eddington_factors_old)

                if numpy.max(flux_diff / abs(self.flux[:, 0])) < 1E-6 \
                        and numpy.max(current_diff / abs(self.current[1:-1, 0])) < 1E-6 \
                        and numpy.max(dnpc_diff) < 1E-10\
                        and numpy.max(eddington_diff / test_moc.eddington_factors) < 1E-6:

                    test_moc.iterate_alpha()

                    # Calculate difference between previous and present alpha
                    alpha_diff = abs(test_moc.alpha - test_moc.alpha_old)

                    if numpy.max(alpha_diff/abs(test_moc.alpha)) < 1E-4:
                        converged = True
                        test_moc.flux_t = self.flux[:, 0]
                        t = t + self.dt

                else:
                    test_moc.update_variables(self.flux[:, 0],
                                              self.delayed_neutron_precursor_concentration[:, 0])
                    test_moc.iterate_alpha()
                    test_moc.solve(False, True)

            self.flux[:, 1] = self.flux[:, 0]
            self.current[:, 1] = self.current[:, 0]
            self.delayed_neutron_precursor_concentration[:, 1] = self.delayed_neutron_precursor_concentration[
                                                                      :, 0]

            flux_t[:, iteration + 1] = test_moc.flux[:, 1]
            precursor_t[:, iteration + 1] = self.delayed_neutron_precursor_concentration[:, 0]

        # plot flux at each time step
        x = numpy.arange(0, self.core_mesh_length)
        ax = plt.subplot(111)
        for iteration in xrange(steps + 1):
            ax.plot(x, flux_t[:, iteration], label= "t = " + str(self.dt * iteration))
        ax.grid(True)
        plt.xlabel('Position [cm]')
        plt.ylabel('Flux [s^-1 cm^-2]')
        plt.title('Grey Group: Neutron Flux')

        # Shrink current axis by 20%
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        # Put a legend to the right of the current axis
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.show()

        # plot precursor concentration at each time step
        ax = plt.subplot(111)
        for iteration in xrange(steps + 1):
            ax.plot(x, precursor_t[:, iteration], label="t = " + str(self.dt * iteration))
        ax.grid(True)
        plt.xlabel('Position [cm]')
        plt.ylabel('Concentration [cm^-3]')
        plt.title('Grey Group: Precursor Concentration')
        # Shrink current axis by 20%
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        # Put a legend to the right of the current axis
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.show()

    def calc_q_z_mms(self, t):

        for position in xrange(self.core_mesh_length):
            sig_a = self.sig_t[self.material[position]] - self.sig_t[self.material[position]]
            A = 2*self.psi_0_mms*((1/self.v) + sig_a - (1-self.beta)*self.nu[self.material[position]]\
                                  *self.sig_f[self.material[position]])
            B = self.C_0_mms*self.lambda_eff
            self.q_z_mms[position] = numpy.sin(self.dx * position) * numpy.exp(t) * (A + B)

    def calc_q_q_mms(self, t):

        for position in xrange(self.core_mesh_length):
            self.q_q_mms[position] = (1 / 3) * numpy.cos(self.dx * position) * numpy.exp(t)

    def calc_q_p_mms(self,t):

        for position in xrange(self.core_mesh_length):
            A = self.lambda_eff*self.C_0_mms - 2 * self.beta * self.nu[self.material[position]]\
               * self.sig_f[self.material[position]]
            B = self.a*(numpy.pi - self.dx*position)*self.C_0_mms
            self.q_p_mms[position] = A * numpy.sin(self.dx * position) * numpy.exp(t)\
                                     + B * numpy.cos(self.dx * position) * numpy.exp(t)

if __name__ == "__main__":

    test = QuasiDiffusionPrecursorConcentration("test_input.yaml")  # test for initialization
    test.solve_transient(5)
