#!/usr/bin/env python

# ToDo: comment
# ToDo: make convergence measurements relative

import numpy
import matplotlib.pyplot as plt
import input.read as read
from numba import jitclass, int64, float64
import moc_transport.step_characteristic as moc
from scipy.sparse import csr_matrix
import datetime
import matplotlib.ticker as ticker


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
        self.dnpc_velocity = 10 * numpy.ones(self.core_mesh_length, dtype=numpy.float64)
        self.dnpc_v_edge = numpy.linspace(input_data.data.dnp_velocity_lhs, input_data.data.dnp_velocity_rhs, self.core_mesh_length + 1)
        self.dnpc_velocity = numpy.linspace(input_data.data.dnp_velocity_lhs*(1.0-1.0/(2.0*self.core_mesh_length)), input_data.data.dnp_velocity_rhs*(1.0+1.0/(2.0*self.core_mesh_length)), self.core_mesh_length)


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
        self.psi_0_mms = 1.0  # constant flux coefficient
        self.C_0_mms = 1.0  # constant precursor coefficient
        self.q_z_mms = numpy.zeros((self.core_mesh_length, 1), dtype=numpy.float64)
        self.q_q_mms = numpy.zeros((self.core_mesh_length + 1, 1), dtype=numpy.float64)
        self.q_p_mms = numpy.zeros((self.core_mesh_length, 1), dtype=numpy.float64)
        self.a = 1240.59  # where a*pi is the velocity on the LHS
        self.a = 1273.239544  # where a*pi is the average velocity in the first cell

    """ Update neutron flux, neutron current, Eddington factors, and delayed neutron precursor concentration variables."""
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

    """Solve the linear system of the zero moment neutron transport equation, quasi diffusion equation, and precursor
        concentration equation (with no precursor drift)."""
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

    """Solve the linear system of the zero moment neutron transport equation, quasi diffusion equation, and precursor
    concentration equation (with a drift term)."""
    def solve_linear_system(self):

        # BUILD THE LINEAR SYSTEM AND THE SOLUTION WILL COME
        # LHS
        n = self.core_mesh_length
        courant = self.dt*self.v/self.dx

        # zeroth moment flux terms
        for position in range(n):

            sig_a = self.sig_t[self.material[position]] - self.sig_s[self.material[position]]
            zeta = 1 + self.v * self.dt * (sig_a - (1 - self.beta) * self.nu[self.material[position]] *
                                           self.sig_f[self.material[position]])
            self.linear_system[position, position] = zeta

        # qd flux terms
        column = 0
        for row in range(n + 1, 2*n):
            self.linear_system[row, column] = -courant*self.eddington_factors[column]
            self.linear_system[row, column + 1] = courant*self.eddington_factors[column + 1]
            column += 1

        # precursor equation flux terms
        column = 1
        for row in range(2*n + 2, 3*n+1): # skip first entry, aka BC.
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

        self.linear_system[n , n] = 1 # LHS current boundary condition
        self.linear_system[2*n, 2*n] = 1 # RHS current boundary condition

        # zeroth moment precursor terms
        row = 0
        for column in range(2 * n + 1, 3 * n + 1):
            self.linear_system[row, column] = -self.dt * self.v * self.lambda_eff
            # note: removing self.dt in line above makes effects of precursors more obvious, but it is not correct
            row += 1

        # precursor equation precursor terms
        position = 1
        self.linear_system[2*n + 1, 2*n + 1] = 1
        for index in range(2*n + 2, 3 * n + 1):
            self.linear_system[index, index - 1] = -self.dnpc_v_edge[position] * self.dt / self.dx
            self.linear_system[index, index] = 1 + self.dnpc_v_edge[position + 1] * self.dt / self.dx + self.dt \
                                               * self.lambda_eff
            position += 1

        # RHS
        self.linear_system_solution[0:n, 1] = numpy.array(self.flux[:, 1])
        self.linear_system_solution[n:2*n+1, 1] = numpy.array(self.current[:, 1])
        self.linear_system_solution[2*n+1:, 1] = numpy.array(self.delayed_neutron_precursor_concentration[:, 1])
        # periodic boundary condition on precursor concentration
        self.linear_system_solution[2*n+1, 1] = numpy.array(self.delayed_neutron_precursor_concentration[-1, 1])

        # Solve!
        self.linear_system_solution[:, 0] = numpy.linalg.solve(self.linear_system, self.linear_system_solution[:, 1])

        # Assign solutions!
        self.flux[:, 0] = self.linear_system_solution[0:n, 0]
        self.current[:, 0] = self.linear_system_solution[n:2*n +1, 0]
        self.delayed_neutron_precursor_concentration[:, 0] = self.linear_system_solution[2*n + 1:, 0]

    """Solve the transient problem by taking the Eddington factors from a StepCharacteristic solve and putting them
     into a linear system of equations."""
    def solve_transient(self, steps):

        # Initialize arrays to store transient solutions
        flux_t = numpy.zeros([self.core_mesh_length, steps + 1])
        precursor_t = numpy.zeros([self.core_mesh_length, steps + 1])

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
                        and numpy.max(current_diff) < 1E-10 \
                        and numpy.max(dnpc_diff) < 1E-10\
                        and numpy.max(eddington_diff / test_moc.eddington_factors) < 1E-6:

                    test_moc.iterate_alpha()

                    # Calculate difference between previous and present alpha
                    alpha_diff = abs(test_moc.alpha - test_moc.alpha_old)/abs(test_moc.alpha_old)

                    if numpy.max(alpha_diff) < 1E-4:
                        converged = True
                        test_moc.flux_t = numpy.array(self.flux[:, 0])

                else:
                    test_moc.update_variables(self.flux[:, 0],
                                              self.delayed_neutron_precursor_concentration[:, 0])
                    #test_moc.iterate_alpha()
                    test_moc.solve(False, True)

            self.flux[:, 1] = numpy.array(self.flux[:, 0])
            self.current[:, 1] = numpy.array(self.current[:, 0])
            self.delayed_neutron_precursor_concentration[:, 1] = numpy.array(self.delayed_neutron_precursor_concentration[
                                                                      :, 0])

            flux_t[:, iteration + 1] = numpy.array(self.flux[:, 1])
            precursor_t[:, iteration + 1] = numpy.array(self.delayed_neutron_precursor_concentration[:, 0])

        # plot flux at each time step
        x = numpy.arange(0, self.core_mesh_length)
        ax = plt.subplot(111)
        for iteration in xrange(steps + 1):
            ax.plot(x, flux_t[:, iteration], label= "t = " + "{:.1E}".format(self.dt * iteration))
        ax.grid(True)
        plt.xlabel('Position [cm]')
        plt.ylabel('Flux' + r'$\left[\frac{1}{s cm^{2}}\right]$')
        #plt.title('Neutron Flux')
        plt.tight_layout()

        # Shrink current axis by 20%
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])
        # Put a legend to the right of the current axis
        ax.legend(loc='center left', bbox_to_anchor=(1, -0.1))
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.0e'))
        plt.show()

        # plot precursor concentration at each time step
        ax = plt.subplot(111)
        for iteration in xrange(steps + 1):
            ax.plot(x, precursor_t[:, iteration], label="t = " + "{:.1E}".format(self.dt * iteration))
        ax.grid(True)
        plt.xlabel('Position'+r'[cm]')
        plt.ylabel('DNPC' + r'$\left[\frac{1}{cm^3}\right]$')
        plt.title('Precursor Concentration')
        # Shrink current axis by 20%
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])
        # Put a legend to the right of the current axis
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.show()

        dnpc_filename = "output/precursor_concentration_"+ str(self.input_file_name) +"_N=" + str(self.core_mesh_length) + "_dt=" + str(self.dt) + ".csv"
        flux_filename = "output/flux_" + str(self.input_file_name) +"_N=" + str(self.core_mesh_length) + "_dt=" + str(self.dt) + ".csv"
        numpy.savetxt(dnpc_filename, precursor_t, delimiter=",")
        numpy.savetxt(flux_filename, flux_t, delimiter=",")

    """Solver for method of manufactured solutions."""
    def solve_linear_system_mms(self, t):

        # BUILD THE LINEAR SYSTEM AND THE SOLUTION WILL COME
        # LHS
        n = self.core_mesh_length
        courant = self.dt*self.v/self.dx

        # zeroth moment flux terms
        for position in range(n):

            sig_a = self.sig_t[self.material[position]] - self.sig_s[self.material[position]]
            zeta = 1 + self.v * self.dt * (sig_a - (1 - self.beta) * self.nu[self.material[position]] *
                                           self.sig_f[self.material[position]])
            self.linear_system[position, position] = zeta

        # qd flux terms
        column = 0
        for row in range(n + 1, 2*n):
            self.linear_system[row, column] = -courant*self.eddington_factors[column]
            self.linear_system[row, column + 1] = courant*self.eddington_factors[column+1]
            column += 1

        # precursor equation flux terms
        column = 0
        for row in range(2*n + 1, 3*n+1):
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
            #self.linear_system[index, index] = 1 # debugging
            position += 1

        self.linear_system[n , n] = 1 # boundary condition
        self.linear_system[2*n, 2*n] = 1 # boundary condition

        # zeroth moment precursor terms
        row = 0
        for column in range(2 * n + 1, 3 * n + 1):
            self.linear_system[row, column] = -self.dt * self.v * self.lambda_eff
            row += 1

        # precursor equation precursor terms
        position = 1
        self.linear_system[2*n + 1, 2*n + 1] = 1 + self.dnpc_v_edge[0] * self.dt / self.dx + self.dt \
                                               * self.lambda_eff # boundary condition
        for index in range(2*n + 2, 3 * n + 1):
            self.linear_system[index, index - 1] = -self.dnpc_velocity[position - 1] * self.dt / self.dx
            self.linear_system[index, index] = 1 + self.dnpc_velocity[position] * self.dt / self.dx + self.dt \
                                               * self.lambda_eff
            # might need to evaluate fluxes on cell boundaries, not at cell centers
            self.linear_system[index, index - 1] = -self.dnpc_v_edge[position] * self.dt / self.dx
            self.linear_system[index, index] = 1 + self.dnpc_v_edge[position+1] * self.dt / self.dx + self.dt \
                                               * self.lambda_eff
            position += 1


        # RHS
        self.calc_q_z_mms(t)
        self.calc_q_q_mms(t)
        self.calc_q_p_mms(t)

        self.linear_system_solution[0:n, 1] = numpy.array(self.flux[:, 1] + self.v * self.dt * self.q_z_mms[:, 0])
        self.linear_system_solution[n:2*n+1, 1] = numpy.array(self.current[:, 1] + self.v * self.dt * self.q_q_mms[:, 0])
        self.linear_system_solution[2*n+1:, 1] = numpy.array(self.delayed_neutron_precursor_concentration[:, 1] \
                                                             + self.dt * self.q_p_mms[:, 0])
        #solve! Uses LAPACK routine_gesv that employs an LU decomposition with partial pivoting.
        self.linear_system_solution[:, 0] = numpy.linalg.solve(self.linear_system, self.linear_system_solution[:, 1])

        #Assign solutions!
        self.flux[:, 0] = self.linear_system_solution[0:n, 0]
        self.current[:, 0] = self.linear_system_solution[n:2*n + 1, 0]
        self.delayed_neutron_precursor_concentration[:, 0] = self.linear_system_solution[2*n + 1:, 0]

    """Solve the modified transient problem by taking known Eddington factors and putting them
     into a linear system of equations. Includes MMS source terms."""
    def solve_transient_mms(self, steps):

        # Initialize arrays to store transient solutions
        flux_t = numpy.zeros([self.core_mesh_length, steps + 1])
        precursor_t = numpy.zeros([self.core_mesh_length, steps + 1])

        # Initialize a StepCharacteristic object
        test_moc = moc.StepCharacteristic(self.input_file_name)

        # Initial flux and precursor concentration
        for position in xrange(self.core_mesh_length):
            self.flux[position, 1] = 2.0*self.psi_0_mms * numpy.sin(position*self.dx + self.dx/2.0)
            self.delayed_neutron_precursor_concentration[position, 1] = self.C_0_mms * numpy.sin(position * self.dx + self.dx/2.0)

        self.current[:, 1] = numpy.zeros(self.core_mesh_length+1)
        # Record initial conditions
        flux_t[:, 0] = self.flux[:, 1]
        precursor_t[:, 0] = self.delayed_neutron_precursor_concentration[:, 1]
        # set Eddington factors to MMS values
        self.eddington_factors = (1.0 / 3.0) * numpy.array(numpy.ones(self.core_mesh_length, dtype=numpy.float64))

        t = self.dt
        for iteration in xrange(steps):
            converged = False
            while not converged:

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
                        and numpy.max(dnpc_diff) < 1E-10:

                    #test_moc.iterate_alpha()

                    # Calculate difference between previous and present alpha
                    #alpha_diff = abs(test_moc.alpha - test_moc.alpha_old)

                    #if numpy.max(alpha_diff/abs(test_moc.alpha)) < 1E-4:
                    converged = True
                    test_moc.flux_t = numpy.array(self.flux[:, 0])
                    t = t + self.dt

                else:
                    test_moc.update_variables(self.flux[:, 0],
                                              self.delayed_neutron_precursor_concentration[:, 0])
                    #test_moc.iterate_alpha()
                    #test_moc.solve(False, True)

            self.flux[:, 1] = self.flux[:, 0]
            self.current[:, 1] = self.current[:, 0]
            self.delayed_neutron_precursor_concentration[:, 1] = self.delayed_neutron_precursor_concentration[
                                                                      :, 0]

            flux_t[:, iteration + 1] = self.flux[:, 0]
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

        #save final curves to csv files
        dnpc_filename = "output/precursor_concentration_N=" + str(self.core_mesh_length) + "_dt=" + str(self.dt) + ".csv"
        flux_filename = "output/flux_N=" + str(self.core_mesh_length) + "_dt=" + str(self.dt) + ".csv"
        current_filename = "output/current_N=" + str(self.core_mesh_length) + "_dt=" + str(self.dt) + ".csv"
        numpy.savetxt(dnpc_filename, self.delayed_neutron_precursor_concentration[:, 0], delimiter=",")
        numpy.savetxt(flux_filename, self.flux[:, 0], delimiter=",")
        numpy.savetxt(current_filename, self.current[:, 0], delimiter=",")

    """Solve the modified transient problem by taking the Eddington factors from a StepCharacteristic solve and putting them
     into a linear system of equations. Includes MMS source terms."""
    def solve_transient_mms_coupled(self, steps):

        # Initialize arrays to store transient solutions
        flux_t = numpy.zeros([self.core_mesh_length, steps + 1])
        precursor_t = numpy.zeros([self.core_mesh_length, steps + 1])

        # Initialize a StepCharacteristic object
        test_moc = moc.StepCharacteristic(self.input_file_name)

        # Initial flux and precursor concentration
        for position in xrange(self.core_mesh_length):
            self.flux[position, 1] = 2.0*self.psi_0_mms * numpy.sin(position*self.dx + self.dx/2.0)
            test_moc.flux_t[position] = 2.0 * self.psi_0_mms * numpy.sin(position * self.dx + self.dx / 2.0)
            self.delayed_neutron_precursor_concentration[position, 1] = self.C_0_mms * numpy.sin(position * self.dx + self.dx/2.0)

        self.current[:, 1] = numpy.zeros(self.core_mesh_length+1)
        # Record initial conditions
        flux_t[:, 0] = numpy.array(self.flux[:, 1])
        precursor_t[:, 0] = numpy.array(self.delayed_neutron_precursor_concentration[:, 1])

        t = self.dt
        for iteration in xrange(steps):
            converged = False
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
                        and numpy.max(current_diff) < 1E-6 \
                        and numpy.max(dnpc_diff) < 1E-6\
                        and numpy.max(eddington_diff / test_moc.eddington_factors) < 1E-6:

                    test_moc.update_variables(self.flux[:, 0],
                                              self.delayed_neutron_precursor_concentration[:, 0])
                    test_moc.iterate_alpha()

                    # Calculate difference between previous and present alpha
                    alpha_diff = abs(test_moc.alpha - test_moc.alpha_old)

                    if numpy.max(alpha_diff) < 1E-6:
                        converged = True
                        test_moc.flux_t = numpy.array(self.flux[:, 0])
                        t = t + self.dt
                        print "time step t=" + str(t) + " completed. " + str(datetime.datetime.now().time())

                    else:
                        test_moc.solve_mms(t, False, False)

                else:
                    test_moc.update_variables(self.flux[:, 0],
                                              self.delayed_neutron_precursor_concentration[:, 0])
                    test_moc.solve_mms(t, False, False)

            self.flux[:, 1] = numpy.array(self.flux[:, 0])
            self.current[:, 1] = numpy.array(self.current[:, 0])
            self.delayed_neutron_precursor_concentration[:, 1] = numpy.array(self.delayed_neutron_precursor_concentration[
                                                                      :, 0])

            flux_t[:, iteration + 1] = numpy.array(self.flux[:, 0])
            precursor_t[:, iteration + 1] = numpy.array(self.delayed_neutron_precursor_concentration[:, 0])

        # # plot flux at each time step
        # x = numpy.arange(0, self.core_mesh_length)
        # ax = plt.subplot(111)
        # for iteration in xrange(steps + 1):
        #     ax.plot(x, flux_t[:, iteration], label= "t = " + str(self.dt * iteration))
        # ax.grid(True)
        # plt.xlabel('Position [cm]')
        # plt.ylabel('Flux [s^-1 cm^-2]')
        # plt.title('Grey Group: Neutron Flux')
        #
        # # Shrink current axis by 20%
        # box = ax.get_position()
        # ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        # # Put a legend to the right of the current axis
        # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        # plt.show()
        #
        # # plot precursor concentration at each time step
        # ax = plt.subplot(111)
        # for iteration in xrange(steps + 1):
        #     ax.plot(x, precursor_t[:, iteration], label="t = " + str(self.dt * iteration))
        # ax.grid(True)
        # plt.xlabel('Position [cm]')
        # plt.ylabel('Concentration [cm^-3]')
        # plt.title('Grey Group: Precursor Concentration')
        # # Shrink current axis by 20%
        # box = ax.get_position()
        # ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        # # Put a legend to the right of the current axis
        # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        # plt.show()

        #save final curves to csv files
        dnpc_filename = "output/precursor_concentration_N=" + str(self.core_mesh_length) + "_dt=" + str(self.dt) + ".csv"
        flux_filename = "output/flux_N=" + str(self.core_mesh_length) + "_dt=" + str(self.dt) + ".csv"
        current_filename = "output/current_N=" + str(self.core_mesh_length) + "_dt=" + str(self.dt) + ".csv"
        numpy.savetxt(dnpc_filename, self.delayed_neutron_precursor_concentration[:, 0], delimiter=",")
        numpy.savetxt(flux_filename, self.flux[:, 0], delimiter=",")
        numpy.savetxt(current_filename, self.current[:, 0], delimiter=",")
        completion_message = "Done! Saved N=" + str(self.core_mesh_length) + " dt=" + str(self.dt) + ".csv final time step solutions. "
        print(completion_message)

    def calc_q_z_mms(self, t):

        for position in xrange(self.core_mesh_length):
            sig_a = self.sig_t[self.material[position]] - self.sig_s[self.material[position]]
            A = 2.0*self.psi_0_mms*((1.0/self.v) + sig_a - (1-self.beta)*self.nu[self.material[position]]\
                                  *self.sig_f[self.material[position]])
            B = self.C_0_mms*self.lambda_eff
            self.q_z_mms[position, 0] = numpy.sin(self.dx * position + self.dx/2.0) * numpy.exp(t) * (A - B)

    def calc_q_q_mms(self, t):

        for position in xrange(self.core_mesh_length+1):
            self.q_q_mms[position, 0] = (2.0 / 3.0) * self.psi_0_mms * numpy.cos(self.dx * position) * numpy.exp(t)

        # accomodate initial conditions
        self.q_q_mms[0, 0] = 0.0
        self.q_q_mms[-1,0] = 0.0

    def calc_q_p_mms(self, t):

        for position in xrange(self.core_mesh_length):
            A = self.C_0_mms*(1+self.lambda_eff-self.a) - 2.0 * self.beta * self.nu[self.material[position]]\
               * self.sig_f[self.material[position]] * self.psi_0_mms
            B = self.a*(2.0*numpy.pi - self.dx*position - self.dx/2.0)*self.C_0_mms
            B = self.dnpc_velocity[position] * self.C_0_mms
            self.q_p_mms[position, 0] = A * numpy.sin(self.dx * position + self.dx/2.0) * numpy.exp(t)\
                                     + B * numpy.cos(self.dx * position + self.dx/2.0) * numpy.exp(t)

if __name__ == "__main__":

    #test = QuasiDiffusionPrecursorConcentration("mms_input.yaml")  # test for initialization
    #test.solve_transient_mms(100)

    #mms1 = QuasiDiffusionPrecursorConcentration("mms_inputs/n50dt01.yaml")  # test for initialization
    #mms1.solve_transient_mms(100)

    #mms2 = QuasiDiffusionPrecursorConcentration("mms_inputs/n100dt005.yaml")  # test for initialization
    #mms2.solve_transient_mms(200)

    #mms3 = QuasiDiffusionPrecursorConcentration("mms_inputs/n200dt0025.yaml")  # test for initialization
    #mms3.solve_transient_mms(400)

    #mms4 = QuasiDiffusionPrecursorConcentration("mms_inputs/n500dt001.yaml")  # test for initialization
    #mms4.solve_transient_mms(1000)

    #mms5 = QuasiDiffusionPrecursorConcentration("mms_inputs/n1000dt0005.yaml")  # test for initialization
    #mms5.solve_transient_mms(2000)

    # illustrative result
    test = QuasiDiffusionPrecursorConcentration("test_input.yaml")  # test for initialization
    test.solve_transient(5)

    # test
    #mms1 = QuasiDiffusionPrecursorConcentration("mms_inputs/n50dt01.yaml")  # test for initialization
    #mms1.solve_transient_mms_coupled(10)

    #test = QuasiDiffusionPrecursorConcentration("mms_input.yaml")  # test for initialization
    #test.solve_transient_mms_coupled(100)

    #mms1 = QuasiDiffusionPrecursorConcentration("mms_inputs/n50dt01.yaml")  # test for initialization
    #mms1.solve_transient_mms_coupled(100)

    #mms2 = QuasiDiffusionPrecursorConcentration("mms_inputs/n100dt005.yaml")  # test for initialization
    #mms2.solve_transient_mms_coupled(200)

    #mms3 = QuasiDiffusionPrecursorConcentration("mms_inputs/n200dt0025.yaml")  # test for initialization
    #mms3.solve_transient_mms_coupled(400)

    #mms4 = QuasiDiffusionPrecursorConcentration("mms_inputs/n500dt001.yaml")  # test for initialization
    #mms4.solve_transient_mms_coupled(1000)

    #mms5 = QuasiDiffusionPrecursorConcentration("mms_inputs/n1000dt0005.yaml")  # test for initialization
    #mms5.solve_transient_mms_coupled(2000)
