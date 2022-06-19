import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt


# Shooting method used as it is fast and accurate and is go to method to BVP as long as IVP does not have infinite
# solutions for some values of z. Can sometimes fail to satisfy boundary conditions, initial guess is required which
# means a good one needs to be made else it may fail. Another disadvantage of the shooting method is if BVP has
# multiple solutions, it may converge to the wrong one. It is possible IVP generated may be unstable.
#
# Secant method used to find root as it is faster than the Newton method and Newtons abilities not necessary for this
# task.
#

def rhs(s, q, fg, fx):
    """


    Parameters
    ----------
    s : TYPE: Float
        DESCRIPTION: Point along arc length of hair
    q : TYPE: Numpy array
        DESCRIPTION: Array containing all parameters: theta, theta hat, x and z
    fg : TYPE: Float
        DESCRIPTION: Force of gravity on hair
    fx : TYPE: Float
        DESCRIPTION: Force of wind on hair

    Returns
    -------
    dqds : TYPE: Numpy array
        DESCRIPTION: Returns the derivative of all parameters in the q array

    """
    assert isinstance(s, float) or isinstance(s,
                                              int), f's was expected to be a positive float, your displacement is :{s}'
    assert type(q).__name__ == 'ndarray', f'Matrix q is not a numpy array. Please verify you have entered a ' \
                                          f'appropriate numpy array. q: {q}'
    assert isinstance(fg, float), f'fg was expecting a positive float whereas yours was: {fg}'
    assert isinstance(fx, float) or isinstance(fx, int), f'fx was expecting a positive float whereas yours was: {fx}'
    dqds = np.zeros_like(q)  # Initialise empty array so index referencing can be used

    # assign values for angles and their derivatives based on q passed in
    theta = q[0]
    theta_hat = q[1]
    # Derivative of q. Index 2, 3 are derivative of x and z respectively
    # Equations defined on problem sheet
    dqds[0] = theta_hat
    dqds[1] = fg * s * np.cos(theta) + fx * s * np.sin(theta)
    dqds[2] = np.cos(theta)
    dqds[3] = np.sin(theta)
    # derivative returned
    return dqds


def shooting_error(theta_hat_0, interval, R, fg, fx, theta_0):
    """


    Parameters
    ----------
    theta_hat_0 : TYPE: Float
        DESCRIPTION: Initial derivative of theta
    interval : TYPE: List
        DESCRIPTION: List containing length of hair and 0 meaning initial displacement
    R : TYPE: Int
        DESCRIPTION: Radius of head
    fg : TYPE: Float
        DESCRIPTION: Force of gravity on hair in cm^-3
    fx : TYPE: Float
        DESCRIPTION: Force of wind on hair in cm^-3
    theta_0 : TYPE: Float
        DESCRIPTION.Theta angle

    Returns
    -------
    TYPE: Float
        DESCRIPTION: Returns solution of initial value problem

    """
    assert isinstance(theta_hat_0, float) and theta_hat_0 <= np.pi, f'The initial value for Theta hat is meant to be a ' \
                                                                    f'float value less than or equal to 1. Yours: {theta_hat_0} '
    assert type(
        interval).__name__ == 'list', f'Interval is meant to be a list with 2 elements yours is, {type(interval)}'
    assert isinstance(fg, float), f'fg was expecting a positive float whereas yours was: {fg}'
    assert isinstance(fx, float) or isinstance(fx, int), f'fx was expecting a positive float whereas yours was: {fx}'
    assert isinstance(R, int) or isinstance(R, float), f'R is meant to be the radius of head and is meant to be a ' \
                                                       f'float or integer, yours is {type(R)} '
    assert isinstance(theta_0, float), f'theta was expecting a float whereas yours was: {theta_0}'
    # x and z coordinates of hair calculated
    x0 = R * np.cos(theta_0)
    z0 = R * np.sin(theta_0)
    # Initialise q with parameters needed for initial value problem
    q0 = [theta_0, theta_hat_0, x0, z0]
    sol = scipy.integrate.solve_ivp(rhs, interval, q0, args=(fg, fx))
    return sol.y[1, -1]


def hair_bvp_2d(theta_0_all, L, R, fx, fg=0.1):
    """


    Parameters
    ----------
    theta_0_all : TYPE: Numpy array
        DESCRIPTION. Array containing all theta values
    L : TYPE: Integer
        DESCRIPTION: Length of hair
    R : TYPE: Int
        DESCRIPTION: Radius of head
    fx : TYPE: Float
        DESCRIPTION: Force of wind in cm^-3
    fg : TYPE: Float, optional
        DESCRIPTION. The default is 0.1. Force of gravity in cm^-3

    Returns
    -------
    x : TYPE: Numpy array
        DESCRIPTION: Array containing x coordinates of the hair, 100 points per hair in this case
    z : TYPE: Numpy array
        DESCRIPTION: Array containing x coordinates of the hair, 100 points per hair in this case

    """
    assert isinstance(fg, float), f'fg was expecting a positive float whereas yours was: {fg}'
    assert isinstance(fx, float) or isinstance(fx, int), f'fx was expecting a positive float whereas yours was: {fx}'
    assert isinstance(R, int) or isinstance(R, float), f'R is meant to be the radius of head and is meant to be a ' \
                                                       f'float or integer, yours is {type(R)} '
    assert type(theta_0_all).__name__ == 'ndarray', f'Matrix theta_0_all is not a numpy array. Please verify you have ' \
                                                    f'entered a appropriate numpy array. Type of theta_0_all: {type(theta_0_all)}'
    assert isinstance(L, float) or isinstance(L, int) and L > 0, f'L was expecting a positive integer or float ' \
                                                                 f'whereas yours was type: {type(L)} and L = {L}'

    # Length of hair to perform operation on
    interval = [0, L]
    # Avoiding hard coding value of number of hairs
    N_hairs = len(theta_0_all)
    # 100 points between start and end point of hair
    s = np.linspace(*interval, 100)
    # Initialise arrays to allow for index referencing later on
    x = np.zeros((N_hairs, len(s)))
    z = np.zeros((N_hairs, len(s)))
    for hair in range(N_hairs):
        theta_0 = theta_0_all[hair]
        x0 = R * np.cos(theta_0)
        z0 = R * np.sin(theta_0)
        # Initial guess for newton method
        if theta_0 < np.pi / 2:
            initial_guess = -0.5
        else:
            initial_guess = 0.5
        theta_hat_0 = scipy.optimize.newton(shooting_error, initial_guess,
                                            args=(interval, R, fg, 0, theta_0))
        # If there is a wind component do a newton solver with the force of wind and initial guess of no wind case
        if fx != 0:
            theta_hat_0 = scipy.optimize.newton(shooting_error, theta_hat_0,
                                                args=(interval, R, fg, fx, theta_0))
        # initialise q with all parameters required for initial value problem
        q0 = [theta_0, theta_hat_0, x0, z0]
        # Solve initial value problem
        sol = scipy.integrate.solve_ivp(rhs, interval, q0, args=(fg, fx), dense_output=True)
        x[hair, :] = sol.sol(s)[2, :]  # ODE solution for x
        z[hair, :] = sol.sol(s)[3, :]  # ODE solution for z
    return x, z


# Conditions as set out in question
fg = 0.1
L = 4.0
R = 10
fx = 0

# Generate circle to be used as head
theta_head = np.linspace(0, np.pi * 2)
x_head = R * np.cos(theta_head)
z_head = R * np.sin(theta_head)
# Plot head
plt.plot(x_head, z_head, lw=4)
# 20 theta values between 0 and pi for each hair to be placed on head evenly spaced
theta_0_all = np.linspace(0, np.pi, 20)
# X and Z values obtained by solving the boundary value problem
x, z = hair_bvp_2d(theta_0_all, L, R, fx)

# Connect x and z values (same index together) for each hair and plot
for x_hair, z_hair in zip(x, z):
    plt.plot(x_hair, z_hair, lw=1, color='k')
plt.xlabel(r'$x$')
plt.ylabel(r'$z$')
plt.title('Hair in no wind case and gravity at 0.1')
plt.gca().set_aspect('equal')
plt.show()

# Case including wind
fx = 0.1
x, z = hair_bvp_2d(theta_0_all, L, R, fx)
plt.plot(x_head, z_head, lw=4)
for x_hair, z_hair in zip(x, z):
    plt.plot(x_hair, z_hair, lw=1, color='k')
plt.xlabel(r'$x$')
plt.ylabel(r'$z$')
plt.title('Hair in wind with force at 0.1 and gravity at 0.1')
plt.gca().set_aspect('equal')
plt.show()


def rhs3d(s, q, fg, fx):
    """


    Parameters
    ----------
    s : TYPE: Float
        DESCRIPTION: Point along arc length of hair
    q : TYPE: Numpy array
        DESCRIPTION: Array containing all parameters: theta, theta hat, x and z
    fg : TYPE: Float
        DESCRIPTION: Force of gravity on hair
    fx : TYPE: Float
        DESCRIPTION: Force of wind on hair

    Returns
    -------
    dqds : TYPE: Numpy array
        DESCRIPTION: Returns the derivative of all parameters in the q array

    """
    assert isinstance(s, float) or isinstance(s,
                                              int), f's was expected to be a positive float, your displacement is :{s}'
    assert type(q).__name__ == 'ndarray', f'Matrix q is not a numpy array. Please verify you have entered a ' \
                                          f'appropriate numpy array. q: {q}'
    assert isinstance(fg, float), f'fg was expecting a positive float whereas yours was: {fg}'
    assert isinstance(fx, float) or isinstance(fx, int), f'fx was expecting a positive float whereas yours was: {fx}'
    dqds = np.zeros_like(q)  # Initialise empty array so index referencing can be used

    # assign values for angles and their derivatives based on q passed in
    theta = q[0]
    theta_hat = q[1]
    phi = q[2]
    phi_hat = q[3]

    # Derivative of q. Index 4, 5, 6 are derivative of x, y and z respectively
    # Equations defined on problem sheet
    dqds[0] = theta_hat
    dqds[1] = fg * s * np.cos(theta) + fx * s * np.sin(theta) * np.cos(phi)
    dqds[2] = phi_hat
    dqds[3] = -s * fx * np.sin(phi) * np.sin(theta)
    dqds[4] = np.cos(theta) * np.cos(phi)
    dqds[5] = -np.sin(phi) * np.cos(theta)
    dqds[6] = np.sin(theta)

    # derivative returned
    return dqds


def shooting_error3d(theta_hat_0, phi_hat_0, interval, R, fg, fx, theta_0, phi_0):
    """


    Parameters
    ----------

    theta_hat_0 : TYPE: Float
        DESCRIPTION: Initial derivative of theta
    phi_hat_0 : TYPE: Float
        DESCRIPTION: Initial derivative of phi
    interval : TYPE: List
        DESCRIPTION: List containing length of hair and 0 meaning initial displacement
    R : TYPE: Int
        DESCRIPTION: Radius of head
    fg : TYPE: Float
        DESCRIPTION: Force of gravity on hair in cm^-3
    fx : TYPE: Float
        DESCRIPTION: Force of wind on hair in cm^-3
    theta_0 : TYPE: Float
        DESCRIPTION: Theta angle
    phi_0 : TYPE: Float
        DESCRIPTION: Phi angle

    Returns
    -------
    TYPE: Float
        DESCRIPTION: Theta hat 0 derivative

    """
    assert isinstance(theta_hat_0, float) and theta_hat_0 <= 1, f'The initial value for Theta hat is meant to be a ' \
                                                                f'float value less than or equal to 1. Yours: {theta_hat_0} '
    assert isinstance(phi_hat_0, float) and phi_hat_0 <= 1, f'The initial value for Phi hat is meant to be a ' \
                                                            f'float value less than or equal to 1. Yours: {phi_hat_0}'
    assert type(
        interval).__name__ == 'list', f'Interval is meant to be a list with 2 elements yours is, {type(interval)}'
    assert isinstance(fg, float), f'fg was expecting a positive float whereas yours was: {fg}'
    assert isinstance(fx, float) or isinstance(fx, int), f'fx was expecting a positive float whereas yours was: {fx}'
    assert isinstance(R, int) or isinstance(R, float), f'R is meant to be the radius of head and is meant to be a ' \
                                                       f'float or integer, yours is {type(R)} '
    assert isinstance(theta_0, float), f'Theta was expecting a float whereas yours was: {theta_0}'
    assert isinstance(phi_0, float), f'Phi was expecting a float whereas yours was: {phi_0}'
    x0 = R * np.cos(theta_0) * np.cos(phi_0)  # x0 calculated
    y0 = -R * np.cos(theta_0) * np.sin(phi_0)  # y0 calculated
    z0 = R * np.sin(theta_0)  # z0 calculated
    # Equations as defined in problem sheet
    # Calculate x, y and z of hair base
    # put all variables into list q
    q0 = [theta_0, theta_hat_0, phi_0, phi_hat_0, x0, y0, z0]
    # Solve initial value problem to include effects of gravity and wind
    sol = scipy.integrate.solve_ivp(rhs3d, interval, q0, args=(fg, fx))
    # Return final value of theta_hat_0
    return sol.y[1, -1]


# Shooting error for phi as it returns derivative of phi hat 0 instead of the
def shooting_error_phi(theta_hat_0, phi_hat_0, interval, R, fg, fx, theta_0, phi_0):
    """


        Parameters
        ----------

        theta_hat_0 : TYPE: Float
            DESCRIPTION: Initial derivative of theta
        phi_hat_0 : TYPE: Float
            DESCRIPTION: Initial derivative of phi
        interval : TYPE: List
            DESCRIPTION: List containing length of hair and 0 meaning initial displacement
        R : TYPE: Int
            DESCRIPTION: Radius of head
        fg : TYPE: Float
            DESCRIPTION: Force of gravity on hair in cm^-3
        fx : TYPE: Float
            DESCRIPTION: Force of wind on hair in cm^-3
        theta_0 : TYPE: Float
            DESCRIPTION: Theta angle
        phi_0 : TYPE: Float
            DESCRIPTION: Phi angle

        Returns
        -------
        TYPE: Float
            DESCRIPTION. Phi_hat_0 derivative

        """
    assert isinstance(theta_hat_0, float) and theta_hat_0 <= 1, f'The initial value for Theta hat is meant to be a ' \
                                                                f'float value less than or equal to 1. Yours: {theta_hat_0} '
    assert isinstance(phi_hat_0, float) and phi_hat_0 <= 1, f'The initial value for Phi hat is meant to be a ' \
                                                            f'float value less than or equal to 1. Yours: {phi_hat_0}'
    assert type(
        interval).__name__ == 'list', f'Interval is meant to be a list with 2 elements yours is, {type(interval)}'
    assert isinstance(fg, float), f'fg was expecting a positive float whereas yours was: {fg}'
    assert isinstance(fx, float) or isinstance(fx, int), f'fx was expecting a positive float whereas yours was: {fx}'
    assert isinstance(R, int) or isinstance(R, float), f'R is meant to be the radius of head and is meant to be a ' \
                                                       f'float or integer, yours is {type(R)} '
    assert isinstance(theta_0, float), f'Theta was expecting a float whereas yours was: {theta_0}'
    assert isinstance(phi_0, float), f'Phi was expecting a float whereas yours was: {phi_0}'
    x0 = R * np.cos(theta_0) * np.cos(phi_0)
    y0 = -R * np.cos(theta_0) * np.sin(phi_0)
    z0 = R * np.sin(theta_0)

    # put all variables into list q
    q0 = [theta_0, theta_hat_0, phi_0, phi_hat_0, x0, y0, z0]
    # Solve initial value problem to include effects of gravity and wind
    sol = scipy.integrate.solve_ivp(rhs, interval, q0, args=(fg, fx))
    return sol.y[3, -1]


def hair_bvp_3d(theta_0_all, phi_0_all, L, R, fx, fg):
    """

    Parameters
    ----------
    theta_0_all : TYPE: Numpy array
        DESCRIPTION. Array containing all theta values
    phi_0_all : TYPE: Numpy array
        DESCRIPTION. Array containing all phi values
    L : TYPE: Integer
        DESCRIPTION: Length of hair
    R : TYPE: Int
        DESCRIPTION: Radius of head
    fx : TYPE: Float
        DESCRIPTION: Force of wind in cm^-3
    fg : TYPE: Float, optional
        DESCRIPTION. The default is 0.1. Force of gravity in cm^-3

    Returns
    -------
    x : TYPE: Numpy array
        DESCRIPTION: Array containing x coordinates of the hair, 100 points per hair in this case
    y : TYPE
        DESCRIPTION.
    z : TYPE: Numpy array
        DESCRIPTION: Array containing x coordinates of the hair, 100 points per hair in this case

    """
    assert isinstance(fg, float), f'fg was expecting a positive float whereas yours was: {fg}'
    assert isinstance(fx, float) or isinstance(fx, int), f'fx was expecting a positive float whereas yours was: {fx}'
    assert isinstance(R, int) or isinstance(R, float), f'R is meant to be the radius of head and is meant to be a ' \
                                                       f'float or integer, yours is {type(R)} '
    assert type(theta_0_all).__name__ == 'ndarray', f'Matrix theta_0_all is not a numpy array. Please verify you have ' \
                                                    f'entered a appropriate numpy array. Type of theta_0_all: {type(theta_0_all)}'
    assert type(phi_0_all).__name__ == 'ndarray', f'Matrix phi_0_all is not a numpy array. Please verify you have ' \
                                                  f'entered a appropriate numpy array. Type of phi_0_all: {type(phi_0_all)}'
    assert isinstance(L, float) or isinstance(L, int) and L > 0, f'L was expecting a positive integer or float ' \
                                                                 f'whereas yours was type: {type(L)} and L = {L}'
    # Length of hair to perform operation on
    interval = [0, L]
    # Avoiding hard coding value of number of hairs
    N_hairs = len(theta_0_all)
    # 100 points between start and end point of hair
    s = np.linspace(*interval, 100)
    # Initialise arrays to allow for index referencing later on
    x = np.zeros((N_hairs, len(s)))
    y = np.zeros((N_hairs, len(s)))
    z = np.zeros((N_hairs, len(s)))
    for hair in range(N_hairs):
        theta_0 = theta_0_all[hair]
        phi_0 = phi_0_all[hair]
        x0 = R * np.cos(theta_0) * np.cos(phi_0)
        y0 = -R * np.cos(theta_0) * np.sin(phi_0)
        z0 = R * np.sin(theta_0)
        # Initial guess for newton method
        if theta_0 < np.pi / 2:
            guess = -0.5
        else:
            guess = 0.5
        # Initialise a guess for phi hat 0 to allow for theta to be optimised
        phi_hat_0 = 0.1
        # Calculate without wind first to get close to answer
        theta_hat_0_no_wind = scipy.optimize.newton(shooting_error3d, guess,
                                                    args=(phi_hat_0, interval, R, fg, 0, theta_0, phi_0))
        # Calculate with wind
        theta_hat_0 = scipy.optimize.newton(shooting_error3d, theta_hat_0_no_wind,
                                            args=(phi_hat_0, interval, R, fg, fx, theta_0, phi_0))
        # Same procedure as theta
        phi_hat_0_no_wind = scipy.optimize.newton(shooting_error_phi, guess,
                                                  args=(theta_hat_0, interval, R, fg, 0, theta_0, phi_0))
        phi_hat_0 = scipy.optimize.newton(shooting_error_phi, phi_hat_0_no_wind,
                                          args=(theta_hat_0, interval, R, fg, fx, theta_0, phi_0))
        # Initialise q to be passed into initial value problem
        q0 = [theta_0, theta_hat_0, phi_0, phi_hat_0, x0, y0, z0]
        # Solve IVP
        sol = scipy.integrate.solve_ivp(rhs3d, interval, q0, args=(fg, fx), dense_output=True)
        # Obtain results for x, y and z from variable
        x[hair, :] = sol.sol(s)[4, :]
        y[hair, :] = sol.sol(s)[5, :]
        z[hair, :] = sol.sol(s)[6, :]
        # Return x, y and z coordinates
    return x, y, z


# Initialise parameters
fg = 0.1
L = 4.0
R = 10
fx = 0.05
# Create theta and phi values in the range specified on a 10 x 10 mesh grid
theta_0 = np.linspace(0, 0.49 * np.pi, 10)
phi_0 = np.linspace(0, np.pi, 10)
theta, phi = np.meshgrid(theta_0, phi_0)

theta_0_all = np.reshape(theta, (theta_0.size * phi_0.size))
phi_0_all = np.reshape(phi, (theta_0.size * phi_0.size))
# Get x, y and z values
x, y, z = hair_bvp_3d(theta_0_all, phi_0_all, L, R, fx, fg)
fig = plt.figure()
ax = fig.gca(projection='3d')

# Link the x, y and z coordinates by matching elements using zip and plot
for x_hair, y_hair, z_hair in zip(x, y, z):
    ax.plot(x_hair, y_hair, z_hair, lw=1, color='k')
# Graph axes labelling and title
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$y$')
ax.set_zlabel(r'$z$')
ax.set_title('3D case with wind at 0.05 at +x and gravity at 0.1 without head')
# plt.gca().set_aspect('equal')
plt.show()

# Make sphere to use as head
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
a = R * np.outer(np.cos(u), np.sin(v))
b = R * np.outer(np.sin(u), np.sin(v))
c = R * np.outer(np.ones(np.size(u)), np.cos(v))

fig = plt.figure()
ax = fig.gca(projection='3d')
# Plot the sphere
ax.plot_surface(a, b, c, color='white', alpha=0.3)
for x_hair, y_hair, z_hair in zip(x, y, z):
    ax.plot(x_hair, y_hair, z_hair, lw=1, color='k')
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$y$')
ax.set_zlabel(r'$z$')
ax.set_title('3D case with wind at 0.05 from +x and gravity at 0.1 with head')
plt.show()

# Create 2d head for x-z and y-z

head = np.linspace(0, 2 * np.pi, 100)
x_head = R * np.cos(head)
z_head = R * np.sin(head)
plt.subplot(1, 2, 1)
plt.plot(x_head, z_head, 'r-', lw=4)
# Link the x and z coordinates by matching elements using zip and plot
for x_hair, z_hair in zip(x, z):
    plt.plot(x_hair, z_hair, 'k', lw=1)
plt.xlabel(r'$x$')
plt.ylabel(r'$z$')
plt.title('x-z plane')
plt.gca().set_aspect('equal')

plt.subplot(1, 2, 2)
plt.plot(x_head, z_head, 'r-', lw=4)
# Link the y and z coordinates by matching elements using zip and plot
for y_hair, z_hair in zip(y, z):
    plt.plot(y_hair, z_hair, 'k', lw=1)
plt.xlabel(r'$y$')
plt.ylabel(r'$z$')
plt.title('y-z plane')
plt.subplots_adjust(wspace=0.4)
plt.gca().set_aspect('equal')
plt.show()
# This is not a cross sectional view so all hairs are shown. I can also do a cross section view to only output hairs
# on one side using if statements but don't think that was what was asked of me
