import matplotlib.pyplot as plt
import numpy as np
import math

#63958d64-3703-46b2-ba6c-4a6cf038dded
def bvector(x):
    """
    

    Parameters
    ----------
    x: TYPE: integer
        x value to sub in to obtain b vector

    Returns
    -------
    TYPE: numpy array
        Numpy array of the bvector in this case always (0, 0)

    """
    return np.array([0, 0]).T  # bvector


# Q1
# def rk3(A, b, y0, interval, N):
#     def rk3step(xn, yn, h):  #
#         y1 = yn + h * (np.dot(A, yn) + b(xn))
#         y2 = 0.75 * yn + 0.25 * y1 + 0.25 * h * (np.dot(A, y1) + b(xn + h))
#         return (yn + 2 * y2 + 2 * h * (np.dot(A, y2) + b(xn + h))) / 3
#
#     x = np.linspace(*interval, N + 1)
#     y = [y0]
#     for n in range(N):
#         y.append(rk3step(x[n], y[n], x[n + 1] - x[n]))
#     return x, y
def rk3(A, b, y0, interval, N):
    """
    

    Parameters
    ----------
    A : TYPE: Matrix: list of lists or numpy array
        This should be the matrix describing the system.
    b : TYPE: vector: numpy array
        bvector  
    y0 : TYPE: vector: list
        Intial start point.
    interval : TYPE: numpy array
        Range of values x can be, minumum and maximum value are the two elements of the array
    N : TYPE: integer
        Number of points within the interval to check

    Returns
    -------
    x :TYPE: List of x numpy array of y
        The function returns a list of floats containing all x values
    y :TYPE: Numpy array 
        Returns a numpy array of all calculated y values for the rk3 algorithm

    """
    assert isinstance(N, int) and N > 0, f'Number of points was expected to be a positive integer, your N is :{N}'
    assert type(A).__name__ == 'ndarray', f'Matrix A is not a numpy array. Please verify you have entered a ' \
                                          f'appropriate numpy array. A: {A}'
    assert type(b).__name__ == 'function', f'b is a function returning a numpy array. b defined is of type {type(b)}.'
    assert type(
        y0).__name__ == 'ndarray', f'y0 should be an numpy array containing initial data. At present, y0 = {type(y0)} '
    assert type(interval).__name__ == 'ndarray', f'interval should be a numpy array containing start and end values. ' \
                                                 f'At present, interval is a {type(interval)} '
    assert len(A) == len(y0), f'A and y0 do not have the same length'
    assert len(A) == len(b(0)), f'A and b do not have the same length'
    x = np.linspace(*interval, N + 1)

    def rk3step(xn, yn, h):
        y1 = yn + h * (np.dot(A, yn) + b(xn))  # calculation to determine y1
        y2 = 0.75 * yn + 0.25 * y1 + 0.25 * h * (np.dot(A, y1) + b(xn + h))  # calculate y2
        return (yn + 2 * y2 + 2 * h * (np.dot(A, y2) + b(xn + h))) / 3  # function returns yn+1

    y = [y0]  # Initial value of y put into a list
    for n in range(N):  # For every N value
        y.append(rk3step(x[n], y[n], x[n + 1] - x[n]))  # add the next y iteration to list
    return x, np.asarray(y).T  # Return x and y


# Q2
mu = 0.5 * (1 - (1 / math.sqrt(3)))  # constants as defined in question
v = 0.5 * (math.sqrt(3) - 1)
gamma = 3 / (2 * (3 + math.sqrt(3)))
lam = (3 * (1 + math.sqrt(3))) / (2 * (3 + math.sqrt(3)))


def dirk3(A, b, y0, interval, N):
    """
    

    Parameters
    ----------
    A : TYPE: Matrix: list of lists or numpy array
        This should be the matrix describing the system.
    b : TYPE: vector: numpy array
        bvector  
    y0 : TYPE: vector: list
        Intial start point.
    interval : TYPE: numpy array
        Range of values x can be, minumum and maximum value are the two elements of the array
    N : TYPE: integer
        Number of points within the interval to check

    Returns
    -------
    x :TYPE: List of x numpy array of y
        The function returns a list of floats containing all x values
    y :TYPE: Numpy array 
        Returns a numpy array of all calculated y values for the dirk3 algorithm


    """
    assert isinstance(N, int) and N > 0, f'Number of points was expected to be a positive integer, your N is :{N}'
    assert type(A).__name__ == 'ndarray', f'Matrix A is not a numpy array. Please verify you have entered a ' \
                                          f'appropriate numpy array. A: {A}'
    assert type(b).__name__ == 'function', f'b is a function returning a numpy array. b defined is of type {type(b)}.'
    assert type(
        y0).__name__ == 'ndarray', f'y0 should be an numpy array containing initial data. At present, y0 = {type(y0)} '
    assert type(interval).__name__ == 'ndarray', f'interval should be a numpy array containing start and end values. ' \
                                                 f'At present, interval is a {type(interval)} '
    assert len(A) == len(y0), f'A and y0 do not have the same length'
    assert len(A) == len(b(0)), f'A and b do not have the same length'

    # assert (isinstance(A, int) and N >= 0, f'Number of points was expected to be a positive integer, N:{N}')
    def dirk3step(xn, yn, h):
        z = np.identity(np.size(A, 1)) - (h * mu * A)  # Define LHS of equation
        y1 = np.dot(np.linalg.inv(z), (yn + h * mu * b(xn + mu * h)))  # inverse of LHS * RHS for y1
        y2 = np.dot(np.linalg.inv(z),
                    y1 + h * v * (np.dot(A, y1) + b(xn + h * mu)) + h * mu * b(xn + h * v + 2 * h * mu))  # y2
        return (1 - lam) * yn + lam * y2 + h * gamma * (np.dot(A, y2) + b(xn + h * v + 2 * h * mu))  # return yn+1

    x = np.linspace(*interval, N + 1)  # within defined interval, assign x values
    y = [y0]  # Initial value of y put into a list

    for n in range(N):  # For every N value
        y.append(dirk3step(x[n], y[n], x[n + 1] - x[n]))  # add the next y iteration to list
    return x, np.asarray(y).T  # Return x and y


a1 = 1000  # Define problem
a2 = 1
A = np.array(([-a1, 0], [a1, -a2]))  # Matrix A
b = bvector  # bvector is passed as an argument
y0 = np.array([1, 0]).T  # Initial y value
interval = np.array([0, 0.1])  # Interval to search in


def error(method, a1, a2):
    """
    

    Parameters
    ----------
    method : TYPE: string
        Chosen method to be applied is passed in 
    a1 : TYPE:  integer
        The value in the matrix A in the bottom left hand corner
    a2 : TYPE:  integer
        The value in the matrix A in the bottom right hand corner

    Returns
    -------
    errors : TYPE: List
        List containing errors for all N values 
    h_list : TYPE: List
        List containing all h values
    x : TYPE: List
        List of x values
    y_exact : TYPE: numpy array
        Array containing the vectors of the exact y solutions 
    y : TYPE: numpy array
        Array containing the vectors of the calculated y solutions 

    """
    assert isinstance(method, str) and (method == 'DIRK3' or method == 'RK3'), f'Method only accepts string RK3 or ' \
                                                                               f'DIRK3, you inputted: {method}'
    assert isinstance(a1, int) or isinstance(a1, float), f'a1 should be a integer or float, currently a1= {a1}'
    assert isinstance(a2, int) or isinstance(a2, float), f'a2 should be a integer or float, currently a2= {a2}'
    k = 1  # As defined in question
    h_list = []  # List of h values to use for graph
    errors = []  # Initialise list to store and return errors
    while k < 11:  # For k values between 1 and 10
        N = k * 40  # As defined in question
        if method == 'RK3':  # Determining the method the error is to be calculated for
            x, y = rk3(A, b, y0, interval, N)  # Returns RK3 results
        elif method == 'DIRK3':
            x, y = dirk3(A, b, y0, interval, N)  # Returns DIRK3 results
        h = (x[-1] - x[0]) / N  # Spacing calculated with equation in question
        h_list.append(h)  # All h values added to list for graph
        x = np.linspace(0, 0.1, N + 1)  # x
        y_exact = []  # Initialise list for all exact y values to be returned
        for i in x:  # Calculate every y value for each x
            result = np.array(([np.exp(-a1 * i)], [(a1 / (a1 - a2)) * (np.exp(-a2 * i) - np.exp(-a1 * i))]))
            # equation defined in question
            y_exact.append(result)  # Add results to list
        # y_exact = np.asarray(y_exact)  # Convert list to numpy array
        # y_exact.transpose()
        #  print(y[:, 6])
        error = 0  # Reset error to 0
        j = 1  # start at 1 to avoid divide by 0 for error

        while N >= j:  # For all elements in y calculate error
            calc = (y[1, j] - y_exact[j][1]) / y_exact[j][1]
            # error += np.linalg.norm((y[:, j] - y_exact[:][j]) / y_exact[:][j], ord=1)  # error as defined in the
            # error += np.linalg.norm(calc, ord=1)
            error += np.linalg.norm(calc, ord=1)
            # question
            j += 1  # increment
        # error = error * (x[n + 1] - x[n])
        errors.append(h * error)  # entire list multiplied by h to calculate error
        k += 1  # Increment to complete calculate error for next N value

    return errors, h_list, x, y_exact, y  # returns list of error, h, x values, exact and calculated y values


errors_rk3, h, x_rk3, y_exact, y_rk3 = error('RK3', a1, a2)  # RK3 method and error calculated

plt.loglog(h, errors_rk3, 'x', label='rk3 error')  # Graph with both axes logarithmic plotted of errors vs h for RK
plt.title('Errors vs h for rk3')
plt.xlabel('h')
plt.ylabel('Errors')
plt.show()

fit = np.polyfit(np.log(h), np.log(errors_rk3), deg=3)  # Fitting the errors and h to a linear line attempting get a
# gradient of 1
plt.loglog(np.exp(fit[0] * np.log(h) + fit[1]), label=f'slope = {fit[0]}')
print('The slope is approximately 1 when polyfit is set to third degree hence it is converging at the third order.')
plt.legend()
plt.title('Polyfit error graph for rk3. Look at console.')
plt.show()

errors_dirk3, h, x_dirk3, y_exact, y_dirk3 = error('DIRK3', a1, a2)  # DIRK3 method and error calculated
plt.loglog(h, errors_dirk3, 'x', label='dirk3 error')  # Graph with both axes logarithmic plotted of errors vs h for
plt.xlabel('h')
plt.ylabel('Errors')
plt.title('Error vs h for dirk 3')
plt.show()

fit = np.polyfit(np.log(h), np.log(errors_dirk3), deg=3)  # Fitting the errors and h to a linear line attempting get a
# gradient of 1
plt.loglog(np.exp(fit[0] * np.log(h) + fit[1]), label=f'slope = {fit[0]}')
plt.legend()
plt.title('Polyfit error graph for dirk3.')
plt.show()
y_exact = np.asarray(y_exact)
y_rk3 = np.asarray(y_rk3)

# Plot all graphs for dirk3 and rk3 x vs y with a comparison of exact vs calculated value
fig, axs = plt.subplots(2)
fig.suptitle('y1 vs x (rk3)')
axs[0].plot(x_rk3, y_rk3[0, :], label='solution')
axs[0].set_title('Calculated y')
axs[1].plot(x_rk3, y_exact[:, 0], label='exact', color='red')
axs[1].set_title('Exact y')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

figure, axs = plt.subplots(2)
figure.suptitle('y2 vs x (rk3)')
axs[0].plot(x_rk3, y_rk3[1, :], label='solution')
axs[0].set_title('Calculated y')
axs[1].plot(x_rk3, y_exact[:, 1], label='exact', color='red')
axs[1].set_title('Exact y')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

fig, axs = plt.subplots(2)
fig.suptitle('y1 vs x (dirk3)')
axs[0].plot(x_dirk3, y_dirk3[0, :], label='solution')
axs[0].set_title('Calculated y')
axs[1].plot(x_dirk3, y_exact[:, 0], label='exact', color='red')
axs[1].set_title('Exact y')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

figure, axs = plt.subplots(2)
figure.suptitle('y2 vs x (dirk3)')
axs[0].plot(x_dirk3, y_dirk3[1, :], label='solution')
axs[0].set_title('Calculated y')
axs[1].plot(x_dirk3, y_exact[:, 1], label='exact', color='red')
axs[1].set_title('Exact y')
plt.xlabel('x')
plt.ylabel('y')
plt.show()


#  Q4
def bvectorQ4(z):
    """
    

    Parameters
    ----------
    z : TYPE: integer
        x value to sub in to obtain b vector

    Returns
    -------
    TYPE: numpy array
        Numpy array of the bvector

    """
    assert isinstance(z, int) or isinstance(z, float), f'z should be a integer or float, currently z= {z}'
    return np.array(
        ([np.cos(10 * z) - 10 * np.sin(10 * z), 199 * np.cos(10 * z) - 10 * np.sin(10 * z), 208 * np.cos(10 * z)
          + 10000 * np.sin(10 * z)])).T  # bvector as defined in question


# Values as set out in question
interval = np.array([0, 1])
A = np.array(([-1, 0, 0], [-99, -100, 0], [-10098, 9900, -10000]))
y0 = np.array([0, 1, 0]).T
b = bvectorQ4


def errorQ4(method):
    """
    

    Parameters
    ----------
    method : TYPE: string
        Chosen method to be applied is passed in 

    Returns
    -------
    errors : TYPE: List
        List containing errors for all N values 
    h_list : TYPE: List
        List containing all h values
    x : TYPE: List
        List of x values
    y_exact : TYPE: numpy array
        Array containing the vectors of the exact y solutions 
    y : TYPE: numpy array
        Array containing the vectors of the calculated y solutions 

    """
    assert isinstance(method, str) and (method == 'DIRK3' or method == 'RK3'), f'Method only accepts string RK3 or ' \
                                                                               f'DIRK3, you inputted: {method}'
    k = 4
    h_list = []
    errors = []
    while k < 17:
        N = k * 200
        if method == 'RK3':
            x, y = rk3(A, b, y0, interval, N)
        elif method == 'DIRK3':
            x, y = dirk3(A, b, y0, interval, N)
        h = (x[-1] - x[0]) / N
        h_list.append(h)  # add h values to list for plotting
        x = np.linspace(0, 1, N + 1)
        y_exact = []
        for i in x:
            result = np.array(([np.cos(10 * i) - np.exp(-i)], [np.cos(10 * i) + np.exp(-i) - np.exp(-100 * i)],
                               [np.sin(10 * i) + 2 * np.exp(-i) - np.exp(-100 * i) - np.exp(-10000 * i)]))
            # equation to obtain exact y values
            y_exact.append(result)
        y_exact = np.asarray(y_exact)
        y_exact.transpose()
        error = 0  # Reset error to 0 for every iteration
        j = 1  # Reset j to 1 for every iteration, 1 as to avoid division by 0 for initial data

        while N >= j:
            calc = (y[2, j] - y_exact[j][2]) / y_exact[j][2]  # Error as defined in the question using y3
            error += np.linalg.norm(calc, ord=1)
            j += 1  # increment

        errors.append(h * error)  # all error values multiplied by h as specified by equation
        k += 1  # Iterate to repeat process for all required values of N
    return errors, h_list, x, y_exact, y


# DIRK3 error for Q4
errors_dirk3, h, x_dirk3, y_exact, y_dirk3 = errorQ4('DIRK3')

plt.loglog(h, errors_dirk3, label='dirk3 error')
plt.xlabel('h')
plt.ylabel('Errors')
plt.title('Error graph for dirk 3 Q4')
plt.show()

y_exact = np.asarray(y_exact)
y_dirk3 = np.asarray(y_dirk3)
# Below is plotting y3 calculated values vs expected for DIRK3
fig, axs = plt.subplots(2)
fig.suptitle('y1 vs x (dirk3) Q4')
axs[0].plot(x_dirk3, y_dirk3[0, :], label='solution')
axs[0].set_title('Calculated y')

axs[1].plot(x_dirk3, y_exact[:, 0], label='exact', color='red')
axs[1].set_title('Exact y')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
# Below is plotting y2 calculated values vs expected for DIRK3
fig, axs = plt.subplots(2)
fig.suptitle('y2 vs x (dirk3) Q4')
axs[0].plot(x_dirk3, y_dirk3[1, :], label='solution')
axs[0].set_title('Calculated y')

axs[1].plot(x_dirk3, y_exact[:, 1], label='exact', color='red')
axs[1].set_title('Exact y')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
# Below is plotting y3 calculated values vs expected for DIRK3
fig, axs = plt.subplots(2)
fig.suptitle('y3 vs x (dirk3) Q4')
axs[0].plot(x_dirk3, y_dirk3[2, :], label='solution')
axs[0].set_title('Calculated y')

axs[1].plot(x_dirk3, y_exact[:, 2], label='exact', color='red')
axs[1].set_title('Exact y')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

#  Plotting error graph for dirk3 for Q4 with the first iteration removed to show a linear trend
plt.title('Errors vs h (dirk) Q4 with anomaly removed')
h.pop(0)
errors_dirk3.pop(0)
plt.loglog(h, errors_dirk3, label='dirk3 error')
plt.show()

errors_rk3, h, x_rk3, y_exact, y_rk3 = errorQ4('RK3')

# Below is plotting y1 calculated values vs expected
fig, axs = plt.subplots(2)
fig.suptitle('y1 vs x (rk3) Q4')
axs[0].plot(x_rk3, y_rk3[0, :], label='solution')
axs[0].set_title('Calculated y')

axs[1].plot(x_rk3, y_exact[:, 0], label='exact', color='red')
axs[1].set_title('Exact y')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
# Below is plotting y2 calculated values vs expected
fig, axs = plt.subplots(2)
fig.suptitle('y2 vs x (rk3) Q4')
axs[0].plot(x_rk3, y_rk3[1, :], label='solution')
axs[0].set_title('Calculated y')

axs[1].plot(x_rk3, y_exact[:, 1], label='exact', color='red')
axs[1].set_title('Exact y')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# Below is plotting y3 calculated values vs expected
fig, axs = plt.subplots(2)
fig.suptitle('y3 vs x (rk3) Q4')
axs[0].plot(x_rk3, y_rk3[2, :], label='solution')
axs[0].set_title('Calculated y')

axs[1].plot(x_rk3, y_exact[:, 2], label='exact', color='red')
axs[1].set_title('Exact y')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
