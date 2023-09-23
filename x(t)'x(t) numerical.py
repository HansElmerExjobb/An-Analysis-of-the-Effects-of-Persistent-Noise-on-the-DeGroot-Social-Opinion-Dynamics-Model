# %%
import numpy as np
import random as rand
import networkx as nx
import matplotlib.pyplot as plt
import random as rand
from scipy.optimize import minimize
# %%
#Creates adjacency matrix from graph object
def Create_Adj(G):
    Adj_sparse = nx.adjacency_matrix(G)
    return np.array(Adj_sparse.todense()), Adj_sparse

#Plotting of graphs
def Plot_G(G):
    pos = nx.spring_layout(G) #Plotting given graph
    nx.draw(G, pos=pos, with_labels=True)
    plt.show()

#Creates P from adjacency matrix
def Create_P(A):
    degree_vector = np.sum(A, axis=1)
    D = np.diag(degree_vector)
    P = np.linalg.inv(D)@A
    return P

#Determines expected convergent consensus alpha (no noise)
def expected_conv_consensus(P):
    # Compute the left eigenvectors and eigenvalues of the transpose of P
    eigvals, eigvecs = np.linalg.eig(np.transpose(P))
    # Find the eigenvector associated with the eigenvalue of 1
    inv_dist = np.real_if_close(eigvecs[:, np.isclose(eigvals, 1)].flatten())
    # Normalize the invariant distribution
    inv_dist /= np.sum(inv_dist)
    alpha = np.transpose(inv_dist)@x_0
    return alpha
#%%
#Function updates opinions according to x(t+1) = Px(t) starting with x(0) to x(t).
#Returns all values over time for each node
def opinions_over_time(x_0: list,P,max_t: int,noise: np.ndarray):
    x_t = np.zeros((len(x_0),max_t+1))
    x_t[:,0] = x_0
    for t in range(0,max_t):
        #Update opinion for each time, t. Adds noise if there is any
        x_t[:,t+1] = P@x_t[:,t] + noise[:,t]
    return x_t
#%%
"""
Create graph and set initial opinions
"""
num_nodes = 150 # Number of nodes in the graph
p = 0.3 # Probability of a edge existing between two nodes (directed)
seed_nr = 42
# Generate a random unweighted graph using the Erdős-Rényi model and binomial
G_erdos = nx.erdos_renyi_graph(num_nodes, p, directed = True, seed=seed_nr)
G_binomial = nx.binomial_graph(num_nodes, p, directed = True, seed=seed_nr)
A, A_sparse = Create_Adj(G_erdos)
P = Create_P(A)

# Sets the intial opinions of the network as random numbers with uniform dist.
x_0 = [rand.uniform(0, 1) for i in range(num_nodes)]
#%%
"""
No Noise convergence
"""
def no_noise(max_time: int):
    #max_time = max_time #How many times should the opinion dynamics update?
    noise_empty = np.zeros((num_nodes,max_time)) #Empty noise (no noise)
    noise = noise_empty

    x_t = opinions_over_time(x_0,P,max_time,noise)
    avg_x_t = np.mean(x_t,axis=0)
    return avg_x_t

max_time = 40
avg_x_t = no_noise(max_time)
# Plot the consensus over time
plt.plot(range(max_time+1),avg_x_t)
plt.xlabel('Number of times opinion updated')
plt.ylabel('Average opinion of network')
plt.title('Change of average opinion in a consensus network over time')

alpha = expected_conv_consensus(P)
print('Expected consensus: {:.6f}'.format(alpha))
print('Consensus stabilizes at approx: {:.6f}'.format(avg_x_t[-1]))
print('Error: {}'.format(alpha-avg_x_t[-1]))
# %%
"""
Initial Noise at t=1.
"""
def initial_noise(mu: float,sigma: float,max_time: int):
    # Creates gaussian distributed random noise at t = 1 only
    noise_mu = mu
    noise_sigma = sigma
    noise_initial = np.random.normal(loc = noise_mu, scale = noise_sigma, size=(num_nodes,1))
    noise_initial = np.pad(noise_initial, ((0,0),(0,max_time-1)), mode='constant', constant_values=0)
    noise = noise_initial

    x_t = opinions_over_time(x_0,P,max_time,noise)
    avg_x_t = np.mean(x_t,axis=0)
    return avg_x_t

noise_mu = 0
noise_sigma = 0.1
max_time = 40 #How many times should the opinion dynamics update?
avg_x_t = initial_noise(noise_mu, noise_sigma, max_time)
# Plot the consensus over time
plt.plot(range(max_time+1),avg_x_t)
plt.xlabel('Number of times opinion updated')
plt.ylabel('Average opinion of network')
plt.title('Change of average opinion in a consensus network over time')

alpha = expected_conv_consensus(P)
print('Expected consensus: {:.6f}'.format(alpha))
print('Consensus stabilizes at approx: {:.6f}'.format(avg_x_t[-1]))
# %%
"""
Noise over time
"""
def noise_over_time(mu: float,sigma: float,max_time: int):
    # Creates gaussian distrubuted random noise over time
    noise_mu = mu
    noise_sigma = sigma
    noise_over_time = np.random.normal(loc = noise_mu, scale = noise_sigma, size=(num_nodes,max_time))
    noise = noise_over_time

    x_t = opinions_over_time(x_0,P,max_time,noise)
    avg_x_t = np.mean(x_t,axis=0)
    return avg_x_t

noise_mu = 0
noise_sigma = 0.1
max_time = 40 #How many times should the opinion dynamics update?
avg_x_t = noise_over_time(max_time)
# Plot the consensus over time
plt.plot(range(max_time+1),avg_x_t)
plt.xlabel('Number of times opinion updated')
plt.ylabel('Average opinion of network')
plt.title('Change of average opinion in a consensus network over time')

alpha = expected_conv_consensus(P)
print('Expected consensus: {:.6f}'.format(alpha))
print('Consensus stabilizes at approx: {:.6f}'.format(avg_x_t[-1]))
# %%
"""
Computes variance of consensus over time
"""
max_time = 40
iterations = 1000
mu = 0
sigma = 0.1
avg_consensus_no_noise = np.zeros((max_time+1,iterations))
avg_consensus_intial_noise = np.zeros((max_time+1,iterations))
avg_consensus_time_noise = np.zeros((max_time+1,iterations))
for i in range(iterations):
    avg_consensus_no_noise[:,i] = no_noise(max_time)
    avg_consensus_intial_noise[:,i] = initial_noise(mu,sigma,max_time)
    avg_consensus_time_noise[:,i] = noise_over_time(mu,sigma,max_time)
avg = np.zeros((max_time+1,3))
avg[:,0] = np.mean(avg_consensus_no_noise,axis=1)
avg[:,1] = np.mean(avg_consensus_intial_noise,axis=1)
avg[:,2] = np.mean(avg_consensus_time_noise,axis=1)
# %%
#Creates all the matrices for the graphs
ring_graph = np.matrix([
    [0, 1, 0, 0, 0 , 1],
    [1, 0, 1, 0, 0 , 0],
    [0, 1, 0, 1, 0 , 0],
    [0, 0, 1, 0, 1 , 0],
    [0, 0, 0, 1, 0 , 1],
    [1, 0, 0, 0, 1 , 0]])
barbell_graph = np.matrix([
    [0, 1, 1, 0, 0 , 1],
    [1, 0, 1, 1, 0 , 0],
    [1, 1, 0, 0, 0 , 0],
    [0, 1, 0, 0, 1 , 1],
    [0, 0, 0, 1, 0 , 1],
    [0, 0, 0, 1, 1 , 0]])
complete_graph = np.matrix([
    [0, 1, 1, 1, 1 , 1],
    [1, 0, 1, 1, 1 , 1],
    [1, 1, 0, 1, 1 , 1],
    [1, 1, 1, 0, 1 , 1],
    [1, 1, 1, 1, 0 , 1],
    [1, 1, 1, 1, 1 , 0]])
wheel_graph = np.matrix([
    [0, 1, 0, 0, 1, 1],
    [1, 0, 1, 0, 0, 1],
    [0, 1, 0, 1, 0, 1],
    [0, 0, 1, 0, 1, 1],
    [1, 0, 0, 1, 0, 1],
    [1, 1, 1, 1, 1, 0]])
star_graph = np.matrix([
    [0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 0]])
utility_graph = np.matrix([
    [0, 0, 0, 1, 1, 1],
    [0, 0, 0, 1, 1, 1],
    [0, 0, 0, 1, 1, 1],
    [1, 1, 1, 0, 0, 0],
    [1, 1, 1, 0, 0, 0],
    [1, 1, 1, 0, 0, 0]])
matrices = []
matrices.extend([ring_graph,barbell_graph, complete_graph, wheel_graph, star_graph, utility_graph])
matrices_names = ["Ring graph", "Barbell graph", "Complete graph", "Wheel graph", "Star graph", "Utility graph"]
#%%
#Determines the total variance given an initial row stochastic matrix P.
def tot_var_calc(P: np.matrix,max_t: int):
    sum = np.zeros(shape = (max_t,1))
    sum[0] = P.shape[0]
    P_updated = P
    for t in range(1,max_t):
        in_sum = 0
        for i in range(P.shape[0]):
            for j in range(P.shape[0]):
                in_sum += P_updated[i,j]**2
        print(in_sum)
        sum[t] = sum[t-1]+in_sum
        #print(P_updated)
        P_updated = P_updated@P
    print("")
    return sum

# %%
max_t = 4
tot_var = np.empty(shape=(max_t,6))
for i in range(len(matrices)):
    A = matrices[i]
    degree_vector = np.sum(A, axis=1)
    D = np.diag(degree_vector.A1)
    P = np.linalg.inv(D)@A
    tot_var[:,i] = tot_var_calc(P,max_t)[:,0]
    plt.plot(range(len(tot_var)),tot_var[:,i])
plt.legend(matrices_names)
plt.title("Total variance over time for six graphs")
plt.xlabel("Time")
plt.ylabel("Total variance, E(x(t)'x(t))")

#%%
def minmax(sign:int, terminal_time:int,P:np.matrix): #positive input if minimizing, negative if maximizing.
    #Minimizing or maximizing total variance by varying where variance is located
    T = terminal_time
    P_upd = P
    squares = np.zeros(shape=(T+1,P.shape[0]))

    def sums_squared_column(P): #Sums the squared values of each column in P
        squares_temp = np.zeros(shape=(1,P.shape[0]))
        for i in range(P.shape[0]):
            temp_square_sum = 0
            for j in range(P.shape[0]):
                temp_square_sum += P[j,i]**2
            squares_temp[0,i] = temp_square_sum
        return squares_temp
    
    #Creates 3d array that contains [values,time] wise summed squares of columns in P
    squares[0,:] = sums_squared_column(P)
    for t in range(1,T+1):
        P_upd = P_upd@P
        squares[t,:] = sums_squared_column(P_upd)

    # Define initial guess
    initial_guess = np.ones(shape = squares.shape)
    initial_guess_sigma = initial_guess*(1/initial_guess.size)
    squares_2d = np.array
    igs_2d = np.array

    # Making arrays 2d so they work better with optimization
    for i in range(squares.shape[0]):
        squares_2d = np.append(squares_2d,squares[i,:])
        igs_2d = np.append(igs_2d, initial_guess_sigma[i,:])
    squares_2d = np.delete(squares_2d, 0)
    igs_2d = np.delete(igs_2d,0)

    # Define the objective function
    def objective(sigma):
        sum = 0
        for i in range(squares.size):
            sum += squares_2d[i]*sigma[i]
        return np.sign(sign)*sum

    # Define the constraint
    def constraint(sigma):
        return sigma.sum() - 1

    # Define the bounds for x
    bounds = [(0,1)]*igs_2d.size

    # Define the optimization problem
    problem = {
        'type': 'eq',
        'fun': constraint
    }

    # Solve the optimization problem
    result = minimize(objective, igs_2d, method='SLSQP', bounds=bounds, constraints=problem)
    opt_sigma= result.x
    min_var = result.fun
    # Print the result
    return result
#%%
T = 100 # Terminal time t
A = matrices[0] #0=ring, 1=barbell, 2=complete, 3=wheel, 4=star, 5=utility 
degree_vector = np.sum(A, axis=1)
D = np.diag(degree_vector.A1)
P = np.linalg.inv(D)@A #Initial P [row,col]
eigvals, eigvecs = np.linalg.eig(np.transpose(P))
# Find the eigenvector associated with the eigenvalue of 1
inv_dist = np.real_if_close(eigvecs[:, np.isclose(eigvals, 1)].flatten())
# Normalize the invariant distribution
inv_dist /= np.sum(inv_dist)
min_var = minmax(1,T,P) #Positive if minimizing
max_var = minmax(-1,T,P) #Negative if maximizing
opt_sigma_min = min_var.x
opt_var_min = min_var.fun
opt_sigma_max = max_var.x
opt_var_max = -max_var.fun
print(f"Min var: {opt_var_min}\nMax var: {opt_var_max}")
# %%
from scipy.optimize import root

# Define the system of equations
def equations(x):
    a, b, c, d, e, f, g, h, i = x
    eq1 = a + d + g - 1
    eq2 = a**2 + d**2 + g**2 -1/3
    eq3 = b+e+h-1
    eq4 = b**2 + e**2 + h**2 -1/3
    eq5 = c+f+i-1
    eq6 = c**2 + f**2 + i**2 -1/3
    eq7 = a+b+c-1
    eq8 = d+e+f-1
    eq9 = g+h+i-1
    return [eq1, eq2, eq3, eq4, eq5, eq6, eq7, eq8, eq9]

# Initial guess for the variables
x0 = [1,0,0,0,1,0,0,0,1]

# Solve the equations
result = root(equations, x0)

# Check if the solver was successful
if result.success:
    print("Solution found:")
    print(result.x)
else:
    print("Failed to find a solution.")
# %%
