# %%
import numpy as np
import random as rand
import networkx as nx
import matplotlib.pyplot as plt
import random as rand
import scipy as scipy
from scipy.sparse import csr_matrix
from scipy.sparse import coo_matrix
from scipy.optimize import minimize
import os
#import graph_tool.all as gt
import pandas as pd
import csv

#df = pd.read_excel('network.gml.zst')
#df = pd.read_csv('edges.csv')
# df_us_agencies = pd.read_csv('US_agencies/edges.csv')
# nodes_df_us_agencies = pd.read_csv('US_agencies/nodes.csv')
df_polblog = pd.read_csv('network_polblogs.csv/edges.csv')
df_nodes_polblog = pd.read_csv('network_polblogs.csv/nodes.csv')

#nodes_df = pd.read_csv('nodes.csv')

#dj_sparse = csr_matrix((n,n),dtype=np.int8)
#adj_sparse = csr_matrix(df.values.T, dtype = np.int8)
#%%
n = df_nodes_polblog.shape[0]#number of nodes
df = df_polblog
row = np.array(df.get("# source"))
col = np.array(df.get(" target"))
val = np.squeeze(np.ones(shape=(1,row.size),dtype=np.int8))
sa = coo_matrix((val, (row,col)), shape=(n,n), dtype=np.int8)
del row, col, val
A = sa.todense()
#sa1 = sa.tocsr() #This one seems easier
#sa2 = sa.tocsc()
#Converting from coo to csc (csr could also work) 
#because python more easily works with these sparse matrices.
def IsConnected(A): #Good for checking if strongly connected or not
    flag = True
    for i in range(A.A.shape[0]):
        if not A.A[i].any():
            flag = False
    return flag
print("Is graph connected?", IsConnected(A))
del df_nodes_polblog, df_polblog, df
#%%
# Removing unconnected components from a directed graph
def remove_unconnected_comp_strong(G2):
    if nx.is_strongly_connected(G2):
        return G2
    largest_cc = max(nx.strongly_connected_components(G2), key=len)
    for component in list(nx.strongly_connected_components(G2)):
        if len(component)<len(largest_cc):
            for node in component:
                G2.remove_node(node)
    return G2
def fix_indexation_issue_strong(G):
        # Assuming you have already removed the nodes not connected to the largest component and have the graph G
    # largest_connected_component is the largest connected component of the graph G
    largest_connected_component = max(nx.strongly_connected_components(G), key=len)
    # Use the largest_connected_component to get the subgraph containing only the nodes of the largest connected component
    largest_connected_subgraph = G.subgraph(largest_connected_component)

    # Create a mapping of old node labels to new labels with continuous numerical order
    node_mapping = {node: i for i, node in enumerate(sorted(largest_connected_subgraph.nodes()))}

    # Create a new graph with the updated node labels and edges
    new_G = nx.DiGraph() if G.is_directed() else nx.Graph()

    for old_node, new_node in node_mapping.items():
        new_G.add_node(new_node)

    for u, v, attrs in largest_connected_subgraph.edges(data=True):
        new_u = node_mapping[u]
        new_v = node_mapping[v]
        new_G.add_edge(new_u, new_v, **attrs)
    return new_G
def remove_unconnected_comp_weak(G2):
    if nx.is_weakly_connected(G2):
        return G2
    largest_cc = max(nx.weakly_connected_components(G2), key=len)
    for component in list(nx.weakly_connected_components(G2)):
        if len(component)<len(largest_cc):
            for node in component:
                G2.remove_node(node)
    return G2
def fix_indexation_issue_weak(G):
        # Assuming you have already removed the nodes not connected to the largest component and have the graph G
    # largest_connected_component is the largest connected component of the graph G
    largest_connected_component = max(nx.weakly_connected_components(G), key=len)
    # Use the largest_connected_component to get the subgraph containing only the nodes of the largest connected component
    largest_connected_subgraph = G.subgraph(largest_connected_component)

    # Create a mapping of old node labels to new labels with continuous numerical order
    node_mapping = {node: i for i, node in enumerate(sorted(largest_connected_subgraph.nodes()))}

    # Create a new graph with the updated node labels and edges
    new_G = nx.DiGraph() if G.is_directed() else nx.Graph()

    for old_node, new_node in node_mapping.items():
        new_G.add_node(new_node)

    for u, v, attrs in largest_connected_subgraph.edges(data=True):
        new_u = node_mapping[u]
        new_v = node_mapping[v]
        new_G.add_edge(new_u, new_v, **attrs)
    return new_G
#Making strongly connected
G = nx.DiGraph(sa.todense().A)
#G_s = G.copy() #G_s will have the strongly connected version
G_w = G.copy() #G_w will have the weakly connected version
#G_s = remove_unconnected_comp_strong(G_s)
#G_s = fix_indexation_issue_strong(G_s)
G_w = remove_unconnected_comp_weak(G_w)
G_w = fix_indexation_issue_weak(G_w)
# Matrix is atm contained in A in a ndarray
def to_numpy_matrix(G):

    # create an empty array
    N_size = len(G.nodes())
    E = np.zeros(shape=(N_size, N_size))

    for i, j in G.edges():
        E[i,j] = 1
    for i, j, attr in G.edges(data=True):
        E[i, j] = attr.get("weight")

    return E
def add_self_loop(A):
    new_A = A.copy()
    for i in range(len(A)):
        if not new_A[i].any():
            new_A[i][i] = 1
    return new_A
#A_s = to_numpy_matrix(G_s)
#A_s_matrix = np.asmatrix(A_s)
A_w = to_numpy_matrix(G_w)
A_w = add_self_loop(A_w)
A_w_matrix = np.asmatrix(A_w)
#Creates P from adjacency matrix
def Create_P(A):
    degree_vector = np.sum(A,axis=1)
    D = np.diagflat(degree_vector)
    P = np.linalg.inv(D)@A
    return P
#P_s = Create_P(A_s_matrix)
P_w = Create_P(A_w_matrix)


P = P_w #P_w är weakly connected, P_s är strongly connected
#Constructs P^1, P^2, P^3, and so on. P_over_time[0] = P^0, P_over_time[n] = P^n
def construct_P_over_time(P_temp,max_t):
    n = len(P_temp)
    P_over_time_temp = np.empty((max_t+1, *(n,n)),dtype=float)
    P_upd = np.identity(n)
    P_over_time_temp[0] = P_upd

    interval = 10  # Percentage interval for printing progress (e.g., 10%)
    step = max_t // (100 // interval)  # Calculate the step size
    for t in range(max_t):
        P_upd = P_upd@P_temp
        P_over_time_temp[t+1] = P_upd
        # Check if the current iteration is at an interval step
        if (t + 1) % step == 0 or t == max_t - 1:
            progress = ((t + 1) / max_t) * 100
            print(f"Progress: {progress:.2f}%")
    print("Complete.")
    return P_over_time_temp
num_P = 1000
import pickle

try:
    P_over_time
except NameError:
    # If noise2 is not defined, initialize it as None
    P_over_time = None

file_name = 'P_over_time.pkl'
if P_over_time is not None and len(P_over_time)>=num_P:
    print(f"P_over_time size = {len(P_over_time)}")
elif os.path.exists(file_name):
    with open(file_name, 'rb') as file:
        P_over_time = pickle.load(file)
else:
    # File does not exist, so create the data and save it to the file
    P_over_time = construct_P_over_time(P,num_P)
    with open(file_name, 'wb') as file:
        pickle.dump(P_over_time, file)
    print(f"The file '{file_name}' has been created and the data has been saved.")
#P_over_time = construct_P_over_time(P,num_P) #1m13.8sek runtime
del num_P, A, A_w, A_w_matrix, file, file_name, G, G_w, P_w, sa
#%%
n = len(P)
x_0 = np.zeros(shape=(n,1)) #Doesn't matter
max_t = 1000
noise_mean = 0
noise_var = 1
noise_std_dev = np.sqrt(noise_var)
seed = 1
#Pre generate noise n x max_t
def generate_noise(n, max_t, mean, std_dev, seed):
    #np.random.seed(seed)
    noise = np.random.normal(loc = mean,scale = std_dev, size = (max_t, n, 1))
    # for arr in noise:
    #     for i in range(len(arr)):
    #         arr[i] = arr[i]
    noise[0] = np.zeros(shape=(n,1))
    return noise
#noise = generate_noise(n,max_t,noise_mean,noise_std_dev, seed)
def convert_seconds(seconds):
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds %= 60
    return hours, minutes, seconds
def generate_noise_from_cov(n, max_t, mean, cov, seed):
    mean_vector = np.full(n, mean)
    noise = np.empty((max_t, n))

    #np.random.seed(seed)

    # Calculate the step size to print progress at 10% intervals
    step = max_t // 50

    for t in range(max_t):
        noise[t] = np.random.multivariate_normal(mean_vector, cov)
        
        # Print progress at 10% intervals
        if (t + 1) % step == 0 or t == max_t - 1:
            progress = ((t + 1) / max_t) * 100
            total_seconds = (time.time() - start_time)
            hours, minutes, seconds = convert_seconds(total_seconds)
            print(f"Progress: {progress:.2f}% || Runtime:{hours:.0f}h{minutes:.0f}m{seconds:.0f}s.",end="\r")  # '\r' to overwrite the previous line

    
    return noise
cov = np.identity(n)
# Check if noise2 is already defined
def generate_noise_from_cov_2(n, max_t, noise_mean, cov, seed):
    try:
        noise2
    except NameError:
        # If noise2 is not defined, initialize it as None
        noise2 = None
    # Check if noise file already exists and its size is sufficient
    file_name = "noise_samples.pkl"
    if max_t > 1000:
        return
    if noise2 is not None and noise2.shape[0] >= max_t:
        print(f"Enough noise, size = {noise2.shape[0]}")
    elif os.path.exists(file_name):
        with open(file_name, 'rb') as file:
            noise2 = pickle.load(file)
        if noise2.shape[0] < max_t:
            print(f"Loaded data has shorter time horizon than max_t,\ngenerating new (more) noise.")
            noise2 = generate_noise_from_cov(n,max_t,noise_mean,cov,seed)
            with open(file_name, 'wb') as file:
                pickle.dump(noise2, file)
            print(f"The file '{file_name}' has been created and the data has been saved.")
    else:   
        # File does not exist, so create the data and save it to the file
        print(f"Starting to generate noise...")
        noise2 = generate_noise_from_cov(n, max_t, noise_mean, cov, seed)
        with open(file_name, 'wb') as file:
            pickle.dump(noise2, file)
        print(f"The file '{file_name}' has been created and the data has been saved.")
    return noise2
# noise2 = generate_noise_from_cov_2(n, max_t, noise_mean, cov, seed)
# noise = noise2
import time
def generate_mega_noise_from_cov(n, max_t, noise_mean, cov, seed):
    print("U R NOW ENTERING MEGA TERRITORY")
    num_meganoises = 20
    seeds = list(range(1,21))
    try:
        meganoise
    except NameError:
        # If meganoise is not defined, initialize it as None
        meganoise = None
    file_name = "mega_noise.pkl"
    if max_t > 1000:
        return
    if meganoise is not None and meganoise.shape[1] >= max_t and meganoise.shape[0] >= num_meganoises:
        print(f"Enough noise, size = {meganoise.shape[1]}")
    elif os.path.exists(file_name):
        with open(file_name, 'rb') as file:
            meganoise = pickle.load(file)
        if meganoise.shape[1] < max_t or meganoise.shape[0] < num_meganoises:
            print(f"Loaded data has shorter time horizon than max_t,\ngenerating new (more) MEGA noise.")
            meganoise = np.empty((num_meganoises,max_t,*(n,)))
            for noi in range(num_meganoises):
                np.random.seed(seeds[noi])
                print(f"Creating noise num: {seeds[noi]}/{num_meganoises}.")
                meganoise[noi] = generate_noise_from_cov(n,max_t,noise_mean,cov,seed)
            with open(file_name, 'wb') as file:
                pickle.dump(meganoise, file)
            print(f"The file '{file_name}' has been created and the data has been saved.")
    else:   
        # File does not exist, so create the data and save it to the file
        #print(f"Starting to generate noise...")
        meganoise = np.empty((num_meganoises,max_t,*(n,)))
        for noi in range(num_meganoises):
            np.random.seed(seeds[noi])
            print(f"Creating noise num: {seeds[noi]}/{num_meganoises}.")
            meganoise[noi] = generate_noise_from_cov(n,max_t,noise_mean,cov,seed)
        with open(file_name, 'wb') as file:
            pickle.dump(meganoise, file)
        print(f"The file '{file_name}' has been created and the data has been saved.")
    print("\nNoise generation complete.")
    return meganoise
start_time = time.time()
meganoise = generate_mega_noise_from_cov(n, max_t, noise_mean, cov, seed)
del start_time
#noise2 = generate_noise_from_cov(n, max_t, noise_mean, cov, seed)
def z_expected_power_over_time(P_over_time,max_t,noise): #Not used
    #z_power_over_time = np.empty((max_t, *(n,1)), dtype=int)
    z_expected_power_over_time = np.zeros(shape=(max_t+1,),dtype = float)
    n = len(P_over_time[0])
    internalsum = np.zeros((max_t,n))
    for t in range(max_t):
        P_t = P_over_time[t]
        temp_arr = np.matmul(np.ones(shape=(1,n)),P_t)
        for j in range(n):
            for i in range(n):
                internalsum[t][j] += (P_t[j][i])**2
            temp_val = (temp_arr)[0][j]**2
            internalsum[t][j] += -(temp_val)/n
        #print(t/10, "%")
    for T in range(max_t):
        add = 0
        if T == 0:
            var = noise[T-t]
            for j in range(n):
                add += var[j]*internalsum[t][j]
        for t in range(T):
            var = noise[T-t]
            for j in range(n):
                add += var[j]*internalsum[t][j]
        print(add)
        z_expected_power_over_time[T+1] = z_expected_power_over_time[T]+add
        #print(T/10, "%")
    return z_expected_power_over_time
# def z_expected_power_over_time_var_const(P_over_time: np.ndarray,max_t: int,var):
#     #z_power_over_time = np.empty((max_t, *(n,1)), dtype=int)
#     z_expected_power_over_time = np.zeros(shape=(max_t+1,),dtype = float)
#     n = len(P_over_time[0])
#     internalsum = np.zeros((max_t,n))
#     for t in range(max_t):
#         P_t = P_over_time[t]
#         temp_arr = np.matmul(np.ones(shape=(1,n)),P_t)
#         for j in range(n):
#             for i in range(n):
#                 internalsum[t][j] += (P_t[j][i])**2
#             temp_val = (temp_arr)[0][j]**2
#             internalsum[t][j] += -(temp_val)/n
#         #print(t/10, "%")
#     for T in range(max_t):
#         add = np.sum(internalsum[0])
#         for t in range(T):
#             add += np.sum(internalsum[t])
#         add = add*var
#         print(add)
#         z_expected_power_over_time[T+1] = z_expected_power_over_time[T]+add
#         #print(T/10, "%")
#     return z_expected_power_over_time
# def z_expected_power_over_time_var_const2(P_over_time: np.ndarray,max_t: int,var):
#     #z_power_over_time = np.empty((max_t, *(n,1)), dtype=int)
#     z_t_z_over_time = np.zeros(shape=(max_t+1,),dtype = float)
#     n = len(P_over_time[0])
#     interval = 10
#     step = max_t // (100 // interval)
#     sumtot = 0
#     for t in range(0,max_t):
#         sum1 = 0
#         sum2 = 0
#         for j in range(0,n):
#             for i in range(0,n):
#                 sum1 += P_over_time[t][i][j]**2
#         for j in range(0,n):
#             part = np.matmul(np.ones(shape=(1,n)),P_over_time[t])[0][j]
#             sum2 += part**2
#         sumtot += sum1-sum2/n
#         z_t_z_over_time[t] = sumtot
#         # Print progress at 10% intervals
#         if (t + 1) % step == 0 or t == max_t - 1:
#             progress = ((t + 1) / max_t) * 100
#             print(f"Progress: {progress:.2f}%")
#     return z_t_z_over_time
def z_expected_power_over_time_var_const3(P_over_time: np.ndarray, max_t: int, var):
    zt_z_over_time = np.zeros(shape=(max_t,), dtype=float)
    n = P_over_time.shape[1]  # Assuming P_over_time is always square (P_over_time.shape[1] == P_over_time.shape[2])
    interval = 10
    step = max_t // (100 // interval)
    sumtot = 0
    zt_z_over_time[0] = 0
    print(f"Now calculating expected power\nfor all times t given identity cov.")
    for t in range(1,max_t):
        # Calculate sum1 using vectorized operations
        sum1 = np.sum(P_over_time[t-1] ** 2) #Funkar bra

        # Calculate sum2 using vectorized operations
        sum2 = np.sum(np.matmul(np.ones((1,n)), P_over_time[t-1]) ** 2)

        # Calculate sumtot using vectorized operations
        if t == 1:
            print("n=",n)
            print(P_over_time[t])
        if t == 1 or t == 2 or t == 3:
            print("t=",t,"sum1=",sum1,"sum2=",sum2/n)
        sumtot += sum1 - sum2 / n
        zt_z_over_time[t] = sumtot

        # Print progress at 10% intervals
        if (t + 1) % step == 0 or t == max_t - 1:
            progress = ((t + 1) / max_t) * 100
            print(f"Progress: {progress:.2f}%")

    return zt_z_over_time


# z_power_over_time = z_expected_power_over_time_var_const3(P_over_time,max_t,noise_var)

def z_expected_power_over_time_var_const_sym(P,max_t,var):
    z_over_time = np.zeros(shape=(max_t,))
    for t in range(1,max_t):
        P_t = P[t]
        z_over_time[t] = z_over_time[t-1] + np.sum(P_t**2)-1
    return z_over_time

def actual_power_over_time(P_over_time, x_0, noise, max_t):
    n = len(P_over_time[0])
    P = P_over_time[1]
    x_over_time = np.empty((max_t, *(n,1)),dtype=float)
    a_over_time = np.empty((max_t,),dtype=float)
    z_over_time = np.empty((max_t, *(n,1)),dtype=float)
    actual_power_ot = np.empty((max_t,),dtype=float)

    x_over_time[0] = x_0
    a_over_time[0] = np.mean(x_over_time[0])
    z_over_time[0] = x_over_time[0] - a_over_time[0]
    for t in range(1,max_t):
        x_over_time[t] = P@x_over_time[t-1]+noise[t]
        a_over_time[t] = np.mean(x_over_time[t])
        z_over_time[t] = x_over_time[t] - a_over_time[t]
    #a_over_time2 = np.mean(np.squeeze(x_over_time),axis=1) #Works
    for t in range(0, max_t):
        #actual_power_ot[t] = np.matmul(np.transpose(z_over_time[t]),z_over_time[t])
        actual_power_ot[t] = np.sum(z_over_time[t]**2)
    return actual_power_ot, x_over_time, a_over_time, z_over_time
#noise = noise2[:,:, np.newaxis]
#actual_power_ot , x_over_time, a_over_time, z_over_time = actual_power_over_time(P_over_time, x_0, generate_noise(n,max_t,noise_mean,noise_std_dev, seed), max_t)

# %%
#t = 0:
n = len(P_over_time[0])
numerical_exp = np.empty(shape=(5,))
numerical_exp[0] = 0
numerical_exp[1] = 1221
numerical_exp[2] = np.trace(P_over_time[1]@np.transpose(P_over_time[1]))+n-1-(1/n)*np.ones(shape=(1,n))@P_over_time[1]@np.transpose(P_over_time[1])@np.ones(shape=(n,1)) #= 1633.6

for i,t in enumerate(numerical_exp):
    print(f"t={i}, NES={t:.1f}, ACT={actual_power_ot[i]}, CAL={z_power_over_time[i]}")
#%%
# def make_directed_undirected(A):
#     n = len(A)
#     for i in range(n):
#         for j in range(n):
#             if A[i][j] == 1 and A[j][i] != 1:
#                 A[j][i] = 1
#     return A
# A_symmetric = make_directed_undirected(A_w)
# A_symmetric_no_self = A_symmetric - np.diagflat(np.diag(A_symmetric))

# def Create_sym_from_upper_P(P_temp): #DOESNT CREATE DOUBLY STOCH!!
#     for i in range(len(P_temp)):
#         for j in range(i+1,len(P_temp)):
#             P_temp[j][i] = P_temp[i][j]
#     print(scipy.linalg.issymmetric(P_temp))
#     return P_temp
P_sym = Create_sym_from_upper_P(np.array(P))
P_ot_sym = construct_P_over_time(P_sym,max_t)
# %%
#P_ot = P_over_time[0:501]
noise_vec = meganoise[:,:,:, np.newaxis]
noise_vec[]
expected_var_nonsym = z_expected_power_over_time_var_const3(P_ot, max_t, noise_var)
expected_var_sym = z_expected_power_over_time_var_const_sym(P_ot_sym, max_t, noise_var)
actual_var_nonsym = actual_power_over_time(P_ot, x_0, noise_vec[0], max_t)[0]
actual_var_sym = actual_power_over_time(P_ot_sym, x_0, noise_vec[0], max_t)[0]
actual_var_nonsym_vec = np.empty(shape=(meganoise.shape[0],actual_var_nonsym.shape[0],))
for num in range(meganoise.shape[0]):
    actual_var_nonsym_vec[num] = actual_power_over_time(P_ot, x_0, noise_vec[num], max_t)[0]
actual_var_nonsym_avg = np.mean(actual_var_nonsym_vec,axis=0)

expected_var_nonsym_diff = np.zeros(shape = expected_var_nonsym.shape)
actual_var_nonsym_diff = expected_var_nonsym_diff.copy()
for n in range(len(expected_var_nonsym)-1):
    actual_var_nonsym_diff[n+1] = actual_var_nonsym_avg[n+1]-actual_var_nonsym_avg[n]
    expected_var_nonsym_diff[n+1] = expected_var_nonsym[n+1]-expected_var_nonsym[n]
#%%
#Plotting actual vs expected power over time (z(t)'z(t))
range_vec = [range(0,10),range(0,150),range(0,500)]
for i in range(3):
    x = range_vec[i]
    plt.plot(x, actual_var_nonsym_avg[0:len(x)], label = "Average power over time")
    plt.plot(x, expected_var_nonsym[0:len(x)], label = "Expected power over time")
    plt.legend()
    plt.title("Sum of square deviations of node states from\naverage node state over time")  # Adding the title
    plt.xlabel("Time")  # Adding the x-axis label (optional)
    plt.ylabel("Power, z(t)'z(t)")  # Adding the y-axis label (optional)
    plt.show()


#Plotting the same but the rate of increase instead
range_vec = [range(0,10),range(0,150),range(0,500)]
for i in range(3):
    x = range_vec[i]
    plt.plot(x,actual_var_nonsym_diff[0:len(x)], label = "Average rate of increase")
    plt.plot(x,expected_var_nonsym_diff[0:len(x)], label = "Expected rate of increase")
    plt.legend()
    plt.title("Rate of increase of power over time")
    plt.xlabel("Time")  # Adding the x-axis label (optional)
    plt.ylabel("Rate of increase of power, z(t)'z(t)")  # Adding the y-axis label (optional)
    plt.show()
avg_rate_of_increase_act= np.mean(actual_var_nonsym_diff[50:])
avg_rate_of_increase_exp= np.mean(expected_var_nonsym_diff[50:])
theoretical_rate_increase = np.mean()
# x = range(0, 100)
# plt.plot(x, actual_var_sym[0:len(x)], label = "Actual power sym")
# plt.plot(x, expected_var_sym[0:len(x)], label = "Expected power sym")
# plt.legend()
# plt.show()

# %%
