import numpy as np
import networkx as nx
import pandas as pd
from scipy.optimize import linprog
import math
from math import comb
import random
import itertools
from itertools import combinations, combinations_with_replacement
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch_geometric.transforms as T



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def generating_graph_with_simplices(n_units, prob_mat2):
    # n_units: node distribution over clusters
    # prob_mat2: B_2 in the manuscript
    
    # Create nodes
    V = [i for i in range(sum(n_units))]
    
    def connection_function(x, y, classes, prob_mat):
        # Find the classes to which x and y belong
        for i in range(len(classes)):
            if x in classes[i]:
                class_x = i
            if y in classes[i]:
                class_y = i
                
        # Check the connection probability between the classes in prob_mat
        return 1 if random.random() < prob_mat[class_x][class_y] else 0
    
    n_V = len(V)
    possible_edges = list(combinations(V, 2))

    # Generate indices for each class
    interval = np.cumsum(np.insert(n_units, 0, 0))
    classes = [set(np.arange(n_V)[interval[i]:interval[i+1]]) for i in range(len(n_units))]
    
    # Determine connections
    connect = [connection_function(possible_edges[i][0], possible_edges[i][1], classes, prob_mat2)
               for i in range(len(possible_edges))]
    
    real_edges = [possible_edges[i] for i in range(len(possible_edges)) if connect[i] == 1]
    
    # Create the graph
    G = nx.Graph()
    G.add_nodes_from(V)
    G.add_edges_from(real_edges)

    # Ensure graph connectivity
    if not nx.is_connected(G):
        connected_components = list(nx.connected_components(G))
        # Connect unconnected components to a random node
        for component in connected_components[1:]:
            random_class = random.choice(list(classes))
            connect_node = np.random.choice(list(random_class))
            G.add_edge(connect_node, list(component)[0])
    
    # Assign labels and add node attributes
    label = {}
    for idx, class_set in enumerate(classes):
        for node in class_set:
            label[node] = idx
    nx.set_node_attributes(G, label, 'label')
    
    # Generate simplices
    all_cliques = list(nx.enumerate_all_cliques(G))
    simplices = [[] for _ in range(len(all_cliques[-1]))]
    for clique in all_cliques:
        n = len(clique)
        simplices[n-1].append(clique)
    
    return G, simplices, classes, list(label.values())




# Generates k-cliques based on the probabilities assigned to each cluster in the probability matrix (prob_mat).
# After generating the cliques, they should be added to the original graph G.
def k_cliques_generation(N, k, prob_mat):
    # N = [N_1, N_2, ..., N_n_L], number of nodes for each cluster.
    # k: k-clique.
    # prob_mat: n_L x n_L matrix, its i, j component implies the connection probability of nodes between cluster i and j
    

    # For a given node distribution, calculate 1. number of nodes for each cluster, 2. the expected number of k-cliques in this case.
    def n_expected_cliques(N, clique_node_dist, prob_mat):
    
        # Count the number of nodes in each cluster
        dist_counts = [clique_node_dist.count(i) for i in range(len(N))]
        # Calculate combinations
        # The dtype is specified to avoid errors in np.prod when handling large integers
        combinations = np.prod([comb(N[i], dist_counts[i]) for i in range(len(N)) if dist_counts[i] > 0], dtype=np.float64)
        # Calculate the connection probability between nodes within the same cluster
        internal_edges = sum(comb(count, 2) for count in dist_counts if count > 1)
        internal_prob = np.prod([prob_mat[i, i]**comb(dist_counts[i], 2) for i in range(len(N)) if dist_counts[i] > 1])
    
        # Calculate the connection probability between nodes in different clusters
        external_prob = 1
        for i in range(len(N)):
            for j in range(i + 1, len(N)):  # Ensure i < j to exclude cases where i and j are the same
                if dist_counts[i] > 0 and dist_counts[j] > 0:
                    external_prob *= prob_mat[i, j]**(dist_counts[i] * dist_counts[j])
    
        # Calculate the final probability
        n_cliques = int(combinations * internal_prob * external_prob)
        return dist_counts, n_cliques

    # Using the number of nodes for each cluster and the expected number of k-cliques, generate k-cliques.
    def generate_unique_cliques(N, dist_counts, n_cliques):
        total_nodes = sum(N)
        node_pools = []
        current_index = 0
    
        # Create a pool of nodes for each cluster
        for count in N:
            node_pools.append(list(range(current_index, current_index + count)))
            current_index += count
    
        # Generate all possible cliques
        all_cliques = set()
        
        while len(all_cliques) < n_cliques:
            clique = []
            for i, count in enumerate(dist_counts):
                if count > 0:
                    clique.extend(random.sample(node_pools[i], count))
            all_cliques.add(tuple(sorted(clique)))  # Sort cliques to avoid duplicates
            
        return list(all_cliques)

    n_L = len(N)  # Number of labels or clusters
    comb_with_rep = list(combinations_with_replacement(range(n_L), k))  # Generate all possible combinations
    n_combs = len(comb_with_rep)  # nHr, n = n_L, r = k    
    
    # Generate k-cliques for all possible cases
    total_cliques = []
    for i in range(n_combs):
        clique_distribution, n_cliques = n_expected_cliques(N, comb_with_rep[i], prob_mat)
        total_cliques.append(generate_unique_cliques(N, clique_distribution, n_cliques))
    total_cliques = sum(total_cliques, [])

    return total_cliques




# Converts the generated higher-dimensional structures (cliques) into edges to add them to the graph.
def to_edges(cliques):
    result_list = []
    for clique in cliques:
        combinations = list(itertools.combinations(clique, 2))




# Generate edges up to 5-cliques
def generation_upto_5cliques(prob_mat2, prob_mat3, prob_mat4, prob_mat5, n_units):

    # prob_mat2, 3, 4, and 5 correspond to B_2, 3, 4, and 5 in the manuscript
    # n_units: node distribution over clusters
    
    V = [i for i in range(sum(n_units))]
    n_L = len(n_units)

    G, simplices, classes, labels = generating_graph_with_simplices(n_units, prob_mat2)
    
    clique_size = 3
    three_cliques = k_cliques_generation(n_units, clique_size, prob_mat3)
    G.add_edges_from(to_edges(three_cliques))
    
    clique_size = 4
    four_cliques = k_cliques_generation(n_units, clique_size, prob_mat4)
    G.add_edges_from(to_edges(four_cliques))
    
    clique_size = 5
    five_cliques = k_cliques_generation(n_units, clique_size, prob_mat5)
    G.add_edges_from(to_edges(five_cliques))
    
    all_cliques = list(nx.enumerate_all_cliques(G))
    simplices = [[] for i in range(len(all_cliques[-1]))]
    for i in range(len(all_cliques)):
        n = len(all_cliques[i])
        simplices[n-1].append(all_cliques[i])

    return G, simplices, classes, labels


# adjacency_matrix for the network.
def adjacency_matrix(G):
    return nx.adjacency_matrix(G).toarray()


# Solves a linear programming problem using the given matrix A and returns the result.
def linear_programming(A, method='highs'):
   
    # A : numpy.ndarray, submatrix of matrix L with dimensions F x F.
    # method : The method used to solve the linear programming problem. The default is 'highs'.
    # Returns: numpy.ndarray, The solution vector X resulting from the linear programming problem.


    n = A.shape[0]  # Size n of matrix A

    # Objective function: minimize the last variable a
    c = np.zeros(n+1)
    c[-1] = 1

    # Inequality constraints AX <= a
    A_ub = np.hstack((A, -np.ones((n, 1))))
    b_ub = np.zeros(n)

    # Equality constraint: sum of X = 1
    A_eq = np.ones((1, n+1))
    A_eq[0, -1] = 0  # No constraint on the last variable a
    b_eq = np.array([1])

    # Variable bounds: 0 <= x_i <= 1 for all i, 0 <= a
    bounds = [(0, 1)] * n + [(0, None)]

    # Solve the linear programming problem
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method=method)
    
    if result.success:
        # If a solution exists, normalize the X vector using the optimized variable a
        a_min = result.x[-1]
        X = result.x[:-1] / a_min
        return X
    else:
        raise ValueError("unable to solve")




def equilibrium_measure(F, L):
    # Calculate the Equilibrium measure for the given set F
    # L is the Graph Laplacian matrix, and F is a list.
    # The entire node set V consists of n elements,
    # where F consists of r (< n) elements.
    # The result is a vector of length r, where all elements except those corresponding to F are 0 (the support of EM is F).
    A = L[F][:, F]
    # P_measure: probability measure
    EM_tem = linear_programming(A)
    result = np.zeros(len(L))
    result[F] = EM_tem
    return result




def add_node(F, node):
    # Add a node to the given list-based set F, maintaining the correct order
    # Since node is an integer, it carries information about its position
    # Find the position where the node should be inserted
    position = np.searchsorted(np.array(F), node)
    return list(np.insert(np.array(F), position, node))



def encode_integer(i, classes):
    # For example, if node 3 belongs to label 1 (out of 3 labels), return [3, 1, 0, 0]
    # For example, if node 60 belongs to label 2 (out of 5 labels), return [60, 0, 1, 0, 0, 0]
    
    # Initialize the result array
    # encoded = np.zeros(6, dtype=int)
    encoded = np.full(len(classes) + 1, 0, dtype=float)

    encoded[0] = i  # The first element is the integer itself

    # Iterate through each sublist in classes to check if it contains i
    for index, class_list in enumerate(classes):
        if i in class_list:
            encoded[index + 1] = 1  # Set 1 at the position corresponding to the class
            break  # Exit the loop since i has been found

    return encoded



def known_sets(classes, BP):
    n_L = len(classes)
    return [np.random.choice(list(classes[i]), size=math.ceil(len(classes[i])*BP), replace=False) for i in range(n_L)]



def initialization(G, classes, simplices, BP):
    
    # n is |V|=N, and Sets[i] is the set of nodes that are already known to belong to the i-th class.
    # Nodes should be integers like [0,1,..,N-1]!
    # A, B, C are composed as lists    
    n_L = len(classes)
    V = list(G.nodes())
    n_V = len(V)
    adj_matrix = adjacency_matrix(G)

    # Extract known nodes randomly
    Sets = known_sets(classes, BP)
    degree_matrix = np.diag(np.sum(adj_matrix, axis=1))
    # Laplacian matrix
    L = degree_matrix - adj_matrix

    # Set of known nodes
    Boundary_union = {element for subset in Sets for element in subset}
    # List of unknown nodes
    F = list(np.sort(list(set(V) - Boundary_union)))

    # Compute v^F
    v_F = equilibrium_measure(F, L)
    
    # Iterate over each set in Sets
    ###############################################
    # This section is time-consuming
    Prob = []
    for k in range(len(Sets)):       
        
        v_F_U_x = []
        Set = Sets[k]
        for i in range(len(Sets[k])):
            s_i = Set[i]
            F_U_s_i = add_node(F, Set[i])
            v_F_U_x.append(equilibrium_measure(F_U_s_i, L))
         
        Prob.append(sum([(v_F_U_x[i] - v_F)/v_F_U_x[i][Set[i]] for i in range(len(Set))]))
    ###############################################
    
    probability_matrix = np.stack(Prob).T  # n_V x n_L matrix, each row sum is 1
    classification_result = np.array([np.argmax(probability_matrix[i]) 
                                      for i in range(len(probability_matrix))])
    # Sorted list of known nodes
    known = sorted(Boundary_union)
    x_known = np.array([encode_integer(i, classes) for i in known])
    
    return probability_matrix, classification_result, x_known





def fac(integer):
    # Calculate factorial. No special GPU operations are needed when using PyTorch.
    return torch.tensor([math.factorial(integer)], dtype=torch.float32)





def basis_mat(n_max, n_classes):
    basis = [torch.eye(n_classes)[i].unsqueeze(-1) for i in range(n_classes)]
    mat = [basis] + [[] for _ in range(n_max - 1)]
    for k in range(n_max - 1):
        for j in range(n_classes):
            for i in range(n_classes ** (k + 1)):
                mat[k + 1].append(torch.cat([basis[j], mat[k][i]], dim=1))
    return mat




def prob_product(vectors):
    result = vectors[0]
    for vector in vectors[1:]:
        result = torch.ger(result, vector).flatten()
    return result




def generalized_outer_product(P, index_lists):
    results = []
    for indices in index_lists:
        selected_vectors = [P[idx] for idx in indices]
        result = prob_product(selected_vectors)
        results.append(result)
    result_matrix = torch.stack(results)
    return result_matrix




def objective(P, simplices, n_L, exp_base, device):
    """
    P: shape = [n_V, n_L] # probability distribution for each node
    """
    n_max = len(simplices)
    prob_prod_set = [generalized_outer_product(P, simplices[i]) for i in range(n_max)]
    mat = basis_mat(n_max, n_L)   
    
    coef = [0]
    for k in range(1, n_max):
        cvals = []
        for j in range(len(mat[k])): 
            
            row_sums = mat[k][j].sum(1)
            row_fac = torch.prod(torch.tensor([fac(int(row_sums[i])) for i in range(n_L)]))
            cvals.append(fac(k) / row_fac)
        coef.append(torch.tensor(cvals, device=device))
    
    
    clique_weight = torch.tensor([exp_base ** i for i in range(n_max)], device=device)
    
    multi_coef_applied_prob = sum([clique_weight[i] * (coef[i] * prob_prod_set[i]).sum() for i in range(1, n_max)])
    
    return multi_coef_applied_prob




class Model(nn.Module):
    def __init__(self, device, initial_data, x_known, exp_base):
        super(Model, self).__init__()
        self.device = device
        self.exp_base = exp_base

        # Ensure initial_data is a torch.Tensor and move it to the correct device
        if not isinstance(initial_data, torch.Tensor):
            initial_data = torch.tensor(initial_data, dtype=torch.float32, device=device)
        else:
            initial_data = initial_data.to(device)

        self.n_V, self.n_L = initial_data.shape
        
        # Extract indices from x_known
        self.fixed_indices = x_known[:, 0].astype(int)
        fixed_values = torch.tensor(x_known[:, 1:], dtype=torch.float32, device=device)
        
        # Store the non-trainable part as a regular tensor
        self.fixed_params = fixed_values
        
        # Set up the trainable parameters
        mask = torch.ones(self.n_V, dtype=torch.bool, device=device)
        mask[self.fixed_indices] = False
        self.trainable_params = nn.Parameter(initial_data[mask])

    def forward(self, simplices):
        # Restore the full data
        full_data = torch.empty((self.n_V, self.n_L), device=self.device)
        full_data[self.fixed_indices] = self.fixed_params
        mask = torch.ones(self.n_V, dtype=torch.bool, device=self.device)
        mask[self.fixed_indices] = False
        all_indices = torch.arange(self.n_V, device=self.device)
        trainable_indices = all_indices[mask]
        full_data[trainable_indices] = self.trainable_params
        
        # Apply Softmax (example applying to the entire data)
        soft_P = F.softmax(full_data, dim=1)
        
        return objective(soft_P, simplices, self.n_L, self.exp_base, self.device)




def training(epochs, device, simplices, initial_data, x_known, lr, exp_base):
    model = Model(device, initial_data, x_known, exp_base).to(device)
    optimizer = optim.Adam([model.trainable_params], lr=lr)

    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = model(simplices)
        loss.backward()
        optimizer.step()

        if epoch % (epochs // 5) == 0:
            print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

    # Data restoration process
    final_data = torch.empty((model.n_V, model.n_L), device=device)
    final_data[model.fixed_indices] = model.fixed_params
    mask = torch.ones(model.n_V, dtype=torch.bool, device=device)
    mask[model.fixed_indices] = False
    all_indices = torch.arange(model.n_V, device=device)
    trainable_indices = all_indices[mask]
    final_data[trainable_indices] = model.trainable_params.detach()
    
    # Apply Softmax only to trainable indices
    softmax_data = torch.empty_like(final_data)  # Create an empty tensor with the same shape as final_data
    softmax_data[model.fixed_indices] = final_data[model.fixed_indices]  # Copy the fixed part as is
    softmax_data[trainable_indices] = F.softmax(final_data[trainable_indices], dim=1)  # Apply softmax only to the trainable part
    
    final_P = softmax_data.cpu().numpy()  # Convert the tensor to a NumPy array
    pred = np.argmax(final_P, axis=1)
    return final_P, pred


