import numpy as np
import networkx as nx
from scipy.optimize import minimize
from scipy.optimize import linprog
import random
from itertools import combinations
import math


# V denotes set of nodes of the form [0,1,....,N]
# n_classes: number of communities
# n_unit: for example, n_unit = [60,40,20] implies there are 3 communities, 
# and number of nodes in each community is [60,40,20]
# p: homo connection probability in the planted partition model
# q: hetoro connection probability in the planted partition model
def Generating_graph_with_simplices(V, n_classes, n_unit, p, q):
    
    def connection_function(p, q, x, y, subsets):
        same_subset = False
        for subset in subsets:
            if x in subset and y in subset:
                same_subset = True
                break

        if same_subset:
            return 1 if random.random() < p else 0
        else:
            return 1 if random.random() < q else 0    

    n_V = len(V)
    possible_edges = list(combinations(V, 2))

    interval = np.cumsum(np.insert(n_unit,0,0))
    Classes = [np.arange(n_V)[interval[i]:interval[i+1]] for i in range(len(n_unit))]
    subsets = [set(Classes[i]) for i in range(n_classes)]

    # For the possible_edges, the connected and unconnected ones are mapped to 1 and 0, respectively.
    connect = [ connection_function(p, q, possible_edges[i][0], possible_edges[i][1], subsets) 
               for i in range(len(possible_edges)) ]  
    real_edges = [ possible_edges[i] for i in range(len(possible_edges)) if connect[i] == 1]
    G = nx.Graph()
    G.add_nodes_from(V)       
    G.add_edges_from(real_edges)  
    
    # Simplices
    all_cliques = list(nx.enumerate_all_cliques(G))
    Simplices = [ [] for i in range(len(all_cliques[-1])) ]
    for i in range(len(all_cliques)):
        n = len(all_cliques[i])
        Simplices[n-1].append(all_cliques[i])
    Label = []
    for i in range(n_classes):
        Label = Label + [i for k in range(n_unit[i])]
    Label_mat = np.concatenate([np.stack([np.eye(n_classes)[i] for j in range(n_unit[i])]) 
                                for i in range(n_classes)]) 
    
    return G, Simplices, Classes, Label, Label_mat   

def Adjacency_matrix(G):
    
    n = len(G)
    adj_matrix = np.zeros((n, n))
    
    for edge in G.edges():
        u, v = edge
        adj_matrix[u][v] = 1
        if not G.is_directed():
            adj_matrix[v][u] = 1
            
    return adj_matrix
    
def Initialization(G, Classes, BP):
       
    n_classes = len(Classes)
    V = list(G.nodes())
    adj_matrix = Adjacency_matrix(G)
    n_classes = len(Classes)
    
    # prior known nodes
    Sets = [np.random.choice(Classes[i], size = math.ceil(len(Classes[i])*BP), replace=False) 
            for i in range(n_classes)]
    
    N = len(adj_matrix)
    degree_matrix = np.diag(np.sum(adj_matrix, axis=1))
    
    # Laplacian_matrix
    L = degree_matrix - adj_matrix
    
    Boundary_union = set()
    for Set in Sets:
        Boundary_union = Boundary_union.union(set(Set))
    F = list(np.sort(list(set(V) - Boundary_union)))

    def LP(A):
        
        # A is a part of L
        # L : F x F matrix, hence A: F x F matrix
        n = A.shape[0]

        c = np.zeros(n+1)
        c[-1] = 1

        # constraint AX <= a
        A_ub = np.hstack((A, -np.ones((n, 1))))

        # upper bound
        b_ub = np.zeros(n)

        # x1 + x2 + ... + xn = 1
        A_eq = np.zeros((1, n+1))
        A_eq[0, :n] = 1

        # b_eq = 1
        b_eq = np.array([1])

        # 0 <= x1, x2, ..., xn <= 1 및 0 <= a (non-negative)
        bounds = [(0, 1) for _ in range(n)] + [(0, None)]

        # linear programming
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

        if result.success:
            X = result.x[:-1]
            a_min = result.x[-1]
            return X / a_min
        else:
            raise ValueError("Problem occurs")    

    def EM(F, L):
        # Equilibrium measure of F
        # |V| = n
        # |F| = r < n
        # Remark that the support of EM is F
        A = L[F][:, F]
        
        # P_measure: probability measure
        EM_tem = LP(A)
        
        def Complement(vec):

            result = np.zeros(len(L))
            for i, value in enumerate(F):
                result[value] = vec[i]
            return result
        
        return Complement(EM_tem)   

    def Add_node(F, node):

        position = np.searchsorted(np.array(F), node)
        return list(np.insert(np.array(F), position, node))
    
    v_F = EM(F, L)
    
    Prob = []
    for k in range(len(Sets)):       
        
        v_F_U_x = []
        Set = Sets[k]
        for i in range(len(Sets[k])):
            s_i = Set[i]
            F_U_s_i = Add_node(F, Set[i])
            v_F_U_x.append(EM(F_U_s_i, L))
         
        Prob.append(sum([ (v_F_U_x[i] - v_F)/v_F_U_x[i][Set[i]] for i in range(len(Set)) ]))
    
    # n_V x n_classes matrix, each row sum is 1
    probability_matrix = np.stack(Prob).T 
    initial_data = probability_matrix.flatten()
    classification_result = np.array([ np.argmax(probability_matrix[i]) 
                                      for i in range(len(probability_matrix)) ])
    # Known information
    Known = []
    for i in range(len(Sets)):
        Known = Known + list(Sets[i])
    Known = list(np.sort(Known))
    x_known_indices = []
    for i in range(len(Known)):
        x_known_indices = x_known_indices + [ n_classes*Known[i] + k for k in range(n_classes) ]
    
    x_known_values = [ initial_data[x_known_indices[i]] for i in range(len(x_known_indices)) ]
    x_known = np.stack([x_known_indices, x_known_values]).T
    
    return initial_data.reshape(-1, n_classes), classification_result, x_known 

def Multiply(vec):
    ans = 1
    for n in vec:
        if n == 0:
            return 0
        ans *= n
    return ans

# Fac(5) = 5*4*3*2*1, factorial function
def Fac(integer):
    return np.prod(np.arange(1,integer+1))

def Basis_mat(Simplices, n_classes):
    HS = len(Simplices[-1][0])
    basis = [np.eye(n_classes)[i].reshape(-1,1) for i in range(n_classes)]
    mat = [ basis ] +  [ [] for i in range(HS-1) ]
    for k in range(HS-1):
        for j in range(n_classes):
            for i in range(n_classes**(k+1)):
                mat[k+1].append(np.concatenate([basis[j], mat[k][i]], axis=1)) 
    return mat, HS

# x is the variable of function
# HOI = 0 means the algorithm only use pairwise interactions between nodes
# HOI = 1 means the algorithm use higher order interaction as well as pairwise interactions
def Objective(x, x_known, Simplices, n_classes, HOI, exp_factor):
    
    if HOI == 0:
        X = x.copy()
        for i in range(len(x_known)):
            X = np.insert(X,int(x_known[i][0]),x_known[i][1])
        basis = [np.eye(n_classes)[i] for i in range(n_classes)]
        mat = []
        for i in range(n_classes):
            for j in range(n_classes):
                tem = np.stack([basis[i], basis[j]]).T
                mat.append(tem)      
        simplex = np.array(Simplices[1])
        coef = np.array([ Fac(2) / np.prod([Fac(mat[j].sum(1)[i]) for i in range(n_classes)]) 
                         for j in range(len(mat))])
        #coef_sum = coef.sum()
        error = []
        for i in range(len(simplex)):
            # In this case, nodes is of the form [a,b], hence len(nodes) = 2
            nodes = simplex[i]
            P_tem = [ [X[n_classes * nodes[j] + t] for j in range(len(nodes))] for t in range(n_classes) ]
            prob_mat = np.stack(P_tem)
            prob = np.array([Multiply((prob_mat*mat[j]).sum(0)) for j in range(len(mat))])
            total_prob = (prob * coef).sum()
            error.append(total_prob)

        ERROR = np.array(error).sum()
        
    if HOI == 1:
        X = x.copy()
        for i in range(len(x_known)):
            X = np.insert(X,int(x_known[i][0]),x_known[i][1])

        mat, HS = Basis_mat(Simplices, n_classes)
        total_error = []
        total_coef_sum = []
        len_simplex = []
        for k in range(1, HS):
            simplex = np.array(Simplices[k]).astype(int)
            len_simplex.append(len(simplex))
            coef = np.array([ Fac(k+1) / np.prod([Fac(mat[k][j].sum(1)[i]) for i in range(n_classes)])
                              for j in range(len(mat[k]))])
            coef_sum = coef.sum()
            total_coef_sum.append(coef_sum)
            error = []
            for i in range(len(simplex)):
                nodes = simplex[i]
                P_tem = [ [X[n_classes * nodes[j] + t] for j in range(len(nodes))] for t in range(n_classes) ]
                prob_mat = np.stack(P_tem)
                prob = np.array([Multiply((prob_mat*mat[k][j]).sum(0)) for j in range(len(mat[k]))])
                total_prob = (prob * coef).sum()
                error.append(total_prob)    
            total_error.append(sum(error)) 
   
        new_error = np.array(total_error) / np.array(total_coef_sum) / np.array(len_simplex)
        weight1 = 1**(np.arange(1,HS)-1)
        ERROR = ( weight1 * np.array(new_error) ).sum() 
        
    return ERROR

def Optimization(x_init, x_known, Simplices, n_classes, HOI, exp_factor):
    n_V = len(Simplices[0])

    # constraints
    constraints = []
    for i in range(n_V-int(len(x_known)/n_classes)):
        con = {"type": "eq", "fun": lambda x, i=i: sum([x[n_classes*i + k] for k in range(n_classes)]) - 1}
        constraints.append(con)
    bounds = [(0, 1)] * (n_classes * n_V - len(x_known))

    # optimization
    result = minimize(lambda x: Objective(x, x_known, Simplices, n_classes, HOI, exp_factor), 
                      x_init, bounds=bounds, constraints=constraints, method='SLSQP')
    x_val1 = result.x
    x_val2 = x_val1.copy()
    for i in range(len(x_known)):
        x_val2 = np.insert(x_val2,int(x_known[i][0]),x_known[i][1])
    result_mat = x_val2.reshape(-1,n_classes)
    prediction = np.array([np.argmax(result_mat[i]) for i in range(n_V)])
    return result_mat, prediction

def confusion_matrix(labels, predictions, n_classes):
    matrix = np.zeros((n_classes, n_classes), dtype=np.int64)

    for true_label, pred_label in zip(labels, predictions):
        matrix[true_label][pred_label] += 1

    return matrix

def precision_recall_f1_accuracy(conf_matrix):
    num_classes = conf_matrix.shape[0]
    true_positives = np.diag(conf_matrix)
    false_positives = np.sum(conf_matrix, axis=0) - true_positives
    false_negatives = np.sum(conf_matrix, axis=1) - true_positives
    
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1_score = 2 * (precision * recall) / (precision + recall)
    
    accuracy = np.sum(true_positives) / np.sum(conf_matrix)
    
    precision = list(precision)
    recall = list(recall)
    f1_score = list(f1_score)
    accuracy = [ accuracy ]
    return precision, recall, f1_score, accuracy

def Reconstruction(G):
    V = [ i for i in range(len(G)) ]
    n_V = len(V)
    label_set = sorted(list(set([G.nodes[node]['value'] for node in G.nodes])))
    n_classes = len(label_set)
    # make labels to 0,1,2,...
    for node in G.nodes:
        for i in range(n_classes):
            if G.nodes[node]['value'] == label_set[i]:
                G.nodes[node]['value'] = i

    # name nodes to 0,1,2,..,len(node)          
    next_node_id = 0
    mapping = {}            
    for i in range(n_classes):
        for node in G.nodes:
            if G.nodes[node]['value'] == i:
                mapping[node] = next_node_id
                next_node_id += 1
    G = nx.relabel_nodes(G, mapping)   
    Classes = [ [node for node in G.nodes if G.nodes[node]['value'] == i] for i in range(n_classes) ]
    n_counts = [ len(Classes[i]) for i in range(n_classes) ]

    # Simplices
    all_cliques = list(nx.enumerate_all_cliques(G))
    Simplices = [ [] for i in range(len(all_cliques[-1])) ]
    for i in range(len(all_cliques)):
        n = len(all_cliques[i])
        Simplices[n-1].append(all_cliques[i])
    Label = []
    for i in range(n_classes):
        Label = Label + [i for k in range(n_counts[i])]
    Label_mat = np.concatenate([np.stack([np.eye(n_classes)[i] for j in range(n_counts[i])]) 
                                for i in range(n_classes)]) 
    return G, Simplices, Classes, Label, Label_mat

