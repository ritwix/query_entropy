# Script to plot optimal node to query which minimizes the conditional entropy
# of the final cascade size
# Author: RM, Date: 06/14/2023

import numpy as np
import matplotlib.pyplot as plt
import pdb


def finite_sum(z, n):
    '''
    Find sum of the finite series z + 2z^2 + 3z^3 + ... + nz^n
    '''
    numer = z*(1-(n+1)*z**n + n*z**(n+1))
    denom = (1-z)**2
    return numer/denom

def cond_entropy_cascade_size(p, k, N):
    '''
    Find conditional entropy of cascade size Y given node k 
    is queried in a path graph of N nodes.
    '''
    H_Y_given_X_takes_0 = -(1-p**(k-1))*np.log(1-p) - (1-p)*np.log(p)*finite_sum(p,k-2)
    H_Y_given_X_takes_1 = -(p**(k-1) - p**(N-1))*np.log(1-p) - (1-p)*np.log(p)*(finite_sum(p, N-2) - finite_sum(p, k-2)) \
        -N*p**N*np.log(p)
    
    H_Y_given_X = (1-p**(k-1))*H_Y_given_X_takes_0 + p**(k-1)*H_Y_given_X_takes_1
    
    return H_Y_given_X

def optimal_k_cond_entropy(p, N):
    k_opt = -1
    k = 2
    min_cond_entropy = float(1e7)
    while k < N:
        cond_entropy_k = cond_entropy_cascade_size(p, k, N)
        if cond_entropy_k < min_cond_entropy:
            k_opt = k
            min_cond_entropy = cond_entropy_k
        k+=1
    
    return k_opt, min_cond_entropy 
    

def main():
    N_list = [5, 10, 20, 50, 100, 500]
    for N in N_list:
        k_opt_list = []
        min_cond_entropy_list = []
        p_list = np.arange(0.05, 1, 0.05).tolist()
        for p in p_list:
            k_opt, min_cond_entropy = optimal_k_cond_entropy(p, N)
            k_opt_list.append(k_opt)
            min_cond_entropy_list.append(min_cond_entropy)
            
        fig, ax = plt.subplots(figsize=(8,8))
        ax.plot(p_list, k_opt_list, marker='s')
        for i in range(len(p_list)):
            ax.annotate(round(min_cond_entropy_list[i],2), (p_list[i], k_opt_list[i]), \
                xytext=(5,3),textcoords = "offset pixels")
        ax.set_xlabel('Transmission rate p')
        ax.set_ylabel('Optimal node k to query')
        ax.set_title('Path graph analysis')
        fig.savefig(f'path_graph_k_vs_p_N{N}.png', dpi=300, bbox_inches='tight')

if __name__=="__main__":
    main()
