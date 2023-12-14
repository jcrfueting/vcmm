###
###   Vine Copula Mixture Models
###

import numpy as np
import pyvinecopulib as vcl
import scipy.stats as stats

### data for development and testing (currently only 2 dimensional)

dfs = [np.loadtxt("project/EMGaussian."+t_flag) for t_flag in ["train", "test"]]

# empirial pseudo copulas
copula_dfs = [vcl.to_pseudo_obs(x) for x in dfs]

### functions

def loglike(data, F, gamma, V, pi, r):
    """
    log-likelihood of vine copula mixture
    """
    llh = 0
    return llh

def initial_prob(data, K, method):
    """
    initial cluster assignment probabilities in VCMM algorithm

    Input:
        data: NxD np array of data
        K: number of groups
        method: string specifying the fast clustering method

    Output:
        z: NxK np array of assignment probabilities
    """
    if method == 'kmeans':
        n, d = data.shape

        centroids_indices = np.random.choice(n, size=K, replace=False)
        centroids = data[centroids_indices]

        # Assign each point to centroid
        z = np.zeros((n, K))
        for i in range(n):
            distances = np.sum((centroids - data[i])**2, axis=1)
            centroid = np.argmin(distances)
            z[i][centroid] = 1

        return z / np.sum(z, axis=1, keepdims=True)  # Normalize

    # More methods can be added below
    else:
        raise ValueError("Unknown method. Please specify a valid clustering method (e.g., 'kmeans').")
  
def select_marginals(data, r):
    """
    initial selection of marginal distributions and parameters
    
    Input:
        data: NxD np array of data
        r: posterior cluster probability
        
    Output:
        families: KxD collection of scipy distributions
        parameters: KxD collection of parameters
        u: KxNxD np array of copula data
    """
    
    n, d = data.shape
    _, k = r.shape
    
    c = r.argmax(axis=1)
    
    def bic(dist, par, data):
        n = data.shape
        n_par = par.__len__()
        
        if n_par > 2 :
            llh = np.sum(dist.logpdf(data, par[:-2], loc=par[-2], scale=par[-1]))
        else :
            llh = np.sum(dist.logpdf(data, loc=par[-2], scale=par[-1]))
        
        return -2*llh-n_par*np.log(n)
      
    def u_fun(dist, par, data):
        n_par = par.__len__()
        if n_par > 2 :
            return dist.cdf(data, par[:-2], loc=par[-2], scale=par[-1])
        else :
            return dist.cdf(data, loc=par[-2], scale=par[-1])

    # TODO: go through stats for set of feasable candidates dists
    candidates = [stats.norm, stats.gumbel_l]
    
    families = []
    parameters = []
    u = []
    
    for i in range(k):
        
        k_data = data[c==i,:]
        
        k_families = []
        k_parameters = []
        k_u = []
        
        for j in range(d):
          
            d_data = k_data[:,j]
            
            candidate_parameters = [dist.fit(d_data) for dist in candidates]
            BICs = [bic(cand, param, d_data) for cand, param in zip(candidates, candidate_parameters)]
            idx = np.argmin(BICs)
            
            k_families.append(candidates[idx])
            k_parameters.append(candidate_parameters[idx])
            k_u = u_fun(candidates[idx], candidate_parameters[idx], data[:,j])
            
        families.append(k_families)
        parameters.append(k_parameters)
        u.append(k_u)
        
    return families, parameters, u

#def select_tree_structure(u, copulas, trunclevel=1):
#    return trees
  
#def select_copulas(V, u):
#    return families, parameters

def RVineStructureSelect(u, trunclevel=1):
    vinecop = vcl.Vinecop()
    V = vinecop.select(u, trunclevel=trunclevel)
    return V, V.families, V.parameters

def posterior_prob(data, pi, chi, psi):
    """
    E-step

    Input:
        data: NxD np array of data
        pi: 
        chi: 
        psi

    Output:
        z: NxK np array of assignment probabilities
    """
    return z

def mixture_weights(r):
    """
    CM-step 1

    Input:
        r: posterior cluster probabilities

    Output:
        pi: 1xK np array of mixture weights (soft group proportions)
    """
    pi = r.sum(axis=0, keepdims=True)
    return pi
  
def marginal_parameters(r, data, F):
    """
    CM-step 2

    Input:
      r: NxK array of cluster probabilites
      data: NxD array with data on original scale
      F: KxD array with families of marginal distributions
    Output:
        gamma: KxD structure of parameters (some parameters might be vectors)
    """
    return gamma
  
def pair_copula_parameters(r, data, F, gamma, V):
    """
    CM-step 3

    Input:

    Output:
        theta: structure of pair copula parameters (some parameters might be vectors)
    """
    V
    return theta
  

  


def vcmm(data, K, tol=0.00001, maxiter=100, initial_method, fitting_trunclevel=1):
  
    n, d = data.shape
    
    # 1. initial assignment
    r = initial_prob(data, K, method=initial_method)
    
    # 2. initial model selection
    F, gamma, u = select_marginals(data, r)
    V, families, parameters = RVineStructureSelect(u, trunclevel=truncation_level)
    
    # 3. parameter estimation
    t = 0
    while True :
      llh_old = llh
      r_old = r
      gamma_old = gamma
      theta_old = theta
      
      # E step
      r = posterior_prob()
      
      # CM step 1
      pi = mixture_weights(r)
      
      # CM step 2
      gamma = marginal_parameters(r, data, F)
      
      # CM step 3
      theta = pair_copula_parameters(r, data, F, gamma, V)
      
      # stopping condition(s)
      llh = loglik(data, r_old, gamma_old, theta_old)
      llhdiff = (llh-llh_old)/llh_old
      if llhdiff < tol :
          break
      if t >= maxiter :
          break
        
    # 4. temporary cluster assignment
    c = r.argmax(axis=1)
    
    # 5. final model selection
    F, gamma, u = select_marginals(data, r)
    V, families, parameters = RVineStructureSelect(u, copulas, trunclevel=d-1)
    
    # 6. final cluster assignment
    c = posterior_prob().argmax(axis=1)
    
    # classifier
    def predict(x):
        pred_prob = posterior_prob()
        return pred_prob.argmax(axis=1), pred_prob
    
    return c, F, gamma, V, B, theta, pi, r, predict
  

### simulation study


### applications


