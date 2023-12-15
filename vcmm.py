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
    n = len(data)
    k = pi.shape[1]

    def pdf(dist, par, data):
        n_par = par.__len__()
        if n_par > 2:
            return dist.pdf(data, par[:-2], loc=par[-2], scale=par[-1])
        else:
            return dist.pdf(data, loc=par[-2], scale=par[-1])

    def component_prob(data, F, gamma, u, V):
        n, d = u.shape
        marginal_prob = np.column_stack([pdf(f, g, data[:, i]) for f, g, i in zip(F, gamma, range(d))])
        copula_prob = V.pdf(u)
        return marginal_prob.prod(axis=1) * copula_prob

    #Creating the latent variable z_i,j
    max_indices = np.argmax(r, axis=1)
    z = np.zeros((n, K), dtype=int)
    for i in range(n):
        max_index = max_indices[i]
        z[i][max_index] = 1

    g = np.column_stack([component_prob(data, F[i], gamma[i], u[i], V[i]) for i in range(k)])

    # Calculating llh
    llh=0
    for i in range(n):
        for j in range(k):
            llh += z[i][j]*np.log(pi[j])+z[i][j]*np.log(g[j])

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
        u: K item list of NxD np arrays of copula data
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

    candidates = [stats.norm, stats.gumbel_l, stats.cauchy, stats.gamma, stats.logistic, stats.lognorm, stats.skewnorm, stats.t, stats.skewcauchy, stats.loggamma]
    
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
            k_u.append(u_fun(candidates[idx], candidate_parameters[idx], data[:,j]))
            
        families.append(k_families)
        parameters.append(k_parameters)
        u.append(np.column_stack(k_u))
        
    return families, parameters, u

#def select_tree_structure(u, copulas, trunclevel=1):
#    return trees
  
#def select_copulas(V, u):
#    return families, parameters

def RVineStructureSelect(u, trunclevel=1):
    
    k = len(u)
    n, d = u[0].shape
    
    copula_candidates = [vcl.BicopFamily.__members__[x] for x in ["bb1","bb6","bb7","bb8","clayton","frank","gaussian","gumbel","indep","joe","student"]]
    fitctrl = vcl.FitControlsVinecop(family_set=copula_candidates, trunc_lvl=trunclevel)
    
    V = [vcl.Vinecop(data=u[i], controls=fitctrl) for i in range(k)]
    
    return V


def posterior_prob(data, pi, F, gamma, u, V):
    """
    E-step

    Input:
        data: NxD np array of data
        pi: 
        F: 
        gamma:
        V:

    Output:
        z: NxK np array of assignment probabilities
    """
    n = len(data)
    k = pi.shape[1]
    
    def pdf(dist, par, data):
        n_par = par.__len__()
        if n_par > 2 :
            return dist.pdf(data, par[:-2], loc=par[-2], scale=par[-1])
        else :
            return dist.pdf(data, loc=par[-2], scale=par[-1])
    
    def component_prob(data, F, gamma, u, V):
        n, d = u.shape
        marginal_prob = np.column_stack([pdf(f, g, data[:,i]) for f, g, i in zip(F, gamma, range(d))])
        copula_prob = V.pdf(u)
        return marginal_prob.prod(axis=1)*copula_prob
        
    g = np.column_stack([component_prob(data, F[i], gamma[i], u[i], V[i]) for i in range(k)])
    
    pig = g*pi
    
    r = pig/pig.sum(axis=1, keepdims=True)
    
    return r


def mixture_weights(r):
    """
    CM-step 1

    Input:
        r: posterior cluster probabilities

    Output:
        pi: 1xK np array of mixture weights (soft group proportions)
    """
    pi = r.mean(axis=0, keepdims=True)
    return pi
  
def marginal_parameters(r, data, F):
    """
    CM-step 2

    Input:
      r: NxK array of cluster probabilites
      data: NxD array with data on original scale
      F: KxD list with families of marginal distributions
    Output:
        parameters: KxD structure of parameters (some parameters might be vectors)
        u: K item list of NxD np arrays of copula data
    """
    n, d = data.shape
    _, k = r.shape
    
    c = r.argmax(axis=1)

    parameters = []
    u = []
    
    for i in range(k):
        
        k_data = data[c==i,:]
        k_parameters = []
        k_u = []
        
        for j in range(d):
          
            d_data = k_data[:,j]
            new_par = F[i][j].fit(d_data)
            k_parameters.append(new_par)
            k_u.append(u_fun(F[i][j], new_par, data[:,j]))
            
        parameters.append(k_parameters)
        u.append(np.column_stack(k_u))
    
    return parameters, u
  
  
def pair_copula_parameters(r, data, F, gamma, u, V):
    """
    CM-step 3

    Input:

    Output:
        theta: structure of pair copula parameters (some parameters might be vectors)
    """
    V
    return theta
  

  


def vcmm(data, K, tol=0.00001, maxiter=100, initial_method="kmeans", fitting_trunclevel=1):
  
    n, d = data.shape
    
    # 1. initial assignment
    r = initial_prob(data, K, method=initial_method)
    pi = mixture_weights(r)
    
    # 2. initial model selection
    F, gamma, u = select_marginals(data, r)
    V = RVineStructureSelect(u, trunclevel=fitting_trunclevel)
    
    # 3. parameter estimation
    t = 0
    while True :
      llh_old = llh
      r_old = r
      gamma_old = gamma
      theta_old = theta
      
      # E step
      r = posterior_prob(data, pi, F, gamma, u, V)
      
      # CM step 1
      pi = mixture_weights(r)
      
      # CM step 2
      gamma, u = marginal_parameters(r, data, F)
      
      # CM step 3
      theta = pair_copula_parameters(r, data, F, gamma, u, V)
      
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
    V = RVineStructureSelect(u, copulas, trunclevel=d-1)
    
    # 6. final cluster assignment
    c = posterior_prob(data, pi, F, gamma, u, V).argmax(axis=1)
    
    # classifier
    def predict(x):
        pred_prob = posterior_prob(data, pi, F, gamma, u, V)
        return pred_prob.argmax(axis=1), pred_prob
    
    return c, F, gamma, V, B, theta, pi, r, predict
  

### simulation study


### applications


