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

## helpers

def pdf(dist, par, data):
    """
    pdf of a scipy distribution object
    """
    n_par = par.__len__()
    if n_par > 2:
        return dist.pdf(data, par[:-2], loc=par[-2], scale=par[-1])
    else:
        return dist.pdf(data, loc=par[-2], scale=par[-1])
      
def logpdf(dist, par, data):
    """
    logpdf of a scipy distribution object
    """
    n_par = par.__len__()
    if n_par > 2:
        return dist.logpdf(data, par[:-2], loc=par[-2], scale=par[-1])
    else:
        return dist.logpdf(data, loc=par[-2], scale=par[-1])

def component_prob(data, F, gamma, u, V):
    """
    density of a singular VCMM component
    """
    n, d = u.shape
    marginal_prob = np.column_stack([pdf(f, g, data[:, i]) for f, g, i in zip(F, gamma, range(d))])
    copula_prob = V.pdf(u)
    return marginal_prob.prod(axis=1) * copula_prob
  
def bic(dist, par, data):
    """
    Bayesian Information Criteion of a scipy distribution object
    """
    n = data.shape
    n_par = par.__len__()
    
    if n_par > 2 :
        llh = np.sum(dist.logpdf(data, par[:-2], loc=par[-2], scale=par[-1]))
    else :
        llh = np.sum(dist.logpdf(data, loc=par[-2], scale=par[-1]))
    
    return -2*llh-n_par*np.log(n)
      
def u_fun(dist, par, data):
    """
    probability transform for a scipy distribution object
    """
    n_par = par.__len__()
    if n_par > 2 :
        return dist.cdf(data, par[:-2], loc=par[-2], scale=par[-1])
    else :
        return dist.cdf(data, loc=par[-2], scale=par[-1])  


## parts 

def loglike(data, F, gamma, u, V, pi, r):
    """
    log-likelihood of vine copula mixture
    """
    n, d = data.shape
    k = pi.shape[1]

    #Creating the latent variable z_i,j
    z = np.apply_along_axis(lambda row : row==row.max(), 1, r) + 0
    
    log_f = np.column_stack([np.column_stack([logpdf(f, g, data[:, i]) for f, g, i in zip(F[j], gamma[j], range(d))]).sum(axis=1) for j in range(k)])
    
    log_c = np.log(np.column_stack([V[i].pdf(u[i]) for i in range(k)]))
    
    # Calculating llh
    llh = np.sum(z*np.log(pi)) + np.sum(z*log_f) + np.sum(z*log_c)
    
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
    
    
    # candidates = [stats.norm, stats.gumbel_l, stats.cauchy, stats.gamma, stats.logistic, stats.lognorm, stats.skewnorm, stats.t, stats.skewcauchy, stats.loggamma]
    # restricting candidate set for now to get a first run
    candidates = [stats.norm, stats.t]
    
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
        u:
        V:

    Output:
        z: NxK np array of assignment probabilities
    """
    n = len(data)
    k = pi.shape[1]
      
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
        r: NxK array of cluster probabilites
        data: NxD array with data on original scale
        F: KxD list with families of marginal distributions
        gamma: Parameters
        u: K item list of NxD np arrays of copula data
        V: Vinecop Object representing the vine copula structure
    Output:
        V: updated VineCop structure
    """
    n, d = data.shape
    _, k = r.shape
    
    def update_pair_param(v, u_k):
        new_V = vcl.Vinecop(structure=v.structure, pair_copulas=v.pair_copulas)
        new_V.select(data=u_k)
        return new_V    
    
    V = [update_pair_param(V[i], u[i]) for i in range(k)]
      
    return V
  

  
## main function

def vcmm(data, K, tol=0.00001, maxiter=100, initial_method="kmeans", fitting_trunclevel=1, trace=False):
  
    n, d = data.shape
    
    # 1. initial assignment
    r = initial_prob(data, K, method=initial_method)
    pi = mixture_weights(r)
    
    # 2. initial model selection
    F, gamma, u = select_marginals(data, r)
    V = RVineStructureSelect(u, trunclevel=fitting_trunclevel)
    
    # 3. parameter estimation
    t = 0
    llh = loglike(data, F, gamma, u, V, pi, r)
    
    while True :
      
      if trace :
          print("iter:",t,"llh:",llh,"proportions:",pi, sep=" ")
      
      llh_old = llh
      
      # E step
      r = posterior_prob(data, pi, F, gamma, u, V)
      
      # CM step 1
      pi = mixture_weights(r)
      
      # CM step 2
      gamma, u = marginal_parameters(r, data, F)
      
      # CM step 3
      V = pair_copula_parameters(r, data, F, gamma, u, V)
      
      # stopping condition(s)
      llh = loglike(data, F, gamma, u, V, pi, r)
      llhdiff = (llh-llh_old)/-llh_old
      
      if llhdiff < tol :
          break
      if t >= maxiter :
          break
        
      t = t+1
        
    # 4. temporary cluster assignment
    c = r.argmax(axis=1)
    
    # 5. final model selection
    F, gamma, u = select_marginals(data, r)
    V = RVineStructureSelect(u, trunclevel=d-1)
    
    # 6. final cluster assignment
    c = posterior_prob(data, pi, F, gamma, u, V).argmax(axis=1)
    
    # classifier
    def predict(x):
        ux = [np.column_stack([u_fun(F[i][j], gamma[i][j], x[:,j]) for j in range(d)]) for i in range(K)]
        pred_prob = posterior_prob(x, pi, F, gamma, ux, V)
        return pred_prob.argmax(axis=1), pred_prob
    
    return c, F, gamma, V, pi, r, predict


## plotting functions

import matplotlib.pyplot as plt
import matplotlib as mpl

# taken from HW3 code provided to us
def show_classification(X_train, X_test, VCMM_predictor):

    shapes = ['o', '*', 'v', '+']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b',
              '#e377c2', '#7f7f7f', '#bcbd22']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    for (ax, data, title) in [(ax1, X_train, 'Training Set'), (ax2, X_test, 'Test Set')]:
        pred_labels, obj = VCMM_predictor(data)
        print("K-Means Objective on " + title)
        cs = [colors[int(_) % len(colors)] for _ in pred_labels]
        ax.scatter(data[:, 0], data[:, 1], alpha=0.5, c=cs)
        ax.set_title(title)
        ax.set_xlim(-12, 12), ax.set_ylim(-12, 12)
        ax.set_aspect('equal')

    plt.show()
    print('\n')
    
    
def show_contour(X_train, predictor):
    xlim = [X_train[:,0].min(), X_train[:,0].max()]
    ylim = [X_train[:,1].min(), X_train[:,1].max()]
    
    x = np.linspace(xlim[0], xlim[1], num=100)
    y = np.linspace(ylim[0], ylim[1], num=100)
    
    xv, yv = np.meshgrid(x, y)
    
    z, probs = predictor(np.column_stack((xv.flatten(), yv.flatten())))
    
    fig = plt.figure()  # an empty figure with no axes
    grid = int(np.ceil(np.sqrt(probs.shape[1])))
    fig, ax_lst = plt.subplots(grid, grid)  # a figure with a 2x2 grid of Axes
    
    k = 0
    for i in range(grid):
        for j in range(grid):
            ax_lst[i,j].contourf(xv, yv, probs[:,k].reshape((100,100)), cmap=mpl.colormaps["Blues"])
            ax_lst[i,j].set_title("Component "+str(k))
            k += 1
    
    fig.suptitle("Contour Plots of VCMM Component Densities")
    
    plt.show()
    print('\n')
    

def show_boundary(X_train, predictor):
    xlim = [X_train[:,0].min(), X_train[:,0].max()]
    ylim = [X_train[:,1].min(), X_train[:,1].max()]
    
    x = np.linspace(xlim[0], xlim[1], num=100)
    y = np.linspace(ylim[0], ylim[1], num=100)
    
    xv, yv = np.meshgrid(x, y)
    
    z, probs = predictor(np.column_stack((xv.flatten(), yv.flatten())))
    
    
    shapes = ['o', '*', 'v', '+']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b',
              '#e377c2', '#7f7f7f', '#bcbd22']

    
    
    pred_labels, probs = predictor(X_train)
    
    cs = [colors[int(_) % len(colors)] for _ in pred_labels]
    
    fig = plt.figure()
    ax = fig.add_subplot()
    
    ax.scatter(X_train[:, 0], X_train[:, 1], alpha=0.5, c=cs)
    fig.suptitle("Cluster Assignment and Decision Boundary")
    # ax.set_xlim(xlim)
    # ax.set_ylim(ylim)
    ax.contour(xv, yv, z.reshape((100,100)), colors="blue")

    plt.show()
    print('\n')
    
    # plt.contour(xv, yv, z.reshape((100,100)), colors="blue")
    # plt.show()
    

assignment, marginals, marg_par, vinecop, proportions, probabilities, predictor = vcmm(dfs[1], K=4, trace=True)

show_classification(dfs[0], dfs[1], predictor)

show_contour(dfs[0], predictor)

show_boundary(dfs[0], predictor)

### simulation study


### applications


