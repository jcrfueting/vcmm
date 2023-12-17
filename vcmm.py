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

## GMM from HW3

def KMeans(X, K=1):
    # helper functions
    def dist(p1, p2):
        return np.linalg.norm(p1-p2)

    def least_dist(r, cents):
        d = np.apply_along_axis(lambda c: dist(c, r), 1, cents)
        return d.argmin()

    def loss(x, cent):
        return np.apply_along_axis(lambda r: dist(cent, r), 1, x).sum()
    
    n, d = X.shape
    
    # fitting parameters
    MAX_ITER = 100
    t = 0
    E_old = -np.ones(X.shape[0])
    E = E_old

    # fitting procedure
    centroids = np.random.randn(K, d)
    while(True):
        t = t + 1
        E_old = E

        # E step
        E = np.apply_along_axis(lambda row: least_dist(row, centroids), 1, X)

        # M step
        centroids = np.stack([X[E==k,:].mean(axis=0) for k in np.arange(K)])

        # stopping if no change in assignment or max iter reached
        if (E_old == E).all() :
            break
        if t >= MAX_ITER :
            break


    def KMeans_predictor(inX):
        pred_labels = np.apply_along_axis(lambda row: least_dist(row, centroids),
                                          1, inX)
        return pred_labels, 0

    return centroids, KMeans_predictor

def GaussianMixture(X, K=1, use_full_cov=True):
    """
    Estimates the parameters of a Gaussian mixture model using training data X

    **Important:** The locations of the centroids must be initialized using your
    K-Means code! With this information, initialize the proportions and variances
    accordingly.

        Inputs:
            X: [nx2] matrix of inputs
            K: [int] number of mixture components to use
            use_full_cov: [bool] if True, estimate a full covariance for each
                mixture component. Else, we use a scaled identity for each
                component (in this case each component might have a
                different scaling of the identity: Sigma_i = sigma_i * I).

        Returns:
            pi: [K] vector of proportion of each class
            centroids: [Kx2] matrix of estimated centroid locations
            sigmas: [Kx2x2] tensor of estimated covariance matrices
            MoG_predictor: function taking a matrix inX and returning the
                predicted cluster number (starting from 0 to K-1) for each row
                of inX; and the normalized log-likelihood (i.e., ln(p(inX)) divided by the
                number of rows of inX).
    """

    def norm_kernel(x, mu, sigma):
        z = np.sqrt(np.linalg.det(sigma))
        siginv = np.linalg.inv(sigma)
        quadform = np.apply_along_axis(lambda r: r @ siginv @ r.T, 1, x-mu)
        return 1/z*np.exp(-.5 * quadform).reshape(-1,1)

    def tau_fun(x, m, sig, p):
        pxk = np.column_stack([norm_kernel(x, m[i,:], sig[i,:])
                               for i in np.arange(sig.shape[0])])
        pipxk = pxk*p
        return pipxk/np.column_stack(sig.shape[0]*[pipxk.sum(axis=1)])

    def sigk2(x, m, tk):
        d = m.shape[0]
        return np.sum(np.diag(np.cov(x-m, rowvar=False, aweights=tk)))/d

    def loglik(x, mus, sigmas, pi):
        unscaled = np.column_stack([norm_kernel(x, mus[i,:], sigmas[i,:])
                               for i in np.arange(sigmas.shape[0])])
        unscaled = unscaled*pi
        n, d = x.shape
        omitted_factor = -n*d/2*np.log(2*np.pi)
        return omitted_factor+np.log(unscaled.sum(axis=1)).sum()
    
    n, d = X.shape

    # hyperparams
    MAX_ITER = 100
    INIT_SIG_MULT = 10
    LOGLIK_TOL = 0.01

    # Initialize centroids using KMeans
    centroids, kmpred = KMeans(X, K)

    # initialize pi with proportions from KMeans
    predictions, _ = kmpred(X)
    pi = np.array([np.mean(predictions==k) for k in np.arange(K)])

    # initialize sigma as large and spherical
    sigmas = np.concatenate([np.var(X)*INIT_SIG_MULT * np.eye(d)[np.newaxis, ...]
                             for i in range(K)], axis=0)

    t = 0
    ll_old = loglik(X, centroids, sigmas, pi)
    ll = ll_old

    # estimation
    while(True):
        t = t + 1

        # E step
        tau = tau_fun(X, centroids, sigmas, pi)

        # M step
        softcount = tau.sum(axis=0)
        pi = softcount/X.shape[0]
        centroids = np.stack([(X*np.column_stack(d*[tau[:,k]])).sum(axis=0)/softcount[k]
                              for k in np.arange(K)])

        if use_full_cov :
            sigmas = np.concatenate([np.cov(X, aweights=tau[:,k], rowvar=False)[np.newaxis, ...]
                                     for k in np.arange(K)], axis=0)

        else :
            sigmak2 = [sigk2(X, centroids[k,:], tau[:,k])
                       for k in np.arange(K)]
            sigmas = np.concatenate([sigmak2[k] * np.eye(d)[np.newaxis, ...]
                                     for k in range(K)], axis=0)


        # stopping
        ll_old = ll
        ll = loglik(X, centroids, sigmas, pi)

        if (ll-ll_old) < LOGLIK_TOL :
            break
        if t >= MAX_ITER :
            break


    def MoG_predictor(inX):
        """
        # Use parameters from above to predict cluster for each row of inX

            Inputs:
                inX: [mx2] matrix of inputs

            Returns:
                pred_labels: [m] array of predicted cluster labels
                norm_loglike: [float] the log-likelihood of inX, i.e. ln(p(inX)),
                    divided by the number of rows of inX.
        """
        pred_probs = tau_fun(inX, centroids, sigmas, pi)
        pred_labels =  pred_probs.argmax(axis=1)

        return pred_labels, pred_probs

    return pi, centroids, sigmas, MoG_predictor

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
      
    elif method=="gmm":
        pi, centroids, sigmas, predictor = GaussianMixture(data, K=K)
        z, r = predictor(data)
        return np.apply_along_axis(lambda row : row==row.max(), 1, r)+0
      
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

def posterior_density(data, pi, F, gamma, u, V):
    k = pi.shape[1]
    g = np.column_stack([component_prob(data, F[i], gamma[i], u[i], V[i]) for i in range(k)])
    return g*pi

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
        pred_dens = posterior_density(x, pi, F, gamma, ux, V)
        return pred_prob.argmax(axis=1), pred_prob, pred_dens
    
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
        pred_labels, obj, _ = VCMM_predictor(data)
        print("K-Means Objective on " + title)
        cs = [colors[int(_) % len(colors)] for _ in pred_labels]
        ax.scatter(data[:, 0], data[:, 1], alpha=0.5, c=cs)
        ax.set_title(title)
        ax.set_xlim(-12, 12), ax.set_ylim(-12, 12)
        ax.set_aspect('equal')

    plt.show()
    print('\n')
    
    
def show_contour(X_train, predictor, contour="density"):
    xlim = [X_train[:,0].min(), X_train[:,0].max()]
    ylim = [X_train[:,1].min(), X_train[:,1].max()]
    
    x = np.linspace(xlim[0], xlim[1], num=100)
    y = np.linspace(ylim[0], ylim[1], num=100)
    
    xv, yv = np.meshgrid(x, y)
    
    z, probs, densities = predictor(np.column_stack((xv.flatten(), yv.flatten())))
    
    if contour=="density":
       plotz = densities
    elif contour=="probability":
       plotz = probs
    
    fig = plt.figure()  # an empty figure with no axes
    grid = int(np.ceil(np.sqrt(plotz.shape[1])))
    fig, ax_lst = plt.subplots(grid, grid)  # a figure with a 2x2 grid of Axes
    
    k = 0
    for i in range(grid):
        for j in range(grid):
            ax_lst[i,j].contourf(xv, yv, plotz[:,k].reshape((100,100)), cmap=mpl.colormaps["Blues"])
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
    
    z, probs, _ = predictor(np.column_stack((xv.flatten(), yv.flatten())))
    
    
    shapes = ['o', '*', 'v', '+']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b',
              '#e377c2', '#7f7f7f', '#bcbd22']

    
    
    pred_labels, probs, _ = predictor(X_train)
    
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
    

assignment, marginals, marg_par, vinecop, proportions, probabilities, predictor = vcmm(dfs[0], K=4, trace=True, initial_method="gmm")

show_classification(dfs[0], dfs[1], predictor)

show_contour(dfs[0], predictor, contour="density")

show_contour(dfs[0], predictor, contour="probability")

show_boundary(dfs[0], predictor)

### simulation study

import pandas as pd

ais = pd.read_csv("project/AIS.csv")
ais_np = ais.iloc[:,2:].to_numpy()

breastcancer = pd.read_csv("project/BreastCancer.csv")
bc_np = breastcancer.iloc[:,1:(breastcancer.shape[1]-1)].to_numpy()

protein = pd.read_table("project/sachs.data.txt").to_numpy()

# right now ais and breastcancer fail to fit...
assignment, marginals, marg_par, vinecop, proportions, probabilities, predictor = vcmm(protein, K=4, trace=True, initial_method="gmm")

### applications


