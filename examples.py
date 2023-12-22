###
###   Vine Copula Mixture Models - Examples
###

import os

### data for development and testing (currently only 2 dimensional)

dfs = [np.loadtxt(os.getcwd()+"/EMGaussian."+t_flag) for t_flag in ["train", "test"]]

# empirial pseudo copulas
copula_dfs = [vcl.to_pseudo_obs(x) for x in dfs]

### toy data

assignment, marginals, marg_par, vinecop, proportions, probabilities, predictor = vcmm(dfs[0], K=4, trace=True, initial_method="gmm")

show_classification(dfs[0], dfs[1], predictor)

show_contour(dfs[0], predictor, contour="density")

show_contour(dfs[0], predictor, contour="probability")

show_boundary(dfs[0], predictor)

### applications

import pandas as pd

ais = pd.read_csv(os.getcwd()+"/AIS.csv")
ais_np = ais.iloc[:,2:].to_numpy()
ais_labels = pd.factorize(ais.iloc[:,0])[0]

breastcancer = pd.read_csv(os.getcwd()+"/BreastCancer.csv")
breastcancer.dropna(inplace=True) 
bc_np = breastcancer.iloc[:,1:(breastcancer.shape[1]-1)].to_numpy()
bc_labels = pd.factorize(breastcancer.iloc[:,-1])[0]

protein = pd.read_table(os.getcwd()+"/sachs.data.txt").to_numpy()

# test runs on real data
assignment, marginals, marg_par, vinecop, proportions, probabilities, predictor = vcmm(ais_np, labels=ais_labels, trace=True, initial_method="gmm")

assignment, marginals, marg_par, vinecop, proportions, probabilities, predictor = vcmm(bc_np, labels=bc_labels, trace=True, initial_method="gmm", maxiter=10)

# various visualisations on the AIS dataset     
show_probability(ais_np, predictor, pdfname="ais_vcmm_probability.pdf", contour="probability",plotgroup=0)

_, _, _, _, _, _, gmm_predictor = vcmm(ais_np, labels=ais_labels, trace=True, initial_method="gmm",
                                       marginals = [stats.norm], copulas=["gaussian"])

show_probability(ais_np, gmm_predictor, pdfname="ais_gmm_probability.pdf", contour="probability",plotgroup=0)


show_missclass(ais_np, predictor, ais_labels, pdfname="ais_vcmm_misclass.pdf", plotgroup=0)

show_missclass(ais_np, gmm_predictor, ais_labels, pdfname="ais_gmm_misclass.pdf", plotgroup=0)


comparison_pairs = [(1,0),(2,0),(3,0)]
compare_missclass(ais_np, comparison_pairs, gmm_predictor, predictor,  ais_labels, 
                  pdfname="ais_comparison.pdf", title="Comparison of Missclassifications",
                  scale=.2)
                  
                  
# illustrative comparison on toy dataset
_, _, _, _, _, _, vcmm_em = vcmm(dfs[0], K=4, trace=True, initial_method="gmm")
_, _, _, _, _, _, gmm_em = vcmm(dfs[0], K=4, trace=True, initial_method="gmm",
                                       marginals = [stats.norm], copulas=["gaussian"])

compare_2d(dfs[0], gmm_em, vcmm_em, contour="density")
