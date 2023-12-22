###
###   Plotting Functions for Vine Copula Mixture Models
###

## plotting functions

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.tri as tri

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
    

def show_probability(X, predictor, pdfname, title=None, contour="density", plotgroup=0, scale=2):

    n, d = X.shape    

    z, probs, densities = predictor(X)

    if contour=="density":
       plotz = densities
    elif contour=="probability":
       plotz = probs

    fig = plt.figure(figsize=(10,10), dpi=600)  # an empty figure with no axes
    fig, ax_lst = plt.subplots(d-1, d-1)
    figscale = fig.dpi/72/scale

    k = 0
    for i in range(d)[1:]:
        for j in range(d-1):

            if i <= j :
                ax_lst[i-1,j].axis("off")
                continue

            x = X[:,j]
            y = X[:,i]
            ax_lst[i-1,j].scatter(x, y, c=plotz[:,plotgroup], cmap=mpl.colormaps["cool"], 
                                  s = figscale/4, linewidths=figscale/16)
            ax_lst[i-1,j].tick_params(axis="both", which="both", labelsize=figscale, 
                                      width=figscale/4, length=figscale*1.5,
                                      grid_linewidth=figscale/4, pad=figscale/2)
            # contour plot is unfortunately not very expressive
            # triangles = tri.Triangulation(x, y)
            # ax_lst[i-1,j].tricontourf(triangles, plotz[:,plotgroup], cmap=mpl.colormaps["Blues"])
            k += 1

    fig.suptitle(title)
    
    plt.savefig(pdfname+".pdf")
    plt.show()
    print('\n')

def show_missclass(X, predictor, labels, pdfname, title=None, plotgroup=0, scale=2):

    n, d = X.shape    

    z, probs, densities = predictor(X)

    plotz = np.abs((labels==plotgroup)+0 - probs[:,plotgroup])

    fig = plt.figure(figsize=(10,10), dpi=600)  # an empty figure with no axes
    fig, ax_lst = plt.subplots(d-1, d-1)
    figscale = fig.dpi/72/scale

    k = 0
    for i in range(d)[1:]:
        for j in range(d-1):

            if i <= j :
                ax_lst[i-1,j].axis("off")
                continue

            x = X[:,j]
            y = X[:,i]
            ax_lst[i-1,j].scatter(x, y, c=plotz, cmap=mpl.colormaps["Reds"], 
                                  s = figscale/4, linewidths=figscale/16)
            ax_lst[i-1,j].tick_params(axis="both", which="both", labelsize=figscale, 
                                      width=figscale/4, length=figscale*1.5,
                                      grid_linewidth=figscale/4, pad=figscale/2)
            k += 1

    fig.suptitle(title)
    
    plt.savefig(pdfname+".pdf")
    plt.show()
    return np.sum(plotz)

def compare_missclass(X, pairs, predA, predB, labels, pdfname, title=None, plotgroup=0, scale=2):
    n, d = X.shape    

    _, probsA, _ = predA(X)
    plotzA = np.abs((labels==plotgroup)+0 - probsA[:,plotgroup])

    _, probsB, _ = predB(X)
    plotzB = np.abs((labels==plotgroup)+0 - probsB[:,plotgroup])

    # return plotzA, plotzB

    fig = plt.figure(figsize=(10,2), dpi=600) 
    fig, ax_lst = plt.subplots(len(pairs), 2)
    figscale = fig.dpi/72/scale

    for i in range(len(pairs)):
        x = X[:,pairs[i][0]]
        y = X[:,pairs[i][1]]

        ax_lst[i,0].scatter(x, y, c=plotzA, cmap=mpl.colormaps["Reds"], 
                                s = figscale/4, linewidths=figscale/16)
        ax_lst[i,0].tick_params(axis="both", which="both", labelsize=figscale, 
                                    width=figscale/4, length=figscale/2,
                                    grid_linewidth=figscale/4, pad=figscale/2)
        
        ax_lst[i,1].scatter(x, y, c=plotzB, cmap=mpl.colormaps["Reds"], 
                                s = figscale/4, linewidths=figscale/16)
        ax_lst[i,1].tick_params(axis="both", which="both", labelsize=figscale, 
                                    width=figscale/4, length=figscale/2,
                                    grid_linewidth=figscale/4, pad=figscale/2)

    fig.suptitle(title)    
    plt.savefig(pdfname+".pdf")
    plt.show()
    return 0

def compare_2d(X, predA, predB, contour, title=None, file=None):
    
    shapes = ['o', '*', 'v', '+']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b',
              '#e377c2', '#7f7f7f', '#bcbd22']

    xlim = [X[:,0].min(), X[:,0].max()]
    ylim = [X[:,1].min(), X[:,1].max()]
    
    x = np.linspace(xlim[0], xlim[1], num=100)
    y = np.linspace(ylim[0], ylim[1], num=100)
    
    xv, yv = np.meshgrid(x, y)
    
    zA, probsA, densitiesA = predA(np.column_stack((xv.flatten(), yv.flatten())))
    zB, probsB, densitiesB = predB(np.column_stack((xv.flatten(), yv.flatten())))
    
    if contour=="density":
       plotzA = densitiesA
       plotzB = densitiesB
    elif contour=="probability":
       plotzA = probsA
       plotzB = probsB

    pred_labelsA, probs, _ = predA(X)
    pred_labelsB, probs, _ = predB(X)
    
    csA = [colors[int(_) % len(colors)] for _ in pred_labelsA]
    csB = [colors[int(_) % len(colors)] for _ in pred_labelsB]

    fig = plt.figure(figsize=(10,5), dpi=600)  # an empty figure with no axes
    fig, ax_lst = plt.subplots(5, 2)  
    
    # classification and boundary
    ax_lst[0,0].scatter(X[:, 0], X[:, 1], alpha=0.5, c=csA)
    ax_lst[0,0].contour(xv, yv, zA.reshape((100,100)), colors="blue")
    ax_lst[0,0].set_title("GMM")

    ax_lst[0,1].scatter(X[:, 0], X[:, 1], alpha=0.5, c=csB)
    ax_lst[0,1].contour(xv, yv, zB.reshape((100,100)), colors="blue")
    ax_lst[0,1].set_title("VCMM")

    # density/probability contours
    k = 0
    for i in range(1, 5):

        ax_lst[i,0].contourf(xv, yv, plotzA[:,k].reshape((100,100)), cmap=mpl.colormaps["Blues"])
        #ax_lst[i,0].set_title("Component "+str(k))
        
        ax_lst[i,1].contourf(xv, yv, plotzB[:,k].reshape((100,100)), cmap=mpl.colormaps["Blues"])
        #ax_lst[i,1].set_title("Component "+str(k))

        k += 1

    plt.savefig(file+".pdf")
    plt.show()


