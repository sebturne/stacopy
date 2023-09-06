import numpy
import scipy.stats
import sklearn.metrics
import sklearn.cluster
import joblib

def V(matrix):
    """
    Calculates Cramer's V index of association from a confusion matrix.
    Use sklearn.metrics.confusion_matrix to set up a confusion matrix.
    """
    chi = scipy.stats.chi2_contingency(matrix)[0]
    n = matrix.sum()
    return (numpy.sqrt(chi / (n*(min(matrix.shape) - 1))))

def STACOK(data, k):
    """
    Runs k-means once for input data at input k.
    Outputs are labels (list) and compactness.
    """
    kmrun = sklearn.cluster.KMeans(n_clusters = k, max_iter = data.shape[0], n_init = 1, algorithm = 'full', tol = 0, init = 'k-means++', random_state = None).fit(data)
    return kmrun.predict(data), kmrun.inertia_

def STACOV(lbls, i):
    """
    Calculates median V for ith solution w.r.t. other solutions at the same k.
    Takes a set of solutions as input; solutions should all be at the same k.
    """
    lbls_others = numpy.delete(lbls, i, axis = 1)
    cvind = numpy.zeros((lbls_others.shape[1],1))
    for q in range(lbls_others.shape[1]):
        cvind[q,0] = V(sklearn.metrics.confusion_matrix(lbls[:,i],lbls_others[:,q]))
    return numpy.median(cvind)

def STACO(data, k=range(2,11), init=100, save_lbls = False):
    """
    Calculates stabilities and compactnesses for init solutions at input k for input data
    Input data must be clustering features only: no IDs, flags, etc. 
    Data prep (normalisations, scaling, etc.) should be complete before use of this function.
    Defaults to saving all labels (save_lbls = False).
    Output is table (dimensions [len(k)*init, 3]) of stabilities, compactnesses, and k.
    """
    obs, feats = data.shape
    nk = len(k)

    lbls = numpy.zeros((obs,init,nk))
    phi = numpy.zeros((init,nk))

    print("Clustering...")

    for j in range(nk):
        print("k = %i" % k[j])
        l, p = zip(*joblib.Parallel(n_jobs = -1, mmap_mode = 'w+')(joblib.delayed(STACOK)(data,k[j]) for i in range(init)))
        lbls[:,:,j] = numpy.transpose(numpy.array(l)) # labels
        phi[:,j] = numpy.transpose(numpy.array(p)) # compactnesses
        
    print("Done.")

    if save_lbls == True:
        for j in range(nk): # saving all labels
            numpy.savetxt('lbls_k%i.txt' % k[j], lbls[:,:,j], delimiter = ',')

    cvmed = numpy.zeros((init,nk))

    print("Measuring stabilities...")

    for j in range(nk):
        print("k = %i" % k[j])
        cvmed[:,j] = joblib.Parallel(n_jobs = -1, mmap_mode = 'w+')(joblib.delayed(STACOV)(lbls[:,:,j],i) for i in range(init)) # stabilities
        
    print("Done.")

    cvmed = cvmed.flatten('F').reshape(-1,1)
    phi   = phi.flatten('F').reshape(-1,1)
    k_arr = numpy.ones((init,1))*k
    k_arr = k_arr.flatten('F').reshape(-1,1)

    return numpy.hstack((cvmed,phi,k_arr))
