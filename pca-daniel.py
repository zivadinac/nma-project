def pca_projection(data,var_threshold=0.9):
    """
    Parameters:
        data           -> expects matrix with shape (samples,features)
        var_threshold  -> percentage of explained variance 
    Returns:
        data projected on k principal components explaining 'var_threshold' percentage of variance
    """
    
    n_samples = data.shape[0]
    data = data - data.mean(axis=0)
    cov =  (data.T @ data) / n_samples
    
    values, vectors = np.linalg.eig(cov)
    
    var_explained = np.cumsum(values)/np.sum(values)
    k = np.sum(var_explained<var_threshold)

    score_allPCs = data @ vectors
    vectors_allPCs = vectors
    score_sigPCs = score_allPCs[:,:k]
    vectors_sigPCs = vectors_allPCs[:,:k]
    
    return score_sigPCs, vectors_sigPCs, score_allPCs, vectors_allPCs

#Test for PCA function
data = dat['sresp'][:100,:].T
score_sigPCs, vectors_sigPCs, score_allPCs, vectors_allPCs = pca_projection(data,var_threshold=0.9)
reconstruction =  (score_allPCs @ vectors_allPCs.T) + data.mean(axis=0)
reconstruction_2 =  (score_sigPCs @ vectors_sigPCs.T) + data.mean(axis=0)
print(data-reconstruction)