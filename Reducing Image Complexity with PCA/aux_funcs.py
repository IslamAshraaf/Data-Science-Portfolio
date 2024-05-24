import numpy as np
import matplotlib.pyplot as plt

#Centering data
def center_data(data_to_center):
    """
    Center your original data
    Args:
         data_to_center (ndarray): input data. Shape (n_observations x n_pixels)
    Outputs:
        X (ndarray): centered data
    """
    mean_vector = np.mean(data_to_center,axis=0)
    mean_matrix = np.repeat(mean_vector,data_to_center.shape[0])
    # use np.reshape to reshape into a matrix with the same size as Y. Remember to use order='F'
    mean_matrix = mean_matrix.reshape((data_to_center.shape[0],data_to_center.shape[1]),order='F')
    
    X = data_to_center - mean_matrix
    return X

#Getting cov_matrix
def get_cov_matrix(data_centered):
    """ Calculate covariance matrix from centered data 
    Args:
        data_centered (np.ndarray): centered data matrix
    Outputs:
       covariance matrix
    """
    cov_mat = data_centered.T @ data_centered
    cov_mat = cov_mat / (len(data_centered)-1)
    return cov_mat

#Applying PCA manually
def perform_PCA(data_centered, eigenvects, k):
    """
    Perform dimensionality reduction with PCA
    Inputs:
        data_centered (ndarray): original centered data matrix. Has dimensions (n_observations)x(n_variables)
        eigenvects (ndarray): matrix of eigenvectors. Each column is one eigenvector. The k-th eigenvector 
                            is associated to the k-th eigenvalue
        k (int): number of principal components to use
    Returns:
        Xred
    """
    V = eigenvects[:,:k]
    Xred = data_centered @ V

    return Xred
#Visualize_exp_var_ratio
def visualize_exp_var(x_range, exp_var, axhline):
    fig,ax = plt.subplots(1,2, figsize=(18,6), sharex=True)
    ax[0].plot(x_range, exp_var)
    ax[0].bar(x_range, exp_var)
    ax[0].set_title('Explained Variance Plot',size=15)
    ax[0].set_xlabel('Principal Components')
    ax[0].set_ylabel('Explained Variance Ratio')
    #------------------
    ax[1].plot(x_range, np.cumsum(exp_var))
    ax[1].axhline(y=axhline,c='r')
    ax[1].set_title('Cummulative Explained Variance Plot',size=15)
    ax[1].set_xlabel('Principal Components')
    ax[1].set_ylabel('Cummulative Explained Variance Ratio')

#Reconstructing images manually
def reconstruct_manually(imgs_pca,eigenvects):
    return np.dot(imgs_pca, eigenvects[:,:imgs_pca.shape[1]].T)
    
#Reconstructin from sklearn
def reconstruct_sk(imgs_pca,eigenvects,k):
    return np.dot(imgs_pca,eigenvects[:k,:])

#Plot data on 2D
def plot_reduced_data(X):
    plt.figure(figsize=(12,12))
    plt.scatter(X[:,0], X[:,1], s=60, alpha=.5)
    for i in range(len(X)):
        plt.text(X[i,0], X[i,1], str(i),size=8)
    plt.show()
