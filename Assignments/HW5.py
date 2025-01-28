import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as linalg
import scipy.spatial.distance as dt
import scipy.stats as stats

group_means = np.array([[-5.0, +0.0],
                        [+0.0, +5.0],
                        [+5.0, +0.0],
                        [+0.0, -5.0]])
group_covariances = np.array([[[+0.4, +0.0],
                               [+0.0, +6.0]],
                              [[+6.0, +0.0],
                               [+0.0, +0.4]],
                              [[+0.4, +0.0],
                               [+0.0, +6.0]],
                              [[+6.0, +0.0],
                               [+0.0, +0.4]]])

# read data into memory
data_set = np.genfromtxt("hw05_data_set.csv", delimiter = ",")

# get X values
X = data_set[:, [0, 1]]

# set number of clusters
K = 4

# STEP 2
# should return initial parameter estimates
# as described in the homework description
def initialize_parameters(X, K):
    # your implementation starts below
    means = np.genfromtxt("hw05_initial_centroids.csv", delimiter=",")
    
    centroid_distances = dt.cdist(X, means)
    nearest_centroids = np.argmin(centroid_distances, axis=1)
    
    covariances = np.zeros((K, X.shape[1], X.shape[1]))
    priors = np.zeros(K)

    for k in range(K):
        cluster_points = X[nearest_centroids == k]
        
        priors[k] = len(cluster_points) / len(X)
        
        means[k] = np.mean(cluster_points, axis=0)

        X_centered = cluster_points - means[k]
        covariances[k] = (X_centered.T @ X_centered) / cluster_points.shape[0]
    # your implementation ends above
    return(means, covariances, priors)

means, covariances, priors = initialize_parameters(X, K)

# STEP 3
# should return final parameter estimates of
# EM clustering algorithm
def em_clustering_algorithm(X, K, means, covariances, priors):
    # your implementation starts below
    N = X.shape[0]
    assignments = np.zeros(N, dtype=int)

    for iteration in range(100):
        # E-step
        likelihood_xi = np.zeros((N, K))
        for k in range(K):
            gaussian_pdf = stats.multivariate_normal.pdf(X, mean=means[k], cov=covariances[k])
            likelihood_xi[:, k] = priors[k] * gaussian_pdf
        likelihood_xi = likelihood_xi / likelihood_xi.sum(axis=1, keepdims=True)

        # M-step
        for k in range(K):
            priors[k] = likelihood_xi[:, k].sum() / N
            
            means[k] = (likelihood_xi[:, k][:, np.newaxis] * X).sum(axis=0) / likelihood_xi[:, k].sum()

            X_centered = X - means[k]
            covariances[k] = sum(likelihood_xi[i, k] * np.outer(X_centered[i], X_centered[i]) for i in range(N)) / likelihood_xi[:, k].sum()

        assignments = np.argmax(likelihood_xi, axis=1)
    # your implementation ends above
    return(means, covariances, priors, assignments)

means, covariances, priors, assignments = em_clustering_algorithm(X, K, means, covariances, priors)
print(means)
print(priors)

# STEP 4
# should draw EM clustering results as described
# in the homework description
def draw_clustering_results(X, K, group_means, group_covariances, means, covariances, assignments):
    # your implementation starts below
    cluster_colors = np.array(["#1f78b4", "#33a02c", "#e31a1c", "#ff7f00", "#6a3d9a", "#b15928",
                               "#a6cee3", "#b2df8a", "#fb9a99", "#fdbf6f", "#cab2d6", "#ffff99"])

    x_min, x_max = -8.0, 8.0
    y_min, y_max = -8.0, 8.0
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

    for k in range(K):
        plt.scatter(X[assignments == k, 0], X[assignments == k, 1], s=5, color=cluster_colors[k])

    def plot_org_gaussian(mean, cov, color):
        rv = stats.multivariate_normal(mean, cov)
        Z = rv.pdf(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.contour(xx, yy, Z, levels=[0.01], colors=color, linestyles='dashed', linewidths=1)
        
    def plot_EM_gaussian(mean, cov, color):
        rv = stats.multivariate_normal(mean, cov)
        Z = rv.pdf(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.contour(xx, yy, Z, levels=[0.01], colors=color, linestyles='solid', linewidths=1)

    for k in range(K):
        plot_org_gaussian(group_means[k], group_covariances[k], 'black')

    for k in range(K):
        plot_EM_gaussian(means[k], covariances[k], cluster_colors[k])

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.show()
    # your implementation ends above
    
draw_clustering_results(X, K, group_means, group_covariances, means, covariances, assignments)