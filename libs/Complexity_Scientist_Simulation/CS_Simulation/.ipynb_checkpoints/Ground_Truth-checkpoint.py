#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 22 15:08:48 2021

@author: marinadubova

@Last Modified by:   Munjung Kim
@Last Modified time: 2023-06-26 23:47:35

"""

# todo
# make it heritated


import scipy.stats as stats
import numpy as np


class multivariate_gaussian:

    def __init__(self, n_dims, max_loc, wishart_scale=5):
        """
        Initializing multivariate gaussian ground truth

        Args:
            n_dims : dimensions of the ground truth space
            max_loc : maximum values of uniform distribution for calculating mean values of the mutlivariate gaussian distribution
            wishart_scale : the scale of covariance matrix of the wishart dsitribution for getting the covariance martrix for multivariate gaussian distribution
        """

        self.n_dims = n_dims
        self.loc = np.random.uniform(max_loc, size=n_dims)

         # Change 5 to a bigger number to get less elongated distributions
        self.scale = stats.wishart.rvs(scale=np.eye(n_dims) * wishart_scale, df=self.n_dims + 2, size=1) 

    def sample(self):

        return stats.multivariate_normal.rvs(mean=self.loc, cov=self.scale)

    def sample_conditioned(self, fixed_dims, values, return_full=False):
        """
        Getting samples with some restrictions 

        Args:
            fixed_dims: list of the dimension controlled
            values : the values where the fixed_dims will be fixed
        """
        assert len(fixed_dims) < self.n_dims, "Too many fixed dimensions!"

        dims_set = set(fixed_dims)
        free_dims = [i for i in range(self.n_dims) if i not in dims_set]


        sigma_22 = self.scale[fixed_dims, :][:, fixed_dims]
        sigma_11 = self.scale[free_dims, :][:, free_dims]
        sigma_12 = self.scale[free_dims, :][:, fixed_dims]

        mu_1 = self.loc[free_dims]
        mu_2 = self.loc[fixed_dims]

        tmp = sigma_12 @ np.linalg.inv(sigma_22)
        sigma_bar = sigma_11 - tmp @ sigma_12.T
        mubar = mu_1 + tmp @ (values - mu_2)

        sample = stats.multivariate_normal.rvs(mean=mubar, cov=sigma_bar)

        if return_full:
            res = np.zeros(self.n_dims)
            res[free_dims] = sample
            res[fixed_dims] = values
            return res

        else:
            return sample

    def marginal_pdf(self, dims, values):
        sigma_new = self.scale[dims, :][:, dims]
        mu_new = self.loc[dims]

        return stats.multivariate_normal.pdf(values, mean=mu_new, cov=sigma_new)


class clustered_multivariate_gaussian:

    def __init__(self, n_dims, max_loc, num_clusters, wishart_scale=5):
        """
        Initializing clustered multivariate gaussian ground truth

        Args:
            n_dims : dimensions of the ground truth space
            max_loc : maximum values of uniform distribution for calculating mean values of the mutlivariate gaussian distribution
            wishart_scale : the scale of covariance matrix of the wishart dsitribution for getting the covariance martrix for multivariate gaussian distribution
            num_clusters : number of clusters of multivariate gaussian distribution
        """

        self.n_dims = n_dims
        self.max_loc = max_loc
        self.num_clusters = num_clusters

        self.cluster_priors = np.full(num_clusters, 1 / num_clusters)
        self.clusters = [multivariate_gaussian(n_dims, max_loc, wishart_scale=wishart_scale) for _ in range(num_clusters)]

    def sample(self, cluster_probs=None, cond_dims=None, cond_vals=None, return_full=False):

        # idea - pick a random cluster, then sample a value
        cluster_ind = np.random.choice(np.arange(self.num_clusters),
                                       p=self.cluster_priors if cluster_probs is None else cluster_probs)
        if cond_dims is not None:
            return self.clusters[cluster_ind].sample_conditioned(cond_dims, cond_vals, return_full)
        else:
            return self.clusters[cluster_ind].sample()

    def sample_conditioned(self, fixed_dims, values, return_full=False):

        # idea - compute (marginal) likelihoods of observed values, compute posteriors
        # for clusters, then sample from these clusters in a conditional way

        posteriors = self.cluster_priors * np.array([c.marginal_pdf(fixed_dims, values) for c in self.clusters])
        posteriors = posteriors / np.sum(posteriors)

        return self.sample(cluster_probs=posteriors, cond_dims=fixed_dims,
                                                     cond_vals=values,
                                                     return_full=return_full)


    
    
    
def find_nearest(array, dim,value):
    
    
    array = np.asarray(array[:,dim])
    
    dist = np.linalg.norm(np.abs(array - np.array( value)), axis=1) 
    idx = dist.argmin()
    
    return array[idx]


class Based_on_Formula:
    
    def __init__(self, data):
        """
        Initializing ground truth based on given data

        Args:
            data : the ground truth dataset (numpy array)
            max_loc : maximum values of uniform distribution for calculating mean values of the mutlivariate gaussian distribution
            wishart_scale : the scale of covariance matrix of the wishart dsitribution for getting the covariance martrix for multivariate gaussian distribution
            num_clusters : number of clusters of multivariate gaussian distribution
        """

        self.ground_truth = data
        self.n_dims = data.shape[1]
        self.n_data = data.shape[0]
        
        
#     def sample(self):
#         random_index = np.random.choice(len(data), 1)

#         return data[random_index]
    
    
    def sample(self, cluster_probs=None, cond_dims=None, cond_vals=None, return_full=False):

        # idea - pick a random cluster, then sample a value
        
        if cond_dims is not None:
            return self.sample_conditioned(cond_dims, cond_vals, return_full)
        else:
            random_index = np.random.choice(len(self.ground_truth), 1)
            return self.ground_truth[random_index]

    def sample_conditioned(self, fixed_dims, values, return_full=False):
        """
        Getting samples with some restrictions 

        Args:
            fixed_dims: list of the dimension controlled (list)
            values : the values where the fixed_dims will be fixed (list)
        """
        assert len(fixed_dims) < self.n_dims, "Too many fixed dimensions!"

        dims_set = set(fixed_dims)
        free_dims = [i for i in range(self.n_dims) if i not in dims_set]
        
        
        closet_data = find_nearest(self.ground_truth,fixed_dims,values)
        

        

        if return_full:
            return closet_data

        else:
            return closet_data[fee_dims]

        
    