# IMPORTS
import csv
import getopt
import glob
import os
import os.path as op
import pickle
import sys

import nibabel as nib
import numpy as np
import pandas as pd
import pylab as plt
import seaborn as sns
from joblib import Parallel, delayed, dump, load
from nilearn.image import coord_transform, math_img, mean_img, threshold_img
from nilearn.input_data import MultiNiftiMasker
from nilearn.masking import apply_mask, compute_epi_mask
from nilearn.plotting import plot_glass_brain
from numpy.random import randint
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, LeaveOneGroupOut

cimport numpy as np



def clean_rscorer(estimator, np.ndarray[double, ndim=2] X, np.ndarray[float, ndim=2] y):
    cdef np.ndarray[double, ndim=1] x1
    x1 = r2_score(y, estimator.predict(X), multioutput='raw_values')
    x1 = np.array([0 if (x < .0 or x >= .99) else x for x in x1])
    return x1

def log(r2_train, r2_test, dim, voxel_id, logcsvwriter=logcsvwriter):
        """ just logging stats per fold to a csv file """
        logcsvwriter.writerow([loglabel, voxel_id, dim, 'training', np.mean(r2_train), np.std(r2_train), np.min(r2_train), np.max(r2_train)])
        logcsvwriter.writerow([loglabel, voxel_id, dim, 'test', np.mean(r2_test), np.std(r2_test), np.min(r2_test), np.max(r2_test)])

def search_best_dim(list fmri_runs, list design_matrices, np.ndarray[long, ndim=1] dimensions, object loglabel, object logcsvwriter, object estimator):
    
    n_dimension_test = len(dimensions)
    n_train = len(fmri_runs)
    n_voxel = fmri_runs[0].shape[1]
    cdef np.ndarray[double, ndim=3] r2_cv_train_score
    cdef np.ndarray[double, ndim=3] r2_cv_test_score
    cdef double r2_train_score
    cdef double r2_test_score
    cdef np.ndarray[float, ndim=2] fmri_data
    cdef np.ndarray[double, ndim=2] predictors_ref
    cdef np.ndarray[double, ndim=2] predictors
    train = [i for i in range(0, n_train)]
    r2_cv_train_score = np.zeros((n_dimension_test, n_train, n_voxel))
    r2_cv_test_score = np.zeros((n_dimension_test, n_train, n_voxel))

    for idx, cv_test_id in enumerate(train):
        fmri_data = np.vstack([fmri_runs[i] for i in train if i != cv_test_id])
        predictors_ref = np.vstack([design_matrices[i] for i in train if i != cv_test_id])
        n_images_train = predictors_ref.shape[0]
        n_images_test = design_matrices[cv_test_id].shape[0]
        for idx1, dim in enumerate(dimensions):
            predictors = np.ones((n_images_train, dim+1))
            predictors[:, :-1] = predictors_ref[:, :dim]
            predictor_test = np.ones((n_images_test, dim+1))
            predictor_test[:, :-1] = design_matrices[cv_test_id][:, :dim]
            if estimator = 'LinearRegression':
                model = LinearRegression(fit_intercept=False, normalize=False, n_jobs=-1).fit(predictors, fmri_data)
            else:
                model = LinearRegression(fit_intercept=False, normalize=False, n_jobs=-1).fit(predictors, fmri_data)
            r2_cv_train_score[idx1, idx, :] = clean_rscorer(model, predictors, fmri_data)
            r2_cv_test_score[idx1, idx, :] = clean_rscorer(model, predictor_test, fmri_runs[cv_test_id])
            
    for voxel_id in range(n_voxel):
        for idx2, dim in enumerate(dimensions):
            log(r2_cv_train_score[idx2, :, voxel_id], r2_cv_test_score[idx2, :, voxel_id], dim, voxel_id)
        
    best_dim_idx = r2_cv_test_score.sum(axis=(1, 2)).argmax()
    best_dim = dimensions[best_dim_idx]
    
    return r2_cv_train_score[best_dim_idx, :].mean(axis=(0)), r2_cv_test_score[best_dim_idx, :].mean(axis=(0)), best_dim

