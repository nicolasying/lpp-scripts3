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
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, LeaveOneGroupOut

cimport numpy as np


def clean_rscorer(estimator, np.ndarray[double, ndim=2] X, np.ndarray[float, ndim=2] y):
    cdef np.ndarray[double, ndim=1] x1
    x1 = r2_score(y, estimator.predict(X), multioutput='raw_values')
    x1 = np.array([0 if (x < .0 or x >= .99) else x for x in x1])
    return x1
    
def parallel_fit(double alpha, np.ndarray[double, ndim=2] x_train, np.ndarray[float, ndim=2] y_train, np.ndarray[double, ndim=2] x_test, np.ndarray[float, ndim=2] y_test):
    model = Ridge(alpha=alpha, fit_intercept=False, normalize=False).fit(x_train, y_train)
    # print(r2_score(y_train, model.predict(x_train)), r2_score(y_test, model.predict(x_test)))
    return clean_rscorer(model, x_train, y_train), clean_rscorer(model, x_test, y_test)

def compute_crossvalidated_r2_all_voxel(fmri_runs, design_matrices, alphas, loglabel, logcsvwriter, voxel_start = 0):

    def log(r2_train, r2_test, alpha, voxel_id):
        """ just logging stats per fold to a csv file """
        logcsvwriter.writerow([loglabel, voxel_id, alpha, 'training', np.mean(r2_train), np.std(r2_train), np.min(r2_train), np.max(r2_train)])
        logcsvwriter.writerow([loglabel, voxel_id, alpha, 'test', np.mean(r2_test), np.std(r2_test), np.min(r2_test), np.max(r2_test)])

    cdef int n_alpha = len(alphas)
    cdef int n_train = len(fmri_runs)
    cdef int n_voxel = fmri_runs[0].shape[1]
    cdef np.ndarray[double, ndim=3] r2_cv_train_score
    cdef np.ndarray[double, ndim=3] r2_cv_test_score
    cdef double r2_train_score
    cdef double r2_test_score
    cdef double best_alpha 
    cdef int best_alpha_idx

    train = [i for i in range(0, n_train)]
    r2_cv_train_score = np.zeros((n_alpha, n_train, n_voxel))
    r2_cv_test_score = np.zeros((n_alpha, n_train, n_voxel))

    for idx, cv_test_id in enumerate(train):
        fmri_data = np.vstack([fmri_runs[i] for i in train if i != cv_test_id])
        predictors = np.vstack([design_matrices[i] for i in train if i != cv_test_id])
        
        parallel_res = Parallel(n_jobs=-2, prefer="threads")(
            delayed(parallel_fit)(alpha, predictors, fmri_data, design_matrices[cv_test_id].to_numpy(), fmri_runs[cv_test_id])
            for idx2, alpha in enumerate(alphas))

        parallel_res =  np.array(parallel_res)
        r2_cv_train_score[:, idx, :] = parallel_res[:, 0, :]
        r2_cv_test_score[:, idx, :] = parallel_res[:, 1, :]

    for voxel_id in range(n_voxel):
        for idx2, alpha in enumerate(alphas):
            log(r2_cv_train_score[idx2, :, voxel_id], r2_cv_test_score[idx2, :, voxel_id], alpha, voxel_id)
        
    best_alpha_idx = r2_cv_test_score.sum(axis=(1, 2)).argmax()
    best_alpha = alphas[best_alpha_idx]
    
    return r2_cv_train_score[best_alpha_idx, :].mean(axis=(0)), r2_cv_test_score[best_alpha_idx, :].mean(axis=(0)), best_alpha
