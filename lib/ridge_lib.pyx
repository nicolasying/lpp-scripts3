# IMPORTS
from nilearn.masking import compute_epi_mask
from nilearn.masking import apply_mask
from nilearn.input_data import MultiNiftiMasker

import glob
import getopt
import os
import os.path as op
import sys
import nibabel as nib
import numpy as np
cimport numpy as np
from numpy.random import randint
from joblib import dump, load
from joblib import Parallel, delayed

import csv
import pandas as pd
import seaborn as sns

from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, LeaveOneGroupOut
from sklearn.metrics import r2_score

from nilearn.image import math_img, mean_img, threshold_img
from nilearn.plotting import plot_glass_brain
from nilearn.image import coord_transform
import pylab as plt
from joblib import Parallel, delayed
import pickle


def clean_rscorer(estimator, np.ndarray X, np.ndarray y):
    cdef double x
    x = r2_score(y, estimator.predict(X))
    return 0 if (x < 0.0 or x >= .99) else x
    
def parallel_fit(float alpha, np.ndarray x_train, np.ndarray y_train, np.ndarray x_test, np.ndarray y_test):
    model = Ridge(alpha=alpha, fit_intercept=False, normalize=False).fit(x_train, y_train)
    # print(r2_score(y_train, model.predict(x_train)), r2_score(y_test, model.predict(x_test)))
    return clean_rscorer(model, x_train, y_train), clean_rscorer(model, x_test, y_test)


def compute_crossvalidated_r2(fmri_runs, design_matrices, alphas, loglabel, logcsvwriter, voxel_start = 0):

    def log(r2_train, r2_test, alpha, voxel_id):
        """ just logging stats per fold to a csv file """
        logcsvwriter.writerow([loglabel, voxel_id, alpha, 'training', np.mean(r2_train), np.std(r2_train), np.min(r2_train), np.max(r2_train)])
        logcsvwriter.writerow([loglabel, voxel_id, alpha, 'test', np.mean(r2_test), np.std(r2_test), np.min(r2_test), np.max(r2_test)])

    # r2_train = None  # array to contain the r2 values (1 row per fold, 1 column per voxel)
    # r2_test = None

    cdef np.ndarray[double, ndim=1] r2_train_score = np.zeros(fmri_runs[0].shape[1])
    cdef np.ndarray[double, ndim=1] r2_test_score = np.zeros(fmri_runs[0].shape[1])
    cdef np.ndarray[double, ndim=1] best_alpha = np.zeros(fmri_runs[0].shape[1])

    cdef int voxel_id
    cdef int test_id
    cdef int n_alpha = len(alphas)
    cdef int n_train = len(fmri_runs) - 1
    cdef np.ndarray[double, ndim=2] r2_cv_train_score
    cdef np.ndarray[double, ndim=2] r2_cv_test_score

    for voxel_id in range(voxel_start, fmri_runs[0].shape[1]):
        test_id = randint(0, len(fmri_runs))
        train = [i for i in range(0, len(fmri_runs)) if i != test_id]
        print(loglabel, 'voxel', voxel_id, 'in', fmri_runs[0].shape[1], 'testing with session #', test_id)
        r2_cv_train_score = np.zeros((n_alpha, n_train))
        r2_cv_test_score = np.zeros((n_alpha, n_train))

        for idx, cv_test_id in enumerate(train):
            fmri_data = np.vstack([fmri_runs[i][:, voxel_id:voxel_id+1] for i in train if i != cv_test_id])
            predictors = np.vstack([design_matrices[i] for i in train if i != cv_test_id])
            # for alpha, idx2 in enumerate(alphas):
            #     model = Ridge(alpha=alpha, fit_intercept=False, normalize=False).fit(predictors, fmri_data)
            #     r2_cv_train_score[idx2, idx] = clean_rscorer(model, design_matrices[cv_test_id], fmri_runs[cv_test_id][:, voxel_id])
            
            parallel_res = Parallel(n_jobs=-2, prefer="threads")(
                delayed(parallel_fit)(alpha, predictors, fmri_data, design_matrices[cv_test_id].to_numpy(), fmri_runs[cv_test_id][:, voxel_id])
                for idx2, alpha in enumerate(alphas))
            # print(np.array(parallel_res).shape)
            r2_cv_train_score[:, idx], r2_cv_test_score[:, idx] = np.array(parallel_res).T

        for idx2, alpha in enumerate(alphas):
            log(r2_cv_train_score[idx2, :], r2_cv_test_score[idx2, :], alpha, voxel_id)
            
        if np.any(r2_cv_test_score.sum(axis=1) != 0):
            best_alpha[voxel_id] = alphas[r2_cv_test_score.sum(axis=1).argmax()]
        else: 
            best_alpha[voxel_id] = alphas[r2_cv_train_score.sum(axis=1).argmax()]

        fmri_data = np.vstack([fmri_runs[i][:, voxel_id:voxel_id+1] for i in train])
        predictors = np.vstack([design_matrices[i] for i in train])
        model = Ridge(alpha=best_alpha[voxel_id], fit_intercept=False, normalize=False).fit(predictors, fmri_data)

        r2_train_score[voxel_id] = clean_rscorer(model, predictors, fmri_data)
        r2_test_score[voxel_id] = clean_rscorer(model, design_matrices[test_id].to_numpy(), fmri_runs[test_id][:, voxel_id])
        logcsvwriter.writerow([loglabel, voxel_id, best_alpha[voxel_id], 'best_alpha', r2_train_score[voxel_id], 0, r2_train_score[voxel_id], r2_test_score[voxel_id]])
        
        # log(r2_train_score[voxel_id], r2_test_score[voxel_id], best_alpha[voxel_id])
    return r2_train_score, r2_test_score, best_alpha
    