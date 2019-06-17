# IMPORTS
from nilearn.masking import compute_epi_mask
from nilearn.masking import apply_mask
from nilearn.input_data import MultiNiftiMasker

import glob
import os
import os.path as op
import sys
import nibabel as nib
import numpy as np
from numpy.random import randint

import csv
import pandas as pd
import seaborn as sns

from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, LeaveOneGroupOut
from sklearn.metrics import r2_score

from nilearn.image import math_img, mean_img, threshold_img
from nilearn.plotting import plot_glass_brain
from nilearn.image import coord_transform
import pylab as plt
from joblib import Parallel, delayed




def get_design_matrices(rootdir):
    matrices = []
    for j in range(1, 10):
        data = pd.read_csv(os.path.join(rootdir, 'dmtx_{}_ortho.csv'.format(j)), header=1)
        dmtx = data[1:]
        const = np.ones((dmtx.shape[0], 1))
        dmtx = np.hstack((dmtx, const))
        matrices.append(dmtx)
    return matrices

def compute_global_masker(rootdir, subjects):
    masks = [compute_epi_mask(glob.glob(os.path.join(rootdir, s, "run*.nii.gz"))) for s in subjects]
    global_mask = math_img('img>0.5', img=mean_img(masks))
    masker = MultiNiftiMasker(global_mask, detrend=True, standardize=True)
    masker.fit()
    return masker

def clean_rscores(rscores, r2min, r2max): # remove negative values (worse than the constant model) and values too good to be true (e.g. voxels without variation)
    return np.array([0 if (x < r2min or x >= r2max) else x for x in rscores])
   
def clean_rscorer(estimator, X, y):
    x = r2_score(y, estimator.predict(X), multioutput='raw_values')
    return 0 if (x < 0.0 or x >= .99) else x
    
def compute_crossvalidated_r2(fmri_runs, design_matrices, alphas, loglabel, logcsvwriter):
    
    def log(r2_train, r2_test):
        """ just logging stats per fold to a csv file """
        logcsvwriter.writerow([loglabel, alpha, 'training', np.mean(r2_train), np.std(r2_train), np.min(r2_train), np.max(r2_train)])
        logcsvwriter.writerow([loglabel, alpha, 'test', np.mean(r2_test), np.std(r2_test), np.min(r2_test), np.max(r2_test)])
    
    # r2_train = None  # array to contain the r2 values (1 row per fold, 1 column per voxel)
    # r2_test = None
    voxelwise_alpha = np.zeros(fmri_runs[0].shape[1])
    test_id = randint(0, 10)
    train = [i for i in range(0, 10) if i != test_id]

    r2_train_score = np.zeros(fmri_runs[0].shape[1])
    r2_test_score = np.zeros(fmri_runs[0].shape[1])
    best_alpha = np.zeros(fmri_runs[0].shape[1])

    for voxel_id in range(fmri_runs[0].shape[1]):
        r2_cv_train_score = np.zeros(len(alphas), (len(train)))

        for cv_test_id, idx in enumerate(train):
            fmri_data = np.vstack([fmri_runs[i][:, voxel_id] for i in train if i != cv_test_id])
            predictors = np.vstack([design_matrices[i] for i in train if i != cv_test_id])
            for alpha, idx2 in enumerate(alphas):
                model = Ridge(alphas=alpha, fit_intercept=False, normalize=False).fit(predictors, fmri_data)
                r2_cv_train_score[idx2, idx] = clean_rscorer(model, design_matrices[cv_test_id], fmri_runs[cv_test_id][:, voxel_id])
        
        best_alpha[voxel_id] = alphas[r2_cv_train_score.sum(axis=1).argmax()]

        fmri_data = np.vstack([fmri_runs[i][:, voxel_id] for i in train)
        predictors = np.vstack([design_matrices[i] for i in train)
        model = Ridge(alphas=best_alpha[voxel_id], fit_intercept=False, normalize=False).fit(predictors, fmri_data)

        r2_train_score[voxel_id] = clean_rscorer(model, predictors, fmri_data)
        r2_test_score[voxel_id] = clean_rscorer(model, design_matrices[test_id], fmri_runs[test_id][:, voxel_id])
        log(r2_train_score[voxel_id], r2_test_score[voxel_id])
    return r2_train_score, r2_test_score, best_alpha
    
    

if __name__ == '__main__':
    # parse command line
    try:
        opts, args = getopt.getopt(sys.argv[1:],
                                   "d:s:o:",
                                   ["design_matrices=",
                                    "subject_fmri_data=",
                                    "output_dir="])
    except getopt.GetoptError as err:
        print(err)
        sys.exit(2)
        
    for o, a in opts:
        if o in ('-d', '--design_matrices'):
            dmtx_dir = a
        if o in ('-s', '--subject_fmri_data'):
            subj_dir = a
        elif o in ('-o', '--output_dir'):
            output_dir = a + '/'

    if not op.isdir(output_dir):
        os.mkdir(output_dir)

    if not op.isdir(op.join(output_dir, 'cache')):
        os.mkdir(op.join(output_dir, 'cache'))

    design_files = sorted(glob.glob(op.join(dmtx_dir, 'dmtx_?_ortho.csv')))
    if len(design_files) != 9:
        print("dmtx_?.csv files not found in %s" % dmtx_dir)
        sys.exit(1)
    dtx_mat0 = [pd.read_csv(df) for df in design_files]
    dtx_mat = [((dtx - dtx.mean()) / dtx.std()) for dtx in dtx_mat0]
    for i, d in enumerate(dtx_mat):
        plt.plot(d)
        plt.savefig(op.join(output_dir, 'dtx_plot_%s.png' % str(i + 1)))
        plt.close()
        print('Run %d. Correlation matrix:' % (i + 1))
        print(np.round(np.corrcoef(d.T), 5))
        d['constant'] = np.ones(len(d))

    subjlist = [op.basename(f) for f in glob.glob(op.join(subj_dir, 'sub*'))]
   
    # if os.getenv('SEQUENTIAL') is not None:
    #     for s in subjlist:
    #         process_subject(subj_dir, s, dtx_mat, output_dir)
    # else:
    #     Parallel(n_jobs=1)(delayed(process_subject)(subj_dir, s, dtx_mat, output_dir) for s in subjlist)

    alphas = np.logspace(-3, 3, 50)
    
    masker = compute_global_masker(subj_dir, subjlist)
    

    logcsvwriter = csv.writer(open("test.log", "a+"))

    for subject in subjlist:

        loglabel = subject
        fmri_filenames = sorted(glob.glob(os.path.join(subj_dir, loglabel, "run*.nii.gz")))
        fmri_runs = [masker.transform(f) for f in fmri_filenames]
        r2train, r2test, best_alpha = compute_crossvalidated_r2(fmri_runs, dtx_mat, alphas, loglabel, logcsvwriter)
    
        nib.save(masker.inverse_transform(r2train), 
                output_dir + 'train_{}.nii.gz'.format(subject))

        nib.save(masker.inverse_transform(r2test), 
                output_dir + 'test_{}.nii.gz'.format(subject))

        nib.save(masker.inverse_transform(best_alpha), 
                output_dir + 'alpha_{}.nii.gz'.format(subject))

            # for reg in range(matrices[0].shape[1]):
            #         """ remove one predictor from the design matrix in test to compare it with the full model """ 
            #         new_design_matrices = [np.delete(mtx, reg, 1) for mtx in matrices]
            #         r2train_dropped, r2test_dropped = compute_crossvalidated_r2(fmri_runs, new_design_matrices, alpha, loglabel, logcsvwriter)
                    
            #         nib.save(masker.inverse_transform(r2train_dropped), 
            #         output_dir + 'alpha_{}_train_{}_{:03}.nii'.format(alpha,reg,subject))

            #         nib.save(masker.inverse_transform(r2test_dropped), 
            #         output_dir + 'alpha_{}_test_{}_{:03}.nii'.format(alpha,reg,subject))

            #         r2train_difference = r2train - r2train_dropped
            #         nib.save(masker.inverse_transform(r2train_difference), 
            #         output_dir + 'alpha_{}_train_without_{}_{:03}.nii'.format(alpha,reg,subject))
                    
            #         r2test_difference = r2test - r2test_dropped
            #         nib.save(masker.inverse_transform(r2test_difference), 
            #         output_dir + 'alpha_{}_test_without_{}_{:03}.nii'.format(alpha,reg,subject))
"""        
       
        img=mean_img(thresholded_score_map_img)
                 display = None
                 display = plot_glass_brain(img, display_mode='lzry', threshold=3.1, colorbar=True, title='{} for alpha = {}, subject{:03}'.format(reg,i,n))
                 display.savefig('alpha{}_{}_only_test_{:03}.png'.format(i,reg,n))
                 display.close()
  
"""
