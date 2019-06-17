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

from ridge_all_lib import compute_crossvalidated_r2_all_voxel
from ridge_lib import compute_crossvalidated_r2

sys.path.append("/home/sying/Documents/LePetitPrince_Pallier2018/lpp-scripts3/models")
from model_utils import compute_global_masker


def get_design_matrices(rootdir):
    matrices = []
    for j in range(1, 10):
        data = pd.read_csv(os.path.join(rootdir, 'dmtx_{}_ortho.csv'.format(j)), header=1)
        dmtx = data[1:]
        const = np.ones((dmtx.shape[0], 1))
        dmtx = np.hstack((dmtx, const))
        matrices.append(dmtx)
    return matrices

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
        #print('Run %d. Correlation matrix:' % (i + 1))
        #print(np.round(np.corrcoef(d.T), 5))
        d['constant'] = np.ones(len(d))

    subjlist = [op.basename(f) for f in glob.glob(op.join(subj_dir, 'sub*'))]
   
    # if os.getenv('SEQUENTIAL') is not None:
    #     for s in subjlist:
    #         process_subject(subj_dir, s, dtx_mat, output_dir)
    # else:
    #     Parallel(n_jobs=1)(delayed(process_subject)(subj_dir, s, dtx_mat, output_dir) for s in subjlist)

    search_ub = 10
    search_lb = 3
    alphas = [0] + list(np.logspace(search_lb, search_ub, 20))
    # alphas = [0]
    # New mask / 0 + -10~3 gives 0 + 10*3

    print('Main: computing global masker.')
    if op.isfile(output_dir + 'cache/masker.pkl'):
        print('Main: global masker retrieved from cache.')
        fl = open(output_dir + 'cache/masker.pkl', mode='rb')
        masker = pickle.load(fl)
        fl.close()
    else:
        masker = compute_global_masker(subj_dir, subjlist)
        fl = open(output_dir + 'cache/masker.pkl', mode='wb')
        pickle.dump(masker, fl, protocol=-1)
        fl.close()
    
    logcsvwriter = csv.writer(open(output_dir + "run_newmask.log", "a+"))

    # for subject in np.array(subjlist)[np.random.choice(len(subjlist), 5, replace=False)]:
    for subject in subjlist:

        loglabel = subject

        print('Main: computing for subject', subject, 'search space :', search_lb, search_ub, flush=True)
        if len(glob.glob(output_dir + 'test_{}_*.nii.gz'.format(subject))) > 0:
            print('Main: using existing results', subject)
            continue

            
        fmri_filenames = sorted(glob.glob(os.path.join(subj_dir, loglabel, "run*.nii.gz")))
        #print("flag")
        #np.seterr(all='raise')
        fmri_runs = [masker.transform(f) for f in fmri_filenames]
        ## np.sqrt exception in transform, standarization
        # for f in fmri_filenames:
        #     try:
        #         masker.transform(f)
        #     except RuntimeWarning:
        #         print(f)
        #     except FloatingPointError:
        #         print(f)
            
        # continue
        
        r2train, r2test, best_alpha = compute_crossvalidated_r2_all_voxel(fmri_runs, dtx_mat, alphas, loglabel, logcsvwriter, 0)
    
        nib.save(masker.inverse_transform(r2train), 
                output_dir + 'train_{}_{}.nii.gz'.format(subject, best_alpha))

        nib.save(masker.inverse_transform(r2test), 
                output_dir + 'test_{}_{}.nii.gz'.format(subject, best_alpha))

        print(subject, 'best alpha', best_alpha, flush=True)

        # best_alpha_base = np.log10(best_alpha)
        # if best_alpha_base > (search_lb + search_ub) / 2:
        #     search_lb = (search_lb + search_ub) / 2
        # else:
        #     search_ub = (search_lb + search_ub) / 2
        # alphas = np.logspace(search_lb, search_ub, 20)

        # nib.save(masker.inverse_transform(best_alpha), 
        #         output_dir + 'alpha_{}.nii.gz'.format(subject))

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
