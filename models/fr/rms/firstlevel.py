
# Time-stamp: <2018-04-08 10:16:53 cp983411>

import sys
import getopt
import os
import os.path as op
import glob

import warnings
#warnings.simplefilter(action='ignore', category=FutureWarning)


import pandas as pd
import numpy as np
import nibabel as nib
from joblib import dump, load
from joblib import Parallel, delayed

import matplotlib.pyplot as plt

import nistats
from nistats.first_level_model import FirstLevelModel
from nistats.reporting import plot_design_matrix
from nilearn.plotting import plot_stat_map
from nilearn.plotting import plot_glass_brain


def process_subject(inputpath, subjid, dtx_mat, outputpath):
    subjglm = op.join(outputpath, "cache", "glm_{}".format(subjid))
    subjid = str(subjid)
    if op.isfile(subjglm):
        print('WARNING: Loading already saved model {}'.format(subjglm))
        fmri_glm = load(subjglm)  # the model has already been estimated
    else:  
        # else, we create and estimate it
        print('Creating model for subject %s' % subjid)
        print("Searching for " + op.join(inputpath, subjid, "run?_medn_afw.nii.gz"))
        imgs = sorted(glob.glob(op.join(inputpath, subjid, "run?_medn_afw.nii.gz")))
        if len(imgs) != 9:
            print("WARNING: %s does not have 9 sessions. We skip it." % subjid)
            return
        
        fmri_glm = FirstLevelModel(
            t_r=2.0,
            hrf_model='spm',
            # mask='mask_ICV.nii',
            noise_model='ar1',
            period_cut=128.0,
            smoothing_fwhm=0,
            minimize_memory=True,
            # memory='/mnt/ephemeral/cache',
            memory=None,
            verbose=2,
            n_jobs=-1)
    
        # creating and estimating the model
        fmri_glm = fmri_glm.fit(imgs, design_matrices=dtx_mat)
        # saving it as a pickle object
        dump(fmri_glm, subjglm)

    # creating the maps for each individual predictor
    # this assumes the same predictors for each session
    print('Computing contrasts for subject %s', subjid)
    contrasts = {}
    con_names = [i for i in dtx_mat[0].columns]
    ncon = len(con_names)
    con = np.eye(ncon)
    for i, name in enumerate(con_names):
        contrasts[name] = con[i, :]

    for name, val in contrasts.items():
        z_map = fmri_glm.compute_contrast(val, output_type='z_score')
        eff_map = fmri_glm.compute_contrast(val, output_type='effect_size')
        #std_map = fmri_glm.compute_contrast(val, output_type='stddev')
        nib.save(z_map, op.join(outputpath, '%s_%s_zmap.nii.gz' % (name, subjid)))
        nib.save(eff_map, op.join(outputpath, '%s_%s_effsize.nii.gz'% (name, subjid)))
        display = None
        display = plot_glass_brain(z_map, display_mode='lzry', threshold=3.1, colorbar=True, title=name)
        display.savefig(op.join(outputpath, '%s_%s_glassbrain.png' % (name, subjid)))
        display.close()


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
            output_dir = a

    if not op.isdir(output_dir):
        os.mkdir(output_dir)

    if not op.isdir(op.join(output_dir, 'cache')):
        os.mkdir(op.join(output_dir, 'cache'))

    design_files = sorted(glob.glob(op.join(dmtx_dir, 'dmtx_?.csv')))
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
    # print(subjlist)
    # subjlist = ['disable_sub-07_fm180074']

    if os.getenv('SEQUENTIAL') is not None:
        for s in subjlist:
            process_subject(subj_dir, s, dtx_mat, output_dir)
    else:
        Parallel(n_jobs=1)(delayed(process_subject)(subj_dir, s, dtx_mat, output_dir) for s in subjlist)
