import csv
import gc
import getopt
import glob
import os
import os.path as op
import pickle
import shutil
import smtplib
import ssl
import sys
import time
import warnings

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed, dump, load
from numpy.random import randint
from scipy.stats import norm
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

from notifyer import send_mail_log
from model_utils import compute_global_masker
from dim_alpha_search_lib import dim_alpha_search_with_log


def generate_subject_imgs(subject, output_dir, masker):
    if isinstance(subject, list):
        for sub in subject:
            generate_subject_imgs(sub, output_dir, masker)
        return
    
    # TODO DEBUG
    r2_score_file = glob.glob(op.join(output_dir, 'cache', '*'+subject+'*.pkl'))
    assert len(r2_score_file) == 9, '{} has no/corrupted pickled score files.'.format(subject)
    with open(r2_score_file[0], mode='rb') as fi:
        assert subject == pickle.load(fi), '{} has wrong pkl file.'.format(subject)
        alpha_space = pickle.load(fi)
        dimension_space = pickle.load(fi)
        train_score = pickle.load(fi)
        test_score = pickle.load(fi)
    for score_file in r2_score_file[1:]:
        with open(score_file, mode='rb') as fi:
            assert subject == pickle.load(fi), '{} not aligned in file {}.'.format(subject, score_file)
            assert alpha_space == pickle.load(fi), '{} has wrong alpha space in file {}.'.format(subject, score_file)
            assert dimension_space == pickle.load(fi), '{} has wrong dim space in file {}.'.format(subject, score_file)
            train_score += pickle.load(fi)
            test_score += pickle.load(fi)
    test_score /= len(r2_score_file)
    train_score /= len(r2_score_file)

    test_mean = test_score.mean(axis=0)
    test_best_dim_id = test_score.argmax(axis=0)
    test_best_dim_score = test_mean.max(axis=0)
    test_best_dim_best_alpha_id = test_best_dim_score.argmax(axis=0)
    test_best_dim_best_alpha_score = test_best_dim_score.max(axis=0)
    test_best_dim_id_of_best_alpha = test_best_dim_id[test_best_dim_best_alpha_id, range(test_mean.shape[-1])]
    test_best_dim_of_best_alpha = dimension_space[test_best_dim_id_of_best_alpha]
    test_best_dim_best_alpha = alpha_space[test_best_dim_best_alpha_id]

    nib.save(masker.inverse_transform(train_score.max(axis=(0,1))), op.join(
                    output_dir, 'train_{}_r2.nii.gz'.format(subject)))
    nib.save(masker.inverse_transform(test_best_dim_best_alpha_score), op.join(
                    output_dir, 'test_{}_r2.nii.gz'.format(subject)))
    nib.save(masker.inverse_transform(test_best_dim_of_best_alpha), op.join(
                    output_dir, 'test_{}_dim.nii.gz'.format(subject)))
    nib.save(masker.inverse_transform(test_best_dim_best_alpha), op.join(
                    output_dir, 'test_{}_alpha.nii.gz'.format(subject)))
    return


def process_subject(subj_dir, subject, dtx_mat, output_dir, model_name, alpha_space, dimension_space, masker):
    if len(glob.glob(op.join(
                output_dir, 'test_{}_*.nii.gz'.format(subject)))) >= 1:
        print('Skip training {}, using cached file.'.format(subject), flush=True)
        return

    fmri_filenames = sorted(glob.glob(os.path.join(subj_dir, subject, "run*.nii.gz")))
    fmri_runs = [masker.transform(f) for f in fmri_filenames]

    dim_alpha_search_with_log(fmri_runs, dtx_mat, alpha_space, dimension_space, subject, model_name, output_dir, send_mail_log)
    gc.collect()


def main(dmtx_dir, subj_dir, output_dir, model_name, alpha_space, dimension_space):
    if not op.isdir(output_dir):
        os.mkdir(output_dir)

    cache_dir = op.join(output_dir, 'cache')
    if not op.isdir(cache_dir):
        os.mkdir(cache_dir)

    design_files = sorted(glob.glob(op.join(dmtx_dir, 'dmtx_?_ortho.csv')))
    if len(design_files) != 9:
        print("dmtx_?.csv files not found in %s" % dmtx_dir)
        sys.exit(1)
    dtx_mat0 = [pd.read_csv(df) for df in design_files]
    dtx_mat = [((dtx - dtx.mean()) / dtx.std()).values for dtx in dtx_mat0]

    if alpha_space is None:
        alpha_space = [0]
    
    if dimension_space is None:
        dimension_space = [dtx_mat[0].shape[1]]
        
    subjlist = [op.basename(f) for f in glob.glob(op.join(subj_dir, 'sub*'))]
        
    with open(op.join(subj_dir, 'masker.pkl'), mode='rb') as fl:
        masker = pickle.load(fl)

    for idx, subject in enumerate(subjlist):
        msg = """Begin processing {}/{}: {} 
Searching space is:
    alpha : {}
    dim   : {}
""".format(idx, len(subjlist), subject, alpha_space, dimension_space)
        print(msg, flush=True)
        send_mail_log('{} loop'.format(model_name), msg)
        process_subject(subj_dir, subject, dtx_mat, output_dir, model_name, alpha_space, dimension_space, masker)
    
    generate_subject_imgs(subj_dir, output_dir, masker)
    return 


if __name__ == '__main__':
    # parse command line
    try:
        opts, args = getopt.getopt(sys.argv[1:],
                                   "d:s:o:m:",
                                   ["design_matrices=",
                                    "subject_fmri_data=",
                                    "output_dir=", "model_name="])
    except getopt.GetoptError as err:
        print(err)
        sys.exit(2)
        
    for o, a in opts:
        if o in ('-m', '--model_name'):
            model_name = a
        elif o in ('-d', '--design_matrices'):
            dmtx_dir = a
        elif o in ('-s', '--subject_fmri_data'):
            subj_dir = a
        elif o in ('-o', '--output_dir'):
            output_dir = a

    main(dmtx_dir, subj_dir, output_dir, model_name, None, None)
    