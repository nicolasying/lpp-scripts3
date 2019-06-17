# IMPORTS
from nilearn.masking import compute_multi_epi_mask
from nilearn.masking import apply_mask
from nilearn.input_data import MultiNiftiMasker

import glob
import getopt
import os
import os.path as op
import sys
import nibabel as nib
import numpy as np
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

def compute_global_masker(rootdir, subjects):
    imgs = glob.glob(os.path.join(rootdir, 'sub*', "run*.nii.gz"))
    if len(imgs) > 0:
        global_mask = compute_multi_epi_mask(imgs, n_jobs=-1)
    # masks = [compute_epi_mask(glob.glob(os.path.join(rootdir, s, "run*.nii.gz"))) for s in subjects]
    # global_mask = math_img('img>0.5', img=mean_img(masks))
        masker = MultiNiftiMasker(global_mask, detrend=True, standardize=True)
        masker.fit()
        return masker
    else:
        print('No fmri data found', os.path.join(rootdir, 'sub_*', "run*.nii.gz"))
        exit(0)