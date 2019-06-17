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


from lib.notifyer import send_mail_log
from lib.model_utils import compute_global_masker
from lib.dim_alpha_search_lib import dim_alpha_search_with_log
from lib import generate_regressors
# from lib import dim_alpha_search_lib

# from lib import generate_regressors

from models.fr import rms_wrate_cwrate_mix200

root_dir = '/home/sying/Documents/LePetitPrince_Pallier2018/lpp-scripts3/'
lingua = 'fr'
model_name = 'rms-wrate-cwrate-sig200'

base_regressors = ['rms', 'wrate', 'cwrate']
base_dim = len(base_regressors)

embedding_dim = 200
embedding_base_name = 'sig_dim200_voc24519_d'
embedding_regressors = [embedding_base_name+str(i) for i in range(1, embedding_dim+1)]

regs = base_regressors + embedding_regressors

subject_fmri_data = "/home/sying/Documents/LePetitPrince_Pallier2018/french-mri-data"

alpha_space = np.array([0] + list(np.logspace(0, 2.4, 5)
                        ) + list(np.logspace(2.5, 4, 20)))
dimension_space = np.array(
    list(range(1, 3)) + list(range(3, 26, 2)) + list(range(25, 106, 10)))

# Config model environment
onset_dir = op.join(root_dir, 'inputs', 'onsets', lingua)
regs_dir = op.join(root_dir, 'outputs', 'regressors', lingua)
design_matrices_dir = op.join(root_dir, 'outputs', 'design-matrices', lingua, model_name)
first_level_results = op.join(root_dir, 'outputs', 'results-indiv-ridge', lingua, model_name)
group_results = op.join(root_dir, 'outputs', 'results-group-ridge', lingua, model_name)

# generate_regressors.main(lingua, regs_dir, onset_dir, False, regs, None, None)
