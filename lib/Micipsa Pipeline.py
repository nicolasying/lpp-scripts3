import csv
import getopt
import glob
import os
import os.path as op
import pickle
import smtplib
import ssl
import sys
import gc
import warnings
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pylab as plt
import seaborn as sns
from numpy.random import randint
from scipy.stats import norm
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, LeaveOneGroupOut
from sklearn.multioutput import MultiOutputRegressor

sys.path.append("/home/sying/Documents/LePetitPrince_Pallier2018/lpp-scripts3/models")

import nibabel as nib
import nistats
#from ridge_all_lib import compute_crossvalidated_r2_all_voxel
from dim_alpha_search_lib import dim_alpha_search
from joblib import Parallel, delayed, dump, load
#import GLM_dim_search_lib
#from lasso_lib import     compute_crossvalidated_r2_all_voxel as     compute_crossvalidated_r2_all_voxel_lasso
from model_utils import compute_global_masker
from notifyer import send_mail_log
from nilearn.image import coord_transform, math_img, mean_img, threshold_img
from nilearn.input_data import MultiNiftiMasker
from nilearn.masking import apply_mask, compute_multi_epi_mask
from nilearn.plotting import plot_glass_brain, plot_stat_map
from nistats.first_level_model import FirstLevelModel
from nistats.reporting import plot_design_matrix
from nistats.second_level_model import SecondLevelModel

from firstlevel import main as first_level_analysis
# from xgboost import XGBRegressor

warnings.simplefilter(action='ignore', category=FutureWarning)

# # Global Config

# In[2]:


comparison_name = 'rms-wrate-cwrate-dim_alpha_search'


# In[3]:


lingua = 'fr'
root_dir = '/home/sying/Documents/LePetitPrince_Pallier2018/lpp-scripts3/'
subjects_fmri_data = '/home/sying/Documents/LePetitPrince_Pallier2018/french-mri-data/'

with open(op.join(root_dir, 'outputs', 'masker', lingua, 'masker.pkl'), mode='rb') as fl:
    masker = pickle.load(fl)

subjlist = [op.basename(f) for f in glob.glob(op.join(subjects_fmri_data, 'sub*'))]
n_subject = len(subjlist)

for model in ['rms-wrate-sim103','rms-wrate-asn200', 'rms-wrate-cwrate-sig200', 'rms-wrate-cwrate-asn200'
     'rms-wrate-cwrate-sim103', 'rms-wrate-cwrate-mix200']:
    print('Begin training {}.'.format(model), flush=True)

    design_matrices_dir = op.join(root_dir, 'outputs', 'design-matrices', lingua, model)
    first_level_results = op.join(root_dir, 'outputs', 'results-indiv-ridge', lingua, model)
    output_dir = first_level_results

    alpha_space = np.array([0] + list(np.logspace(0, 2.4, 5)) + list(np.logspace(2.5, 4, 20)) + list(np.logspace(4.1, 5, 3)))
    dimension_space = np.array(list(range(1,3)) + list(range(3, 26, 2)) + list(range(25, 206, 10)))
    if model == 'rms-wrate-sim103':
        dimension_space = np.array(list(range(1, 3)) + list(range(3, 26, 2)) + list(range(25, 106, 10)))
    first_level_analysis(design_matrices_dir, subjects_fmri_data, output_dir, model, alpha_space, dimension_space)
