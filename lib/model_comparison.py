import getopt
import glob
import os
import os.path as op
import pickle
import sys
import warnings

import matplotlib.pyplot as plt
import nibabel as nib
import nistats
import numpy as np
import pandas as pd
from joblib import Parallel, delayed, dump, load
from nilearn.image import math_img
from nilearn.plotting import plot_glass_brain, plot_stat_map
from nistats.first_level_model import FirstLevelModel
from nistats.reporting import plot_design_matrix
from nistats.second_level_model import SecondLevelModel
from scipy.stats import norm

warnings.simplefilter(action='ignore', category=FutureWarning)

# Global Config
comparison_name = 'GLM_4_10_10_oldmask'

subject_path = '/home/sying/Documents/LePetitPrince_Pallier2018/french-mri-data/'
model_parent_path = '/home/sying/Documents/LePetitPrince_Pallier2018/lpp-scripts3/models/fr'
input_dir = '/home/sying/Documents/LePetitPrince_Pallier2018/lpp-scripts3/outputs/results-indiv/fr'
output_dir = '/home/sying/Documents/LePetitPrince_Pallier2018/lpp-scripts3/outputs/cross-model-comp'

base_model_name = 'rms-wrate-cwrate'
sim_model_name = 'rms-wrate-cwrate-sim10'
asn_model_name = 'rms-wrate-cwrate-asn10'


def comp_parallel(sub, base_model_name=base_model_name, sim_model_name=sim_model_name, asn_model_name=asn_model_name, 
                    output_dir=op.join(output_dir, comparison_name), input_dir=input_dir, file_identifier='test'):
    if not op.isdir(output_dir):
        os.mkdir(output_dir)

    base_imgs = glob.glob(op.join(input_dir, base_model_name,'{}_{}*.nii.gz'.format(file_identifier, sub)))
    sim_imgs = glob.glob(op.join(input_dir, sim_model_name,'{}_{}*.nii.gz'.format(file_identifier, sub)))
    asn_imgs = glob.glob(op.join(input_dir, asn_model_name,'{}_{}*.nii.gz'.format(file_identifier, sub)))

    assert len(base_imgs) == 1, sub
    assert len(sim_imgs) == 1,  sub
    assert len(asn_imgs) == 1,  sub
    base_img = nib.load(base_imgs[0])
    sim_img = nib.load(sim_imgs[0])
    asn_img = nib.load(asn_imgs[0])
    name = 'sim-base'
    sim_contrast = math_img("sim - base", sim=sim_img, base=base_img)
    display = None
    display = plot_glass_brain(sim_contrast, display_mode='lzry', colorbar=True, title=name, plot_abs=False, cmap=plt.cm.coolwarm)
    display.savefig(op.join(output_dir, '%s_%s_glassbrain.png' % (name, sub)))
    display.close()
    nib.save(sim_contrast, op.join(output_dir, '%s_%s_r2.nii.gz' % (name, sub)))
    name = 'asn-base'
    asn_contrast = math_img("asn - base", asn=asn_img, base=base_img)
    display = None
    display = plot_glass_brain(asn_contrast, display_mode='lzry', colorbar=True, title=name, plot_abs=False, cmap=plt.cm.coolwarm)
    display.savefig(op.join(output_dir, '%s_%s_glassbrain.png' % (name, sub)))
    display.close()
    nib.save(asn_contrast, op.join(output_dir, '%s_%s_r2.nii.gz' % (name, sub)))
    name = 'sim-asn'
    sim_asn_contrast = math_img("sim - asn", asn=asn_contrast, sim=sim_contrast)
    display = None
    display = plot_glass_brain(sim_asn_contrast, display_mode='lzry', colorbar=True, title=name, plot_abs=False, cmap=plt.cm.coolwarm)
    display.savefig(op.join(output_dir, '%s_%s_glassbrain.png' % (name, sub)))
    display.close()
    nib.save(sim_asn_contrast, op.join(output_dir, '%s_%s_r2.nii.gz' % (name, sub)))




if __name__ == "__main__":
    subjlist = [op.basename(f) for f in glob.glob(op.join(subject_path, 'sub*'))]

    base_imgs = glob.glob(op.join(input_dir, base_model_name,'test_sub-*.nii.gz'))
    sim_imgs = glob.glob(op.join(input_dir, sim_model_name,'test_sub-*.nii.gz'))
    asn_imgs = glob.glob(op.join(input_dir, asn_model_name,'test_sub-*.nii.gz'))

    if len(base_imgs) != len(subjlist):
        exit(0)
    if len(sim_imgs) != len(subjlist):
        exit(0)
    if len(asn_imgs) != len(subjlist):
        exit(0)

    Parallel(n_jobs=-2)(delayed(comp_parallel)(subject) for subject in subjlist)


# with open('./rms-wrate-cwrate/cache/masker.pkl', mode='rb') as file:
#     masker = pickle.load(file)
    
# sub = subjlist[0]
# base_imgs = glob.glob(op.join('rms-wrate-cwrate','test_{}*.nii.gz'.format(sub)))
# sim_imgs = glob.glob(op.join('rms-wrate-cwrate-sim103/GS_3_4_20','test_{}*.nii.gz'.format(sub)))
# asn_imgs = glob.glob(op.join('rms-wrate-cwrate-asn200','test_{}*.nii.gz'.format(sub)))
# assert len(base_imgs) == 1, sub
# assert len(sim_imgs) == 1,  sub
# assert len(asn_imgs) == 1,  sub
# base_img = nib.load(base_imgs[0])
# sim_img = nib.load(sim_imgs[0])
# asn_img = nib.load(asn_imgs[0])

# base_scores = masker.transform(base_img)
# sim_scores = masker.transform(sim_img)
# asn_scores = masker.transform(asn_img)

# f, ax= plt.subplots(2, 2, figsize=(15,15))

# ax[0, 0].scatter(base_scores[0], sim_scores[0], s=1)
# ax[0, 0].set_xlabel('base')
# ax[0, 0].set_ylabel('sim')
# ax[0, 0].plot(np.linspace(base_scores.min(), base_scores.max(), 1000), np.linspace(base_scores.min(), base_scores.max(), 1000), c='r')

# ax[0, 1].scatter(base_scores[0], asn_scores[0], s=1)
# ax[0, 1].set_xlabel('base')
# ax[0, 1].set_ylabel('asn')
# ax[0, 1].plot(np.linspace(base_scores.min(), base_scores.max(), 1000), np.linspace(base_scores.min(), base_scores.max(), 1000), c='r')

# ax[1, 0].scatter(sim_scores[0], asn_scores[0], s=1)
# ax[1, 0].set_xlabel('sim')
# ax[1, 0].set_ylabel('asn')
# ax[1, 0].plot(np.linspace(sim_scores.min(), sim_scores.max(), 1000), np.linspace(sim_scores.min(), sim_scores.max(), 1000), c='r')

# ax[1, 1].hist((base_scores[0], sim_scores[0], asn_scores[0]), bins=20, label=('base', 'sim', 'asn'))
# ax[1, 1].legend()

# f.show()

# def create_one_sample_t_test(name, maps, output_dir, smoothing_fwhm=8.0):
#     if not op.isdir(output_dir):
#         op.mkdir(output_dir)

#     model = SecondLevelModel(smoothing_fwhm=smoothing_fwhm)
#     design_matrix = pd.DataFrame([1] * len(maps),
#                                  columns=['intercept'])
#     model = model.fit(maps,
#                       design_matrix=design_matrix)
#     z_map = model.compute_contrast(output_type='z_score')
#     nib.save(z_map, op.join(output_dir, "{}_group_zmap.nii.gz".format(name)))

#     for p_val in np.linspace(-17, -3, 20):
#         z_th = norm.isf(10 ** p_val)
#         # z_th = 3.1
#         display = plot_glass_brain(
#             z_map, threshold=z_th,
#             colorbar=True,
#             plot_abs=False,
#             display_mode='lzry',
#             title=name)
#         display.savefig(op.join(output_dir, "{}_{}_group_zmap.png".format(name, p_val)))
#         display.close()

#         for con in ['sim-base', 'asn-base', 'sim-asn']:
#     mask = op.join('%s_sub-*.nii.gz' % con)
#     maps = glob.glob(mask)
#     if maps == []:
#         print("Warning: %s : no such files" % mask)
#     else:
#         create_one_sample_t_test(con, maps, './')