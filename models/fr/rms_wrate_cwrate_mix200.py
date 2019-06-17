import numpy as np

# Set the environment variables when executing a model from python

root_dir = '/home/sying/Documents/LePetitPrince_Pallier2018/lpp-scripts3/'
lingua = 'fr'
model_name = 'rms-wrate-cwrate-mix200'

base_regressors = ['rms', 'wrate', 'cwrate']
base_dim = len(base_regressors)

embedding_dim = 200
embedding_base_name = 'mix_dim200_voc24519_d'
embedding_regressors = [embedding_base_name+str(i) for i in range(1, embedding_dim+1)]

regs = base_regressors + embedding_regressors

subject_fmri_data = "/home/sying/Documents/LePetitPrince_Pallier2018/french-mri-data"

alpha_space = np.array([0] + list(np.logspace(0, 2.4, 5)
                        ) + list(np.logspace(2.5, 4, 20)))
dimension_space = np.array(
    list(range(1, 3)) + list(range(3, 26, 2)) + list(range(25, 106, 10)))

# local_dict = locals()
print("Imported configuration for {}/{}:".format(model_name, lingua))
# for key, val in local_dict.items():
#     print("{:>10} has value {}".format(key, val))

# print(locals())