#! /usr//bin/env python
# Time-stamp: <2017-07-19 22:44:53 cp983411>


""" read the design matrices dmt_*.csv and perform a sequential orthogonalization of the variables """

import sys
import getopt
import os
import glob
import os.path as op
import numpy as np
import numpy.linalg as npl
from numpy import (corrcoef, around, array, dot, identity, mean)
from numpy import column_stack as cbind
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
sys.path.append("/home/sying/Documents/LePetitPrince_Pallier2018/lpp-scripts3/models")

from orthonormalize_lib import main_parallel

# def ortho_proj(Y, M):
#     """ returns the orthogonal component of Y to the space spanned by M and the constant vector 1 """
#     if M.ndim == 1:   # M is a vector but needs to be a 2-D matrix
#         M = M[:, np.newaxis]
#     I = np.ones(len(M))
#     I = I[:, np.newaxis]
#     M2 = np.hstack((I, M))  # adding the constant 
#     betas,_,_,_ = npl.lstsq(M2, Y)
#     Xc = np.dot(M2, betas)  # colinear component "residuals"
#     Xo = Y - Xc
#     return Xo

# def main_parallel(r, f):
#     print("Run #%d:" % r)
#     df = pd.read_csv(f)
#     M1 = df.values.T
#     print(around(corrcoef(M1), 2))

#     display = sns.pairplot(df)
#     fn, ext = op.splitext(op.basename(f))
#     display.savefig(op.join(output_dir, fn + '_nonortho.png'))

#     # X1 = df.rms - mean(df.rms)
#     # X2 = ortho_proj(df.wrate, df.rms)
#     # X3 = ortho_proj(df.cwrate, cbind((df.rms, df.wrate)))
#     # M2 = cbind((X1, X2, X3, X4, df[df.columns[4:]]))
#     # newdf = pd.DataFrame(data=M2,
#     #                     columns=['rms', 'wrate0', 'cwrate0', 'freqO'])
#     # newdf = df
#     # newdf.loc[:, 'rms'] = X1
#     # newdf.loc[:, 'wrate'] = X2
#     # newdf.loc[:, 'cwrate'] = X3
    
#     df.loc[:, df.columns[0]] = df.loc[:, df.columns[0]] - mean(df.loc[:, df.columns[0]])
#     for idx, column in enumerate(df.columns[1:]):
#         print("Run #%d: column %d" % (r, idx))
#         df.loc[:, column] =  ortho_proj(df.loc[:, column], df.loc[:, df.columns[:idx+1]])

#     fname, ext  = op.splitext(op.basename(f))
#     newfname = op.join(output_dir, fname + '_ortho' + ext)
#     df.to_csv(newfname, index=False)
#     display = sns.pairplot(df)
#     display.savefig(op.join(output_dir, fn + '_ortho.png'))
#     M2 = df.values
#     print(around(corrcoef(M2.T), 2))
#     plt.close('all')

if __name__ == '__main__':
    data_dir = '.'
    output_dir = '.'
    
    # parse command line to change default
    try:
        opts, args = getopt.getopt(sys.argv[1:],
                                   "d:o:w:",
                                   ["design_matrices=", "output_dir=","--no-overwrite"])
    except getopt.GetoptError as err:
        print(err)
        sys.exit(2)
    
    overwrite = True
    for o, a in opts:
        if o in ('-d', '--design_matrices'):
            data_dir = a
        elif o in ('-o', '--output_dir'):
            output_dir = a
        elif o == '--no-overwrite':
            overwrite = False
    
    filter = op.join(data_dir, 'dmtx_?.csv')
    dfiles = glob.glob(filter)
    if len(dfiles) == 0:
        print("Cannot find files "+ filter)
        sys.exit(2)
        
    if not op.isdir(output_dir):
        os.mkdir(output_dir)
        
    if os.getenv('SEQUENTIAL') is not None:
        for r, f in enumerate(dfiles):
            main_parallel(r, f, output_dir)
    else:
        Parallel(n_jobs=-2)(delayed(main_parallel)(r, f, output_dir) for r, f in enumerate(dfiles))
