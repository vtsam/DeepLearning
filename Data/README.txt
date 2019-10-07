Datasets used for hep-applications of deep learning.
Your code on Google Colab should start with the following commands to load and prepare these datasets.

import os
import h5py
import glob
import numpy as np

! git clone https://github.com/vtsam/jetdata.git

target = np.array([])
features = np.array([])
# We can't use all files on Colab, so we select a few of them.
datafiles = ['jetdata/jetImage_7_100p_30000_40000.h5',
           'jetdata/jetImage_7_100p_60000_70000.h5',
            'jetdata/jetImage_7_100p_50000_60000.h5',
            'jetdata/jetImage_7_100p_10000_20000.h5',
            'jetdata/jetImage_7_100p_0_10000.h5']
# You don't have to worry about the features. If you're interested, you can take a look at the following papers:
* https://arxiv.org/pdf/1709.08705.pdf
* https://arxiv.org/pdf/1804.06913.pdf
for fileIN in datafiles:
    print("Appending %s" %fileIN)
    f = h5py.File(fileIN)
    myFeatures = np.array(f.get("jets")[:,[12, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 48, 52]])
    mytarget = np.array(f.get('jets')[0:,-6:-1])
    features = np.concatenate([features, myFeatures], axis=0) if features.size else myFeatures
    target = np.concatenate([target, mytarget], axis=0) if target.size else mytarget
print(target.shape, features.shape)
