import os
import h5py
from NN.Inputs import NNSet_from_fastMRI
from utils.fastMRI_utils import image_from_kspace

path = '/mnt/d/fastMRI_DATA/singlecoil_val/'

files = [f for f in os.listdir(path) if (os.path.isfile(os.path.join(path,f)) and ('.h5' in f))]


inputs = []
outputs = []
for f in files:
    h5f = h5py.File(os.path.join(path,f),'r')
    # import pdb;pdb.set_trace()
    if ('kspace' not in h5f) or ('reconstruction_esc' not in h5f):
        continue
    for i,slic in enumerate(h5f['kspace']):
        inputs.append(image_from_kspace(slic,multicoil=False))
        # import pdb;pdb.set_trace()
        outputs.append(h5f['reconstruction_esc'][i]/485.3040)
    h5f.close()

pairs = list(zip(inputs,outputs))
# import pdb;pdb.set_trace()
my_set = NNSet_from_fastMRI(pairs)

outfile = h5py.File('/mnt/d/NN_DATA/fastMRI/test/test.h5','w')
my_set.save(outfile)
outfile.close()