import os,sys
import numpy as np
import binvox_rw
import scipy.io as sio

debug = False

MODEL_PATH = sys.argv[1] 
BINVOX_OUTPUT_PATH = MODEL_PATH[:-4]+".binvox"
MAT_OUTPUT_PATH = MODEL_PATH[:-3]+"mat"
os.system( "../pix3d_dataset/src/binvox " + MODEL_PATH + " -d 128" )
with open(BINVOX_OUTPUT_PATH, 'rb') as f:
    model = binvox_rw.read_as_3d_array(f)

if(debug):
    print("MODEL DIMENSIONS: ",model.dims)
data = np.array(model.data)
if(debug):
    print(data)

sio.savemat( MAT_OUTPUT_PATH, {'voxel':data} )
