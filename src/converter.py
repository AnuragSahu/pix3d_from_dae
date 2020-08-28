import os,sys
import numpy as np
import binvox_rw
import scipy.io as sio

debug = False

MODEL_PATH = sys.argv[1] 
BINVOX_OUTPUT_PATH = MODEL_PATH[:-4]+".binvox"
MAT_OUTPUT_PATH = MODEL_PATH[:-3]+"mat"
os.system( "../src/binvox " + MODEL_PATH + " -d 128" )
with open(BINVOX_OUTPUT_PATH, 'rb') as f:
    model = binvox_rw.read_as_3d_array(f)

if(debug):
    print("MODEL DIMENSIONS: ",model.dims)
data = np.array(model.data)
if(debug):
    print(data)

voxels = data
length, bredth, height = voxels.shape
min_x = 128
min_y = 128
min_z = 128
max_x = 0
max_y = 0
max_z = 0

#to find the bounding box
for le in range(length):
    for br in range(bredth):
        for he in range(height):
            if(voxels[le, br, he]==1):
                #for X
                if(min_x > le):
                    min_x = le
                if(max_x < le):
                    max_x = le
                    
                #for Y
                if(min_y > br):
                    min_y = br
                if(max_y < br):
                    max_y = br
                
                #for Z
                if(min_z > he):
                    min_z = he
                if(max_z < he):
                    max_z = he

                    
translate = [0, 0, 0]
trans_x = int((128/2) - (max_x/2))
trans_y = int((128/2) - (max_y/2))
trans_z = int((128/2) - (max_z/2))

# to shift the voxels
for le in range(max_x, -1, -1):
    for br in range(max_y, -1, -1):
        for he in range(max_z, -1, -1):
            voxels[le+trans_x][br+trans_y][he+trans_z] = voxels[le, br, he]
            voxels[le, br, he] = 0

sio.savemat( MAT_OUTPUT_PATH, {'voxel':voxels} )
