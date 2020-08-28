import os
import gc
import cv2
import bpy
import sys
import json
import time
import bpycv
import numpy as np
from mathutils import Euler
from mathutils import Matrix
from pycocotools import mask


# For Correct results Make all the flags True
# Use these flags as a sanity check for images/masks/mat/binvox
# Use these flags while development only
BINVOX_RENDER = False # Turn flag TRUE to render everthing
RENDER_IMAGES = True # Turn flag TRUE to render everthing

# This is the scripts that is used to generate the 
# pix3d dataset train and test splits,

models_path = "../pix3d/models"
images_path = "../pix3d/Images"
masks_path  = "../pix3d/masks"

im_path = "./Images"
mk_path = "./masks"
ml_path = "./models"

models = os.listdir(models_path)
models.sort()

obj_par_file = open("../src/obj_locations_small.txt", 'r')
obj_locations = []
for line in obj_par_file:
    inner_list = [float(etl.strip()) for etl in line.split(' ')]
    obj_locations.append(inner_list)


pix3d = {
        "categories": [],      # append the id with names
        "annotations": [],     # append the mappings
        "licenses": "",
        "images": [],           # append the Images
        "info" : {
            "url": "",         # give link to download able data
            "contributer": "", # name the contributer
            "year" : "2020",
            "description": ""  # Some Description
        }
    }

annotations = {
    "id" : 0,
    "iscrowd" : 0,
    "segmentation" : "", # address to the binary mask of the object
    "model" : "",        # address to the .obj model
    "category_id" : "",  # categroy id like rack -> 1, box -> 2
    "K" : [],            # find K is a list of 3 numbers camera matrix?
    "bbox" : [],         # can be found from pycocotools.mask.toBbox
    "trans_mat": [],     # can be found from Blender
    "area" : "",         # can be found from pycocotools.mask.area
    "image_id": "",      # Image Mapping
    "rot_mat": [],       # Rotation matrix
    "voxel" : ""         # voxel grid mapping
    }

image = {
    "height" : 0,        # y axis length
    "width" : 0,         # x axis length
    "id" : 0,            # generate the ids
    "file_name" : ""     # rgb image path mapping
    }

categories = {
    "id" : 0,
    "name" : ""
}

def delete_models():
    collection_name = "Collection"
    collection = bpy.data.collections[collection_name]
    meshes = set()
    for obj in [o for o in collection.objects if o.type == 'MESH']:
        meshes.add( obj.data )
        bpy.data.objects.remove( obj )

    for mesh in [m for m in meshes if m.users == 0]:
        bpy.data.meshes.remove( mesh )

def mask_from_depth(img):
    img = np.array(img)
    h, w = img.shape
    for r in range(h):
        for c in range(w):
            if(img[r,c] == 0):
                img[r,c] = 0
            else:
                img[r,c] = 255
    return img

def get_category_id(name):
	# Hardcoded this function
	ret = 1
	if(name == "BoxA"):
		ret = 1
	elif(name == "BoxB"):
		ret = 2
	elif(name == "BoxC"):
		ret = 3
	elif(name == "BoxD"):
		ret = 4
	elif(name == "BoxF"):
		ret = 5
	elif(name == "BoxH"):
		ret = 6
	elif(name == "RackA"):
		ret = 7
	category_id = ret
	return category_id

def get_K(camd):
	# https://github.com/facebookresearch/meshrcnn/issues/8
    f_in_mm = camd.lens
    # f_in_mm is equivalent to f_pix3d
    scene = bpy.context.scene
    sensor_width_in_mm = camd.sensor_width

    image_width = scene.render.resolution_x
    image_height = scene.render.resolution_y
    K = [0,0,0]

    K[0] = f_in_mm * image_width / sensor_width_in_mm
    K[1] = image_width / 2
    K[2] = image_height / 2
    
    return K

def get_bbox_and_area(msk):
    ground_truth_binary_mask = np.array(msk,dtype=np.uint8)
    #np.savetxt("../test.txt",ground_truth_binary_mask)
    fortran_ground_truth_binary_mask = np.asfortranarray(ground_truth_binary_mask)
    encoded_ground_truth = mask.encode(fortran_ground_truth_binary_mask)
    ground_truth_bounding_box = mask.toBbox(encoded_ground_truth)
    ground_truth_area = mask.area(encoded_ground_truth)
    bbox = [m for m in ground_truth_bounding_box]
    area = int(ground_truth_area)
    return bbox, area

def euler2mat(euler):
    """ Convert Euler Angles to Rotation Matrix.  See rotation.py for notes """
    euler = np.asarray(euler, dtype=np.float64)
    assert euler.shape[-1] == 3, "Invalid shaped euler {}".format(euler)

    ai, aj, ak = -euler[..., 2], -euler[..., 1], -euler[..., 0]
    si, sj, sk = np.sin(ai), np.sin(aj), np.sin(ak)
    ci, cj, ck = np.cos(ai), np.cos(aj), np.cos(ak)
    cc, cs = ci * ck, ci * sk
    sc, ss = si * ck, si * sk

    mat = np.empty(euler.shape[:-1] + (3, 3), dtype=np.float64)
    mat[..., 2, 2] = cj * ck
    mat[..., 2, 1] = sj * sc - cs
    mat[..., 2, 0] = sj * cc + ss
    mat[..., 1, 2] = cj * sk
    mat[..., 1, 1] = sj * ss + cc
    mat[..., 1, 0] = sj * cs - sc
    mat[..., 0, 2] = -sj
    mat[..., 0, 1] = cj * si
    mat[..., 0, 0] = cj * ci
    return mat

def MatrixToList(mat):
    lst = []
    for i in range(len(mat)):
        lst.append([])
        for j in range(len(mat[i])):
            lst[i].append(mat[i][j])
    
    return lst

# get the camera Instance 
camera = bpy.data.objects['Camera']
os.system("rm -rf "+images_path)
os.system("rm -rf "+masks_path)
os.system("mkdir "+masks_path)
os.system("mkdir "+images_path)

img_id = 1
cat_id = 1
ann_id = 1
# Iterative loop steps
for model in models:
    
    categories_copy = categories.copy()
    categories_copy["id"] = cat_id
    categories_copy["name"] = model
    cat_id = cat_id + 1
    pix3d["categories"].append(categories_copy)

    # 1. Clear all obj files
    delete_models()

    #remove any .binvox or .mat file if already existing
    if(BINVOX_RENDER):
        os.system("rm "+ models_path + "/" + model + "/*.binvox")
        os.system("rm "+ models_path + "/" + model + "/*.mat")
        os.system("rm "+ models_path + "/" + model + "/*.obj")
        os.system("rm "+ models_path + "/" + model + "/*.mtl")

    # Import the model
    model_path = models_path + "/" + model + "/model.obj"
    model_path_dae = models_path + "/" + model + "/model.dae"
    #bpy.ops.import_scene.obj(filepath = model_path)
    bpy.ops.wm.collada_import(filepath = model_path_dae)

    # Get the object
    obj = bpy.data.objects[model]

    # Rotate the object by 90 in X
    obj.rotation_euler = [1.570796, 0, 0]

    # Export the model as .obj
    bpy.ops.export_scene.obj(filepath = model_path)
    
    # generate the voxel grid and convert to .mat format
    voxel_model_path = models_path + "/" + model + "/model.mat"
    if(BINVOX_RENDER):
        print("Creating the .mat files")
        os.system("python ../src/converter.py " + model_path)
    
    # Remove the rotation
    #TODO

    # make the directory for storing the masks and rgb
    os.mkdir(images_path + "/" + model)
    os.mkdir(masks_path + "/" + model)

    for loc_number in range(len(obj_locations)):

        # Set camera positions
        #camera.location = obj_locations[loc_number][:3] #set the camera location and angles
        #camera.rotation_euler = obj_locations[loc_number][3:] #[1.57, 0 , 1.57]
        
        trans_vector = [0, 0, 0]
        euler_vector = [0, np.pi, 0]
        rot_mat = euler2mat(euler_vector)
        trans_4x4 = Matrix.Translation(trans_vector)
        rot_4x4 = Matrix(rot_mat).to_4x4()
        scale_4x4 = Matrix(np.eye(4)) # don't scale here
        camera.matrix_world = trans_4x4 @ rot_4x4 @ scale_4x4

        #Set the object locations
        # Change the object location instead of the camera
        # inspired from https://github.com/xingyuansun/pix3d/blob/e0fc891041e8c3d381240e3d699ba734a50d26c5/demo.py#L77
        obj = bpy.data.objects[model]
        obj.location = obj_locations[loc_number][:3] #set the camera location and angles
        obj.rotation_euler = obj_locations[loc_number][3:]

        # get the trans and rot matrix of the object
        trans_mat = obj_locations[loc_number][:3]
        euler_vector = obj_locations[loc_number][3:]
        eul = Euler(euler_vector,'XYZ')
        rot_mat = eul.to_matrix()
        rot_mat = MatrixToList(rot_mat) # change the Matrix to list
        
        # Render the scene
        result = bpycv.render_data()
    
        # Get the depth Images
        depth = result["depth"] / result["depth"].max() * 255
    
        # Get the mask from depth
        msk = mask_from_depth(depth)

        # Get the rgb
        if(RENDER_IMAGES):
            rgb = cv2.cvtColor(result["image"], cv2.COLOR_RGB2BGR)

        # Set the paths
        rgb_path = images_path + "/" + model + "/" + str(loc_number) + ".png"
        msk_path =  masks_path + "/" + model + "/" + str(loc_number) + ".png"

        rgb_path_annotation = im_path + "/" + model + "/" + str(loc_number) + ".png"
        msk_path_annotation = mk_path + "/" + model + "/" + str(loc_number) + ".png"
        mod_path_annotation = ml_path + "/" + model + "/model.obj"
        vxl_path_annotation = ml_path + "/" + model + "/model.mat"

        # Save the Images

        if(RENDER_IMAGES):
            cv2.imwrite(rgb_path, rgb)
            cv2.imwrite(msk_path, msk)		

        # Get some essential values
        bbox, area = get_bbox_and_area(msk)

		# Fill the annotations json
        annotations_copy = annotations.copy()
        annotations_copy["id"] = ann_id
        annotations_copy["segmentation"] = msk_path_annotation
        annotations_copy["model"] = mod_path_annotation
        annotations_copy["category_id"] = get_category_id(model)
        annotations_copy["K"] = get_K(camera.data)
        annotations_copy["bbox"] = bbox
        annotations_copy["trans_mat"] = trans_mat
        annotations_copy["rot_mat"] = rot_mat
        annotations_copy["area"] = area
        annotations_copy["image_id"] = img_id
        annotations_copy["voxel"] = vxl_path_annotation

        # append the filled values to the pix3d
        pix3d["annotations"].append(annotations_copy)

        # Fill the images json
        image_copy = image.copy()
        image_copy["height"] = rgb.shape[0]
        image_copy["width"] = rgb.shape[1]
        image_copy["id"] = img_id
        image_copy["file_name"] = rgb_path_annotation

        #increment img_id
        img_id = img_id + 1
        ann_id = ann_id + 1

        # append the images to pix3d
        pix3d["images"].append(image_copy)
        time.sleep(0)

with open('../pix3d/pix3d_s1_train.json', 'w+') as outfile:
    json.dump(pix3d, outfile)

gc.collect()
