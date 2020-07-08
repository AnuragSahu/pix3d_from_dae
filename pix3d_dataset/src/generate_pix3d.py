import os
import gc
import cv2
import bpy
import sys
import json
import bpycv
import numpy as np
from mathutils import Matrix
from pycocotools import mask

# For Correct results Make all the flags True
# Use these flags as a sanity check for images/masks/mat/binvox
# Use these flags while development only
BINVOX_RENDER = False # Turn every thingoff to render everthing

# This is the scripts that is used to generate the 
# pix3d dataset train and test splits,

models_path = "../pix3d_train/models"
images_path = "../pix3d_train/Images"
masks_path  = "../pix3d_train/masks"

im_path = "./Images"
mk_path = "./masks"
ml_path = "./models"

models = os.listdir(models_path)
models.sort()

if(len(sys.argv) == 2):
    cam_par_file_path = sys.args[1]
    cam_par_file = open(cam_par_file_path, 'w')
    camera_locations = cam_par_file.readlines()

else:
    camera_locations = [[10, 0, 0, 1.57, 0 , 1.57],    #change this to make a 
                        [11, 0, 0, 1.57, 0 , 1.57],    #file location giving the camera positions
                        [17, 0, 2, 1.57, 0 , 1.57]]

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
    "Name" : ""
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
            if(img[r,c] < 200):
                img[r,c] = 0
            else:
                img[r,c] = 255
    return img

def get_category_id(name):
	if(len(name) == 4): # for Boxes
		category_id = ord(name[-1]) - ord('A') + 1
	elif(len(name) == 5):
		category_id = ord(name[-1]) - ord('A') + 7
	return category_id

def get_K(camd):
	# https://github.com/facebookresearch/meshrcnn/issues/8
    f_in_mm = camd.lens
    scene = bpy.context.scene
    resolution_x_in_px = scene.render.resolution_x
    resolution_y_in_px = scene.render.resolution_y
    scale = scene.render.resolution_percentage / 100
    sensor_width_in_mm = camd.sensor_width
    K = [0,0,0]
    K[0] = f_in_mm * resolution_x_in_px * scale / 2
    K[1] = resolution_x_in_px * scale / 2
    K[2] = resolution_y_in_px * scale / 2
    return K

def get_bbox_and_area(msk):
    #print(msk)
    #np.savetxt('../msk.txt',msk)
    ground_truth_binary_mask = np.array(msk,dtype=np.uint8)
    #np.savetxt("../test.txt",ground_truth_binary_mask)
    fortran_ground_truth_binary_mask = np.asfortranarray(ground_truth_binary_mask)
    encoded_ground_truth = mask.encode(fortran_ground_truth_binary_mask)
    ground_truth_bounding_box = mask.toBbox(encoded_ground_truth)
    ground_truth_area = mask.area(encoded_ground_truth)
    bbox = [m for m in ground_truth_bounding_box]
    area = int(ground_truth_area)
    return bbox, area
	
def get_trans_and_rot_mat(cam):
    # bcam stands for blender camera
    R_bcam2cv = Matrix(
        ((1, 0,  0),
         (0, -1, 0),
         (0, 0, -1)))
    # Transpose since the rotation is object rotation, 
    # and we want coordinate rotation
    # R_world2bcam = cam.rotation_euler.to_matrix().transposed()
    # T_world2bcam = -1*R_world2bcam * location
    #
    # Use matrix_world instead to account for all constraints
    location, rotation = cam.matrix_world.decompose()[0:2]
    R_world2bcam = rotation.to_matrix().transposed()
    # Convert camera location to translation vector used in coordinate changes
    # T_world2bcam = -1*R_world2bcam*cam.location
    # Use location from matrix_world to account for constraints:     
    T_world2bcam = -1*R_world2bcam @ location
    # Build the coordinate transform matrix from world to computer vision camera
    # NOTE: Use * instead of @ here for older versions of Blender
    # TODO: detect Blender version
    R_world2cv = R_bcam2cv@R_world2bcam
    T_world2cv = R_bcam2cv@T_world2bcam
    rot_mat = []
    for i in R_world2cv:
        row = []
        for j in i:
            row.append(j)
        rot_mat.append(row)

    trans_mat = [m for m in T_world2cv]

    return rot_mat, trans_mat

# get the camera Instance 
camera = bpy.data.objects['Camera']
os.system("rm -rf "+images_path)
os.system("rm -rf "+masks_path)
os.system("mkdir "+masks_path)
os.system("mkdir "+images_path)

img_id = 1
cat_id = 1
# Iterative loop steps
for model in models:
    
    categories_copy = categories.copy()
    categories_copy["id"] = cat_id
    categories_copy["Name"] = model
    cat_id = cat_id + 1
    pix3d["categories"].append(categories_copy)

    # 1. Clear all obj files
    delete_models()

    #remove any .binvox or .mat file if already existing
    os.system("rm "+ models_path + "/" + model + "/*.binvox")
    os.system("rm "+ models_path + "/" + model + "/*.mat")
    os.system("rm "+ models_path + "/" + model + "/*.obj")
    os.system("rm "+ models_path + "/" + model + "/*.mtl")

    # Import the model
    model_path = models_path + "/" + model + "/model.obj"
    model_path_dae = models_path + "/" + model + "/model.dae"
    #bpy.ops.import_scene.obj(filepath = model_path)
    bpy.ops.wm.collada_import(filepath = model_path_dae)

    # Export the model as .obj
    bpy.ops.export_scene.obj(filepath = model_path)
    
    # generate the voxel grid and convert to .mat format
    voxel_model_path = models_path + "/" + model + "/model.mat"
    if(BINVOX_RENDER):
        os.system("python ../pix3d_dataset/src/converter.py " + model_path)
        

    # make the directory for storing the masks and rgb
    os.mkdir(images_path + "/" + model)
    os.mkdir(masks_path + "/" + model)

    for loc_number in range(len(camera_locations)):

        # Set camera positions
        camera.location = camera_locations[loc_number][:3] #set the camera location and angles
        camera.rotation_euler = camera_locations[loc_number][3:] #[1.57, 0 , 1.57]

        # Render the scene
        result = bpycv.render_data()
    
        # Get the depth Images
        depth = result["depth"] / result["depth"].max() * 255
    
        # Get the mask from depth
        msk = mask_from_depth(depth)

        # Get the rgb
        rgb = cv2.cvtColor(result["image"], cv2.COLOR_RGB2BGR)

        # Set the paths
        rgb_path = images_path + "/" + model + "/" + str(loc_number) + ".png"
        msk_path =  masks_path + "/" + model + "/" + str(loc_number) + ".png"

        rgb_path_annotation = im_path + "/" + model + "/" + str(loc_number) + ".png"
        msk_path_annotation = mk_path + "/" + model + "/" + str(loc_number) + ".png"
        mod_path_annotation = ml_path + "/" + model + "/model.obj"
        vxl_path_annotation = ml_path + "/" + model + "/model.mat"

        # Save the Images

        cv2.imwrite(rgb_path, rgb)
        cv2.imwrite(msk_path, msk)		

        # Get some essential values
        bbox, area = get_bbox_and_area(msk)
        rot_mat, trans_mat = get_trans_and_rot_mat(camera)

		# Fill the annotations json
        annotations_copy = annotations.copy()
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
        image_copy["height"] = rgb.shape[1]
        image_copy["width"] = rgb.shape[0]
        image_copy["id"] = img_id
        image_copy["file_name"] = rgb_path_annotation

        #increment img_id
        img_id = img_id + 1

        # append the imagesto pix3d
        pix3d["images"].append(image_copy)

with open('../pix3d_train/pix3d_train.json', 'w+') as outfile:
    json.dump(pix3d, outfile)

gc.collect()