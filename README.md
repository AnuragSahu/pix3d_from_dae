# Collada to Pix3d Format

This can be used to convert the model in .dae to the pix3d format so that you can train your 3d Reconstrution Model. <br>

## Requirements
- Blender
    - Numpy
    - Opencv
    - pycocotools
- binvox
- scipy

## Installation

You need to install the packages mentioned above under the blender to the python which comes bundled with blender.<br>
The Packages can be installed to python bundled with the following steps<br>
1. Clone the repo 
```
git clone https://github.com/AnuragSahu/pix3d_from_dae.git
cd pix3d_from_dae/
```

2. Setup Blender
### Method 1 (Long) Download and Install
a. Get Blender
```
wget https://ftp.nluug.nl/pub/graphics/blender/release/Blender2.83/blender-2.83.1-linux64.tar.xz
tar -xf blender-2.83.1-linux64.tar.xz 
mv blender-2.83.1-linux64.tar.xz ./blender
cd blender
```
b. Install packages
```
cd 2.83/python/bin/
./python3.7m -m ensurepip
./python3.7m -m pip install opencv-python
./python3.7m -m pip install numpy
./python3.7m -m pip install pycocotools
cd ../../../../
```

### Method 2 (Short) Download my Blender
a. Get the blender from Google Drive
```
git clone https://github.com/AnuragSahu/download_to_Server_from_Google_drive.git
cd download_to_Server_from_Google_drive/
python download_gdrive.py 1Yt-FS-_ppg0-j9CZgxdIGIjaptazKIhN ../blender.zip
cd ../
rm -rf download_to_Server_from_Google_drive
```
b. Unzip Blender
```
cd ..
unzip blender.zip
```

3. Get the binvox and scipy, binvox is already given with the github repo
```
pip install scipy
```

## Get the datasets
1. Setup,
You should place you models in ./pix3d_train/models with the following file structure
```
models
    |
    |----> Object1_Name
                |
                |----> model.dae
    |
    |----> Object2_Name
                |
                |----> model.dae
```
*note that the name model.dae should not be changed<br>
2. Run the code
```
cd ./blender/
./blender --background -P ../pix3d_dataset/src/generate_pix3d.py
```

