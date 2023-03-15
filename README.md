# Building-Road-AI-Detecting
This is a project for detecting building &amp; road with mask-rcnn &amp; unet model.


For calculating geojson coordinate, need input ".tif" file with correct "crs" value.

# Package
This project only support python>=3.8.

# Requirements
Requirements file is in the requirements.txt
	pip install -r requirements_path/requirements.txt

# Usage
Run main_code.py, and result will be created to path:
	
	input_image_path/image_name/result/

#### |-- Change Weight:
	config/UNet.py: "cfg = {'weights': 'your UNET weight'}"
	config/Mask_RCNN.py: "cfg = {'weights': 'your MASK-RCNN weight'}"

#### |-- Code:
Main_code.py: 
    
	Main code need input your_image_path

mask_rcnn.py & unet.py:

	For detecing and predict

# Folder:
src/: Model utils code

src/cocojson/: COCOJson create tools 

# References
COCO create tools: https://github.com/waspinator/pycococreator

Mask-RCNN: https://github.com/bubbliiiing/mask-rcnn-tf2

Unet: https://github.com/bubbliiiing/unet-tf2
