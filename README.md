
#  Pytorch Training Template

This repo contains template code for quickly setting up any training pipeline in pytorch. A lot of boiler plate code that we rewrite while setting up a new training pipeline is already present and only the custom features needs to be added. This code is based on training with config file. So all the parameters needed for training should be present inside a yaml file in configs folder and train.py depends on that.

Code structure is as follows,

 - models	
	 - sample_model.py - Contains the model architecture definition	
 - configs
	 - sample_config.yaml - Contains all the parameters needed for training
 - utils
	 - common.py - Contains all the functions needed for carrying out training, evaluation and loss calculation
	 - dataset.py - Contains the dataset class that tells how to load the data
 - train.py - File that starts the training. Can do both distributed and single GPU training

## Credits
Most of the code is adapted from the YOLOV5 repo at https://github.com/ultralytics/yolov5. 