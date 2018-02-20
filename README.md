# Behavioral-cloning-project


# Behaviorial Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
This repository contains the required files for the Behavioral Cloning Project.


### Dependencies
This lab requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

The lab enviroment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.

The following files can be found in this github repository:
* drive.py
* model.py
* model.h5
* README.md
* writeup_template.md

The simulator can be downloaded from the udacity Classroom.


## Details About Files In This Directory

### `model.py`

The python code for the neural network architecture that trains and outputs the model.

### `model.h5`

The neural network is saved in an h5 format to be used for testing in the simulator.

### `drive.py`

Usage of `drive.py` requires you have saved the trained model as an h5 file.

Once the model has been saved, it can be used with drive.py using this command:

```sh
python drive.py model.h5
```

The above command will load the trained model and use the model to make predictions on individual images in real-time and send the predicted angle back to the server via a websocket connection.

Note: There is known local system's setting issue with replacing "," with "." when using drive.py. When this happens it can make predicted steering values clipped to max/min values. If this occurs, a known fix for this is to add "export LANG=en_US.utf8" to the bashrc file.

For further details on the implementation of the model. refere to the writeup.md file.



