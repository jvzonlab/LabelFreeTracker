# LabelFreeTracker
Predicts nuclear and cell envelope reporters from transmitted light images. This allows you to view nuclei and cell envelopes in organoids without fluorescent markers, allowing you to segment and track those cells.

![Method overview](https://user-images.githubusercontent.com/1462188/222784969-bc1b02a0-a0a3-459c-92c9-10b9cc5e16a4.png)

## Table of contents
* [System requirements](#system-requirements)
* [Installation](#installation)
* [Training a neural network for nucleus prediction or cell membrane prediction](#training-a-neural-network-for-nucleus-prediction-or-cell-membrane prediction)
* [Predicting nucleus images or cell membrane images](#predicting-nucleus-images-or-cell-membrane-images)
* [Training a neural network for position detection](#training-a-neural-network-for-position-detection)
* [Predicting positions](#predicting-positions)
* [Predicting links over time](#predicting-links-over-time)
* [Tracking cells over time](#tracking-cells-over-time)

## System requirements
Installation was tested on Microsoft Windows only. We're using a CUDA-compartible graphics card with 10 GB of VRAM, but you can likely get it to work too with weaker specs.

## Installation
For installation, please follow the following steps:

1. LabelFreeTracker has been developed as a plugin for [OrganoidTracker](https://github.com/jvzonlab/OrganoidTracker). To use LabelFreeTracker, first follow the installation instructions of OrganoidTracker, and verify that you can open the program.
2. Then, [download LabelFreeTracker](https://github.com/jvzonlab/LabelFreeTracker/archive/refs/heads/main.zip). The download will contain a `LabelFreeTracker-main` folder, which in turn contains a `Plugins for OrganoidTracker` folder.
3. Next, open OrganoidTracker, and then use the `File` menu to open the folder that contains the plugins:

  ![How to open the plugins folder](https://user-images.githubusercontent.com/1462188/222796147-380612db-54da-44ab-aebe-f7825a02643f.png)

4. Now place the files *inside* the `Plugins for OrganoidTracker` folder inside the plugins folder of OrganoidTracker, like this:

  ![Drag and drop the files](https://user-images.githubusercontent.com/1462188/222797179-22a5e81e-feb9-41d0-8023-281a917e67ec.png)

5. Now back in OrganoidTracker, use `File` -> `Reload all plugins...`. If everything went successfully, you will now have four new menu options:

  ![New menu options](https://user-images.githubusercontent.com/1462188/222797841-2730abd9-0af1-485e-975d-089559a4ff87.png)


## Training a neural network for nucleus prediction or cell membrane prediction
In this section, we are going to train a neural network to predict nuclei from transmitted light images. For this, we need a dataset of fluorescent nucleus images with accompanying transmitted light images. For this training process, you will need organoids with fluorescent nuclei. After the training is complete, you can use the network on organoids without a nuclear reporter.

To make sure that you can follow our instructions, we are going to use [this example dataset](https://zenodo.org/record/7197573) from an earlier paper from our lab. If you have managed to train the network successfully on our data, you can add your images to the mix, or even train the network exclusively on your images. Note that it helps training if you make your images as oversaturated as our images, i.e. almost the entire nucleus is fully white. This is because then the neural network doesn't need to learn to predict the fluorescent intensity, but just whether there's nucleus at the location.

Download the example dataset, and open each time lapse in OrganoidTracker. You load each time lapse separately in another tab. The buttons for loading images and opening tabs are shown here:

![OrganoidTracker interface](https://user-images.githubusercontent.com/1462188/222802432-56ab6492-83a1-4f1c-b65a-e2a98b3fc8cd.png)

You must make sure that all time lapses are opened in OrganoidTracker, each in their own tab. (RAM usage shouldn't be a problem, as OrganoidTracker doesn't attempt to load the entire time lapse into memory.) When opening an image, select the image of the first time point and the first channel, like image `nd799xy08t001c1.tif`. OrganoidTracker will then automatically find all other images in the time lapse.

Once all time lapses are loaded, switch to the `<all experiments>` tab. Then, use `Tools` -> `From transmitted light` -> `Train cell painting...`. LabelFreeTracker will prompt you to create a folder. Then, open this folder. You should see something like this:

![Folder for training](https://user-images.githubusercontent.com/1462188/222803808-7451bcfa-b0d4-40ae-9676-698804e4d668.png)

Double-click `train_cell_painting.bat`. This won't start the training process. Instead, you will see this screen:

![Command window](https://user-images.githubusercontent.com/1462188/222804415-074248cb-e944-43c6-85f4-024e85d74791.png)

Now, open the `organoid_tracker.ini` file in the same folder, which will now contain the settings of the training process. The default settings are mostly fine, but I would set `epochs` to 4, so that the training process goes through all the data four times. If your graphics card runs out of memory during training, you can set `patch_size_zyx = 128,128,16`. If you're using your own images set, make sure that `fluorescence_channel` and `transmitted_light_channel` are set correctly.

```ini
[DEFAULTS]

[scripts]
extra_plugin_directory = C:\Users\Rutger\AppData\Roaming\OrganoidTracker\Plugins

[train_cell_painting]
; Please paste the path to the .autlist file containing the training data
input_file = training_and_validation_dataset.autlist
; Folder that will contain all output files
output_folder = training_output
; Specify the path to an existing model here, if you want to do transfer learning.
starting_model_path = 
; Number of epochs to train
epochs = 4
; Batch size. Lower this if you run out of memory
batch_size = 4
; Fluorescence channel, used as the training target. Only used if painting_target is set to FLUORESCENCE_CHANNEL. Use 1 for the first channel, 2 for the second channel, etc. Use "1 from end" to use the last channel.
fluorescence_channel = 1
; Transmitted light channel, used as training input.
transmitted_light_channel = 2
; Determines how large the training volume is
patch_size_zyx = 256,256,16
; Fraction of time points that is used for validation data instead of training data
validation_fraction = 0.2
; Normally, patches with no fluorescent signal are skipped for training. This number controls what fraction is still included in the training data.
leak_empty_fraction = 0.05
```

Now start the training process by running `train_cell_painting.bat` again. Within an hour or so, training should be done. You'll get an output folder with example images showing how the results of the network improved (or not) after each trianing epoch.

## Predicting nucleus images or cell membrane images



## Training a neural network for position detection



## Predicting positions


## Predicting links over time



## Tracking cells over time

