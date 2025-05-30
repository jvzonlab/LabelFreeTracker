# LabelFreeTracker
Predicts nuclear and cell envelope reporters from transmitted light images. This allows you to view nuclei and cell envelopes in organoids without fluorescent markers, allowing you to segment and track those cells.

![Method overview](https://user-images.githubusercontent.com/1462188/222784969-bc1b02a0-a0a3-459c-92c9-10b9cc5e16a4.png)

## Table of contents
* [System requirements](#system-requirements)
* [Installation](#installation)
* [Training a neural network for nucleus prediction or cell membrane prediction](#training-a-neural-network-for-nucleus-prediction-or-cell-membrane-prediction)
* [Predicting nucleus images or cell membrane images](#predicting-nucleus-images-or-cell-membrane-images)
* [Training a neural network for position detection](#training-a-neural-network-for-position-detection)
* [Predicting positions](#predicting-positions)
* [Predicting links over time](#predicting-links-over-time)
* [Tracking cells over time](#tracking-cells-over-time)
* [Segmenting cells](#segmenting-cells)

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

The procedure for training cell membrane images is identical, except that you need to provide fluorescent membrane images instead of nucleus images. Therefore, we will not discuss training the network to draw membranes.

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
Once you've got a trained network, you can use it as follows. First, open the time lapse of transmitted light images that you want to use in OrganoidTracker. *Make sure you are on the transmitted light channel, and not any other channel that you might have imaged.* Then, use `Tools` -> `From transmitted light` -> `Predict cell painting`:

![Starting cell painting](https://user-images.githubusercontent.com/1462188/222900652-3f7ea7dc-678a-48e1-a8f0-8b5c5ef82251.png)

LabelFreeTracker will then ask you two questions: where you have stored the trained model (it will be in a folder named `saved_model`, that can be found in the output files from the training procedure), and in which folder you want to save the painted images. Then, open that folder (LabelFreeTracker will show a button to open it for you), and run the `predict_cell_painting.bat` script:

![Running cell painting](https://user-images.githubusercontent.com/1462188/222900814-108a0191-54ee-4eb4-912d-3ddc4692c69d.png)

After the script is finished, all output images will be placed in a subfolder. You can load these images in OrganoidTracker, which allows you to manually track the cells.


## Training a neural network for position detection
If you have obtained nucleus images from the above steps, you can in principle track the cells manually in OrganoidTracker. However, depending on your usecase it might be better to automate this step.

In this section, we are going to train a neural network to directly tell us where the cell centers (or actually: nucleus centers) are, based on transmitted light images. The required training data consists of transmitted light images with corresponding nucleus center positions. To obtain this data, a good strategy would be to use organoids with a fluorescent nucleus reporter, and detect the nucleus center positions using a program like [CellPose](https://www.cellpose.org/), as CellPose works well enough for most datasets even without retraining. (Note that CellPose gives you the full segmentation of the nucleus. We will only use the center position.) 

Alternatively, if you want to set up full cell tracking instead of just nucleus center position detection, it is better to already bite the bullet, and create a dataset of manual tracking data using for example OrganoidTracker. This will later on allow you to evaluate the performance of your cell tracking.

In this example, we are going to simply use the pre-existing tracking data, so that you can follow along. Load all images as shown in the [fluorescence prediction section](#training-a-neural-network-for-nucleus-prediction-or-cell-membrane-prediction). Next, also load all OrganoidTracker tracking files from the same dataset. Make sure that you have all orgaonids loaded in OrganoidTracker, one organoid per tab.

![Loading tracking data](https://user-images.githubusercontent.com/1462188/222903440-9d0b9317-e5f8-4732-b2c8-2f70719923f6.png)

Then, switch to the `<all experiments>` tab and use `Tools` -> `From transmitted light` -> `Train nucleus center prediction`. Save the folder somewhere, and open it. The folder should look like this:

![Training folder](https://user-images.githubusercontent.com/1462188/222903568-cb5fefbd-3fcd-4524-aee2-f5e56a685fcf.png)

Then, run the `train_nucleus_centers_from_transmitted_light.bat` script by double-clicking it. It will modify the `organoid_tracker.ini` file to place the default settings in it. Open that file, and check whether the settings are corerct. I changed the number of epochs to 4, and verified that the transmitted light channel was indeed channel 2. If you run out of GPU memory during training, you can reduce the `patch_size_zyx` to `128,128,16`.

Now run the `train_nucleus_centers_from_transmitted_light.bat` script again, and the training will start. This might take several hours. You will get an output folder with example images of how the training improved over time.

## Predicting positions
Once you've got a trained network, you can use it as follows. First, open the time lapse of transmitted light images that you want to use in OrganoidTracker. *Make sure you are on the transmitted light channel, and not any other channel that you might have imaged.* Then, use `Tools` -> `From transmitted light` -> `Predict nucleus center positions...`:

LabelFreeTracker will then ask you two questions: where you have stored the trained model (it will be in a folder named `saved_model`, that can be found in the output files from the training procedure), and in which folder you want to save the detected ositions. Then, open that folder (LabelFreeTracker will show a button to open it for you), and run the `predict_cell_painting.bat` script:

![Running position prediction](https://user-images.githubusercontent.com/1462188/222904633-c65a86be-f39a-42ba-b90f-983f81870295.png)

After the script is finished, you'll get an OrganoidTracker output file. If you want to process this file yourself, it's a [JSON file](https://www.w3schools.com/python/python_json.asp), and it should be easy to parse using any JSON parser.

If you're interested in cell tracking, continue reading in the next section. If you're interested in cell segmentation instead, please jump to the [cell segmentation section](#segmenting-cells).

## Predicting links over time
In this step, we will predict the likelihood of two cell observations in two subsequent time points representing the same cell. LabelFreeTracker uses a quite simplistic approach for this: it looks at the spatial distance that the cell observations are apart, and then looks up what the chance is that two observations are indeed of the same cell.

This part is a bit less fledged-out. We don't have a nice menu option here to generate the configuration files and .bat file, so we need to do that ourselves. Create a folder somewhere on your PC, and create the following files in it:

#### labelfreetracker_predict_link_scores.bat
```batch
@echo off
@CALL "C:\PATH\TO\CONDA\condabin\conda.bat" activate organoid_tracker
"C:\PATH\TO\CONDA\envs\organoid_tracker\python.exe" "C:\PATH\TO\OrganoidTracker\organoid_tracker.py" "set_link_scores_from_distance"
pause
```

Replace the paths with the real path to your Anaconda installation and OrganoidTracker folder. If you don't know those paths, then look at the contents of the generated .bat file in any of the earlier steps, they will contain the correct paths. For me, Anaconda was installed in `C:\Users\Rutger\AppData\Roaming\miniconda\` and OrganoidTracker in `C:\Users\Rutger\Documents\Git\OrganoidTracker`, and for you it will be different.

### organoid_tracker.ini
```ini
[scripts]
extra_plugin_directory = C:\Users\USERNAME\AppData\Roaming\OrganoidTracker\Plugins
```

This one contains the path to the location where user plugins are installed for OrganoidTracker. You can copy this from any other `organoid_tracker.ini` file. It's possible that this setting contains multiple paths; this happens if you are using multiple plugin directories.

-------------

Once you have created those two files, you can double-click the `labelfreetracker_predict_link_scores.bat` file. It will ask you first where you have stored the file containing positions that you want to link. This should be an AUT file, obtained from a previous step. (AUT is the default saving format of OrganoidTracker.)

Tip: you can simply drag an AUT file into the command prompt (like you would drag an image into ImageJ) to paste its path there. Once you have entered a path, press Enter.

Then you will be asked for a path to the training data. This should be an AUTLIST file containing one or more already-tracked datasets. It will use these files to determine what the chance is that two cell observations represent the same cell, given a certain movement distance. You can create an AUTLIST file in OrganoidTracker by loading multiple tracked time lapses (each in their own tab), and then using `File` -> `Tabs` -> `Export open tabs...`.

![Creating an AUTLIST file](https://user-images.githubusercontent.com/1462188/223542784-07738a89-482c-4576-a7e9-c81e55d75fcf.png)

Then, the script will update the `organoid_tracker.ini` file with your answers and with some default settings. The file should look something like this:

```ini
[scripts]
extra_plugin_directory = C:\Users\Username\AppData\Roaming\OrganoidTracker\Plugins

[DEFAULTS]

[set_link_scores_from_distance]
; In which file the positions stored, which will receive link scores?
input_file = "C:\path\to\input\file.aut"
; In which file is training data stored for linking?
training_file = "C:\path\to\training\file.autlist"
; Output file.
output_file = Scored positions.aut
; Beyond this Z level, positions are deleted. This speeds up things if you wouldn't need those positions anyways.
max_z = 999
```

Then, run the script again. You will then get a new AUT file. In this AUT file the positions won't be linked already, but it will contain the scores for doing so. Tracking is done in the next step, so please continue reading.

## Tracking cells over time
In this last step, you need an AUT file generated by the previous step. For this step, we will not be using any LabelFreeTracker-specific code. Instead, we will use the linker of OrganoidTracker, together with the link likelikhoods that we generated in the previous step.

Load the output file of the previous step into OrganoidTracker, as well as the images. Then, use `Tools` -> `Create links between time points...`. Save the folder somewhere, open it and run it. After a while, you'll have your tracks ready.

Note that LabelFreeTracker doesn't support automatic detection of divisions, so unfortunately you'll need to add those by hand. OrganoidTracker has an error checker, so when you load a file with tracks back in, it will assist you in finding errors. For us, the error correction was still a lot of work, since the tracking quality was not so good. We expect that label-free tracking will become easier over time, as prediction of nuclei becomes better, and better linking approaches become available. Still, we were able to track cells successfully over 60 hours in an organoid line without a nuclear marker, which was completely impossible before.


## Segmenting cells

Cell segmentation is implemented as a watershed transform based on the predicted cell membrane signal, with the predicted nucleus centers used as starting points. So you'll need to have both a predicted membrane signal and predicted nucleus centers to get started. If you don't have that data available yet, please see the respective sections on how to obtain those:

* [Predicting nucleus centers](#predicting-positions)
* [Predicting cell membrane images (and nucleus images)](#predicting-nucleus-images-or-cell-membrane-images)

To segment cells, use the following menu option: (`Tools` -> `From transmitted light` -> `Segment cells`)

![Segmenting images](https://github.com/user-attachments/assets/c85ea819-6836-4cf5-93e6-da578ad78c81)


Make sure you have the membrane channel as the current channel. Then simply select an output folder, and LabelFreeTracker will write so-called labeld images to that folder.

LabelFreeTracker does not provide any tools to analyze those images. However, for both Fiji and Napari plugins are available to analyze those, and if you know some Python you can also write your own scripts using `tifffile`, `scipy` or `scikit-image`. If you Google "how to work with labeled images in X", with X your favourite tool, you should be able to get started. For example, here's a render of the labeled image in Napari:

![Rendering of segmentation in Napari](https://github.com/user-attachments/assets/0c1f3f52-a81b-4fea-b01b-4ddd44e4f3a4)

