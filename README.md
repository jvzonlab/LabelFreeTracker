# LabelFreeTracker
Predicts nuclear and cell envelope reporters from transmitted light images. This allows you to view nuclei and cell envelopes in organoids without fluorescent markers, allowing you to segment and track those cells.

![Method overview](https://user-images.githubusercontent.com/1462188/222784969-bc1b02a0-a0a3-459c-92c9-10b9cc5e16a4.png)

## Installation
Installation was tested on Microsoft Windows only. For installation, please follow the following steps:

1. LabelFreeTracker has been developed as a plugin for [OrganoidTracker](https://github.com/jvzonlab/OrganoidTracker). To use LabelFreeTracker, first follow the installation instructions of OrganoidTracker, and verify that you can open the program.
2. Then, [download LabelFreeTracker](https://github.com/jvzonlab/LabelFreeTracker/archive/refs/heads/main.zip). The download will contain a `LabelFreeTracker-main` folder, which in turn contains a `Plugins for OrganoidTracker` folder.
3. Next, open OrganoidTracker, and then use the `File` menu to open the folder that contains the plugins:

  ![How to open the plugins folder](https://user-images.githubusercontent.com/1462188/222796147-380612db-54da-44ab-aebe-f7825a02643f.png)

4. Now place the files *inside* the `Plugins for OrganoidTracker` folder inside the plugins folder of OrganoidTracker, like this:

  ![Drag and drop the files](https://user-images.githubusercontent.com/1462188/222797179-22a5e81e-feb9-41d0-8023-281a917e67ec.png)

5. Now back in OrganoidTracker, use `File` -> `Reload all plugins...`. If everything went successfully, you will now have four new menu options:

  ![New menu options](https://user-images.githubusercontent.com/1462188/222797841-2730abd9-0af1-485e-975d-089559a4ff87.png)


## Training a neural network for nucleus or cell membrane prediction



## Predicting nucleus or cell membrane images



## Training a neural network for position detection



## Predicting positions


## Predicting links over time



## Tracking cells over time

