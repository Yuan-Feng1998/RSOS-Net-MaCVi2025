# RSOS-Net

## RSOS-Net: Real-time Surface Obstacle Segmentation Network for Unmanned Waterborne Vehicles

### Introduction

RSOS-Net is a real-time surface obstacle segmentation network designed for unmanned waterborne vehicles. Utilizing deep learning and computer vision techniques, it accurately segments surface obstacles, providing crucial information for autonomous navigation and safe operation of unmanned waterborne vehicles.

### Data

Download the LaRS dataset: [https://lojzezust.github.io/lars-dataset/](https://lojzezust.github.io/lars-dataset/)

The dataset contains annotated images and corresponding obstacle labels for training and testing RSOS-Net. Please ensure that the data is downloaded and organized correctly according to the dataset instructions.

### Experimental Results  
#### Segmentation results on MODS dataset   

| ![Fig. 1](https://github.com/Yuan-Feng1998/RSOS-Net-MaCVi2025/blob/main/results_gif/Multi-scal%20Obstacles.gif) <br> *Fig. 1: Multi-scale Obstacles* | ![Fig. 2](https://github.com/Yuan-Feng1998/RSOS-Net-MaCVi2025/blob/main/results_gif/Wake%20and%20Water%20Surface%20Reflection.gif) <br> *Fig. 2: Wake & Reflection* |
|-----------------------------------------------------|--------------------------------------------------|
| ![Fig. 3](https://github.com/Yuan-Feng1998/RSOS-Net-MaCVi2025/blob/main/results_gif/Water%20Surface%20Reflection%20and%20Glare.gif) <br> *Fig. 3: Sunlight Glare* | ![Fig. 4](https://github.com/Yuan-Feng1998/RSOS-Net-MaCVi2025/blob/main/results_gif/Water%20Surface%20Reflection.gif.gif) <br> *Fig. 4: Reflection & Irregular Waterline*  |

### Competition Results
The RSOS-Net has completed the competition and ranked first in the embedded obstacle segmentation competition based on USV. The code will be uploaded as soon as possible.

For the leaderboard details, please visit：[https://macvi.org/leaderboard/surface/lars/embedded-challenge](https://macvi.org/leaderboard/surface/lars/embedded-challenge)

![Current Ranking](https://github.com/Yuan-Feng1998/RSOS-Net2024/blob/main/Rank.png)
