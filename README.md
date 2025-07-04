# RSOS-Net

## RSOS-Net: Real-time Surface Obstacle Segmentation Network for Unmanned Waterborne Vehicles

### Introduction

RSOS-Net is a real-time surface obstacle segmentation network designed for unmanned waterborne vehicles. Utilizing deep learning and computer vision techniques, it accurately segments surface obstacles, providing crucial information for autonomous navigation and safe operation of unmanned waterborne vehicles.

### Data

Download the LaRS dataset: [https://lojzezust.github.io/lars-dataset/](https://lojzezust.github.io/lars-dataset/)

The dataset contains annotated images and corresponding obstacle labels for training and testing RSOS-Net. Please ensure that the data is downloaded and organized correctly according to the dataset instructions.

### Experimental Results  
#### Segmentation results on MODS dataset   
<div style="max-width: 800px;
            margin: 0 auto;
            display: flex;
            flex-wrap: wrap;
            justify-content: center;
            gap: 20px;">
  <!-- 第一行动图1-2 -->
  <div style="flex: 0 0 48%;  /* 单图占比48%，留2%间隙 */
              text-align: center;">
    <img src="url1" width="350">
    <p><em>Fig. 1</em></p>
  </div>
  
  <div style="flex: 0 0 48%;
              text-align: center;">
    <img src="url2" width="350">
    <p><em>Fig. 2</em></p>
  </div>
  
  <!-- 第二行动图3-4 -->
  <div style="flex: 0 0 48%;
              text-align: center;">
    <img src="url3" width="350">
    <p><em>Fig. 3</em></p>
  </div>
  
  <div style="flex: 0 0 48%;
              text-align: center;">
    <img src="url4" width="350">
    <p><em>Fig. 4</em></p>
  </div>
</div>

### Competition Results
The RSOS-Net has completed the competition and ranked first in the embedded obstacle segmentation competition based on USV. The code will be uploaded as soon as possible.

For the leaderboard details, please visit：[https://macvi.org/leaderboard/surface/lars/embedded-challenge](https://macvi.org/leaderboard/surface/lars/embedded-challenge)

![Current Ranking](https://github.com/Yuan-Feng1998/RSOS-Net2024/blob/main/Rank.png)
