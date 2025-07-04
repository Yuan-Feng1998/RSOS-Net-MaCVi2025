# RSOS-Net

## RSOS-Net: Real-time Surface Obstacle Segmentation Network for Unmanned Waterborne Vehicles

### Introduction

RSOS-Net is a real-time surface obstacle segmentation network designed for unmanned waterborne vehicles. Utilizing deep learning and computer vision techniques, it accurately segments surface obstacles, providing crucial information for autonomous navigation and safe operation of unmanned waterborne vehicles.

### Data

Download the LaRS dataset: [https://lojzezust.github.io/lars-dataset/](https://lojzezust.github.io/lars-dataset/)

The dataset contains annotated images and corresponding obstacle labels for training and testing RSOS-Net. Please ensure that the data is downloaded and organized correctly according to the dataset instructions.

### Experimental Results  
#### Segmentation results on MODS dataset   
<div style="max-width: 800px;  /* 设定容器最大宽度 */
            margin: 0 auto;     /* 水平居中 */
            display: grid; 
            grid-template-columns: repeat(2, 1fr); 
            gap: 20px; 
            justify-items: center;">
  <!-- 第一行动图1-2 -->
  <div style="text-align: center;">
    <img src="https://github.com/Yuan-Feng1998/RSOS-Net-MaCVi2025/blob/main/results_gif/Multi-scal%20Obstacles.gif" width="350">
    <p style="font-size: 14px; margin-top: 5px;"><em>Fig. 1: Multi-scale Obstacles</em></p>
  </div>
  
  <div style="text-align: center;">
    <img src="https://github.com/Yuan-Feng1998/RSOS-Net-MaCVi2025/blob/main/results_gif/Wake%20and%20Water%20Surface%20Reflection.gif" width="350">
    <p style="font-size: 14px; margin-top: 5px;"><em>Fig. 2: Wake & Reflection</em></p>
  </div>
  
  <!-- 第二行动图3-4 -->
  <div style="text-align: center;">
    <img src="https://github.com/Yuan-Feng1998/RSOS-Net-MaCVi2025/blob/main/results_gif/Water%20Surface%20Reflection%20and%20Glare.gif" width="350">
    <p style="font-size: 14px; margin-top: 5px;"><em>Fig. 3: Sunlight Glare</em></p>
  </div>
  
  <div style="text-align: center;">
    <img src="https://github.com/Yuan-Feng1998/RSOS-Net-MaCVi2025/blob/main/results_gif/Wake%20and%20Water%20Surface%20Reflection.gif" width="350">
    <p style="font-size: 14px; margin-top: 5px;"><em>Fig. 4: Reflection & Irregular Waterline</em></p>
  </div>
</div>

### Competition Results
The RSOS-Net has completed the competition and ranked first in the embedded obstacle segmentation competition based on USV. The code will be uploaded as soon as possible.

For the leaderboard details, please visit：[https://macvi.org/leaderboard/surface/lars/embedded-challenge](https://macvi.org/leaderboard/surface/lars/embedded-challenge)

![Current Ranking](https://github.com/Yuan-Feng1998/RSOS-Net2024/blob/main/Rank.png)
