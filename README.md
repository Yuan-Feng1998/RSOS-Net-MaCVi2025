# RSOS-Net

## RSOS-Net: Real-time Surface Obstacle Segmentation Network for Unmanned Waterborne Vehicles

### Introduction

Due to water-surface reflection, wake and sun glitter, an unmanned waterborne vehicle (UWV) faces a long-standing challenge in identifying water-surface obstacles especially with small-scale appearance. Inspired by the encoder-decoder architecture, a real-time surface obstacle segmentation network (RSOS-Net) is created to enable online surface-obstacle detection for a UWV. Primarily, the improved lightweight feature pyramid network structure is deployed to flexibly accommodate significant scale-variations and enhance focus on small obstacles, simultaneously. To address visual ambiguities caused by water-surface disturbances, the fast pyramid pooling module (FPPM) and attention-based feature fusion module (AFFM) are holistically devised within lightweight encoder and decoder, respectively. Accordingly, the FPPM is able to distinguish obstacles from sun glitters by capturing both local and global contextual information via cascaded pooling, while the AFFM can rule out reflections by virtue of channel-spatial attention mechanism augmenting detailed features and spatial locations. Notably, the RSOS-Net secured first place in the the 3rd USV-based Embedded Obstacle Segmentation Challenge, with official results available at https://macvi.org/workshop/macvi25/summary.

### Proposed RSOS-Net Scheme
#### Overall Structure
<div align="center">
  <img src="https://github.com/Yuan-Feng1998/RSOS-Net-MaCVi2025/blob/main/overall_scheme/RSOS-Net.png" width="95%">  
</div>

### Experimental Results  
#### Segmentation results on MODS & LaRS dataset
<!DOCTYPE html>
<html>
<head>
<body>

<table style="width: 100%; border-collapse: collapse;">
    <tr>
        <td style="padding: 10px; text-align: center;">
            <img src="https://github.com/Yuan-Feng1998/RSOS-Net-MaCVi2025/blob/main/results_gif/Water%20surface%20reflection%20and%20Glare.gif?raw=true" 
                 style="width: 600px; height: auto;" 
                 alt="Sunlight Glare">
            <div style="font-style: italic; margin-top: 5px;">Fig. 1. Sunlight Glare</div>
        </td>
        <td style="padding: 10px; text-align: center;">
            <img src="https://github.com/Yuan-Feng1998/RSOS-Net-MaCVi2025/blob/main/results_gif/Water%20Surface%20Reflection.gif?raw=true" 
                 style="width: 600px; height: auto;" 
                 alt="Reflection & Irregular Waterline">
            <div style="font-style: italic; margin-top: 5px;">Fig. 2. Reflection & Irregular Waterline</div>
        </td>
    </tr>
    <tr>
        <td style="padding: 10px; text-align: center;">
            <img src="https://github.com/Yuan-Feng1998/RSOS-Net-MaCVi2025/blob/main/results_gif/Multi-scale%20Obstacles.gif?raw=true" 
                 style="width: 600px; height: auto;" 
                 alt="Multi-scale Obstacles">
            <div style="font-style: italic; margin-top: 5px;">Fig. 3. Multi-scale Obstacles</div>
        </td>
        <td style="padding: 10px; text-align: center;">
            <img src="https://github.com/Yuan-Feng1998/RSOS-Net-MaCVi2025/blob/main/results_gif/Wake%20and%20Water%20Surface%20Reflection.gif?raw=true" 
                 style="width: 600px; height: auto;" 
                 alt="Wake & Reflection">
            <div style="font-style: italic; margin-top: 5px;">Fig. 4. Wake & Reflection</div>
        </td>
    </tr>
    <tr>
        <td style="padding: 10px; text-align: center;">
            <img src="https://github.com/Yuan-Feng1998/RSOS-Net-MaCVi2025/blob/main/results_gif/glare.gif?raw=true" 
                 style="width: 600px; height: auto;" 
                 alt="Sunlight Glare & low light">
            <div style="font-style: italic; margin-top: 5px;">Fig. 5. Sunlight Glare & Low Light</div>
        </td>
        <td style="padding: 10px; text-align: center;">
            <img src="https://github.com/Yuan-Feng1998/RSOS-Net-MaCVi2025/blob/main/results_gif/dark.gif?raw=true" 
                 style="width: 600px; height: auto;" 
                 alt="Multi-scale Obstacles & low light">
            <div style="font-style: italic; margin-top: 5px;">Fig. 6. Multi-scale Obstacles & Low Light</div>
        </td>
    </tr>
</table>

</body>
</html>

### Data

Download the LaRS dataset: [https://lojzezust.github.io/lars-dataset/](https://lojzezust.github.io/lars-dataset/)

Download the MODS dataset: [https://vision.fe.uni-lj.si/public/mods/](https://vision.fe.uni-lj.si/public/mods/)

The dataset contains annotated images and corresponding obstacle labels for training and testing RSOS-Net. Please ensure that the data is downloaded and organized correctly according to the dataset instructions.

### Code
The code will be uploaded as soon as possible.

### Competition Results
The RSOS-Net has completed the competition and ranked first in the embedded obstacle segmentation competition based on USV. For the leaderboard details, please visit：[https://macvi.org/leaderboard/surface/lars/embedded-challenge](https://macvi.org/leaderboard/surface/lars/embedded-challenge)


