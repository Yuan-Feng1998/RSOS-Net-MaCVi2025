# RSOS-Net

## RSOS-Net: Real-time Surface Obstacle Segmentation Network for Unmanned Waterborne Vehicles

### Introduction

Due to water-surface reflection, wake and sun glitter, an unmanned waterborne vehicle (UWV) faces a long-standing challenge in identifying water-surface obstacles especially with small-scale appearance. In this paper, inspired by the encoder-decoder architecture, a real-time surface obstacle segmentation network (RSOS-Net) is created to enable online surface-obstacle detection for a UWV. Primarily, the improved lightweight feature pyramid network structure is deployed to flexibly accommodate significant scale-variations and enhance focus on small obstacles, simultaneously. To address visual ambiguities caused by water-surface disturbances, the fast pyramid pooling module (FPPM) and attention-based feature fusion module (AFFM) are holistically devised within lightweight encoder and decoder, respectively. Accordingly, the FPPM is able to distinguish obstacles from sun glitters by capturing both local and global contextual information via cascaded pooling, while the AFFM can rule out reflections by virtue of channel-spatial attention mechanism augmenting detailed features and spatial locations. Notably, the RSOS-Net secured first place in the the 3rd USV-based Embedded Obstacle Segmentation Challenge, with official results available at \url{https://macvi.org/workshop/macvi25/summary}.

### Proposed RSOS-Net Scheme
#### Overall Structure
<div align="center">
  <img src="https://github.com/Yuan-Feng1998/RSOS-Net-MaCVi2025/blob/main/overall_scheme/RSOS-Net.png" width="70%">  
</div>

### Data

Download the LaRS dataset: [https://lojzezust.github.io/lars-dataset/](https://lojzezust.github.io/lars-dataset/)

The dataset contains annotated images and corresponding obstacle labels for training and testing RSOS-Net. Please ensure that the data is downloaded and organized correctly according to the dataset instructions.

### Experimental Results  
#### Segmentation results on MODS dataset   
| ![Fig. 1](https://github.com/Yuan-Feng1998/RSOS-Net-MaCVi2025/blob/main/results_gif/Water%20Surface%20Reflection%20and%20Glare.gif) <br> *Sunlight Glare* | ![Fig. 2](https://github.com/Yuan-Feng1998/RSOS-Net-MaCVi2025/blob/main/results_gif/Water%20Surface%20Reflection.gif) <br> *Reflection & Irregular Waterline*  |
| :------------------: | :------------------: |
| ![Fig. 3](https://github.com/Yuan-Feng1998/RSOS-Net-MaCVi2025/blob/main/results_gif/Multi-scal%20Obstacles.gif) <br> *Multi-scale Obstacles* | ![Fig. 4](https://github.com/Yuan-Feng1998/RSOS-Net-MaCVi2025/blob/main/results_gif/Wake%20and%20Water%20Surface%20Reflection.gif) <br> *Wake & Reflection* |

### Competition Results
The RSOS-Net has completed the competition and ranked first in the embedded obstacle segmentation competition based on USV. The code will be uploaded as soon as possible.

For the leaderboard details, please visit：[https://macvi.org/leaderboard/surface/lars/embedded-challenge](https://macvi.org/leaderboard/surface/lars/embedded-challenge)

![Current Ranking](https://github.com/Yuan-Feng1998/RSOS-Net2024/blob/main/Rank.png)
