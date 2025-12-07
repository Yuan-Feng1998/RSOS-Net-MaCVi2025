# RSOS-Net

## RSOS-Net: Real-time Surface Obstacle Segmentation Network for Uncrewed Waterborne Vehicles

### Introduction
Due to water-surface reflection, wake and sun glitter, an unmanned waterborne vehicle (UWV) faces a long-standing challenge in identifying water-surface obstacles especially with small-scale appearance. Inspired by the encoder-decoder architecture, a real-time surface obstacle segmentation network (RSOS-Net) is created to enable online surface-obstacle detection for a UWV. Primarily, the improved lightweight feature pyramid network structure is deployed to flexibly accommodate significant scale-variations and enhance focus on small obstacles, simultaneously. To address visual ambiguities caused by water-surface disturbances, the fast pyramid pooling module (FPPM) and attention-based feature fusion module (AFFM) are holistically devised within lightweight encoder and decoder, respectively. Accordingly, the FPPM is able to distinguish obstacles from sun glitters by capturing both local and global contextual information via cascaded pooling, while the AFFM can rule out reflections by virtue of channel-spatial attention mechanism augmenting detailed features and spatial locations. Notably, the RSOS-Net secured first place in the the 3rd USV-based Embedded Obstacle Segmentation Challenge, with official results available at https://macvi.org/workshop/macvi25/summary.

### Proposed RSOS-Net Scheme
#### Overall Structure
<div align="center">
  <img src=".\overall_scheme\RSOS-Net.png" width="70%">  
</div>

## Getting started

### Dataset
Download the LaRS dataset: [https://lojzezust.github.io/lars-dataset/](https://lojzezust.github.io/lars-dataset/)  
Download the MODS dataset: [https://vision.fe.uni-lj.si/public/mods/](https://vision.fe.uni-lj.si/public/mods/)  

The dataset contains annotated images and corresponding obstacle labels for training and testing RSOS-Net. Please ensure that the data is downloaded and organized correctly according to the dataset instructions.

### Environment Configuration
This code is based on **mmsegmentation** framework. Please follow the steps below to configure the environment:

#### Create Conda Environment & Install Dependencies
```shell
# Create a new Conda environment named 'rsos-net' with Python 3.9, auto-confirm (-y) all prompts
conda create -n rsos-net python=3.9 -y

# Activate the Conda environment 'rsos-net' to use it for subsequent operations
conda activate rsos-net

# Install Python dependencies listed in the requirements.txt file using pip
pip install -r requirements.txt
```
### Training
Run the following commands to start training with different configurations:

If your computational resources are relatively abundant, it is recommended to use the rsos_r101_macvi config file as much as possible.
```shell
# Training with paper configuration (ResNet18 backbone)
# - Uses only feature information from 3 scales output by the backbone network
python tools/train.py rsos-net/rsos_r18.py

# Training with competition configuration (ResNet101 backbone)
# - Uses feature information from 4 scales output by the backbone network
# - Removes max-pooling from the channel attention module in AFFM
# - Recommended if your computational resources are sufficiently available
python tools/train.py rsos-net/rsos_r101_macvi.py
```
### Experimental Results  
#### Segmentation results on MODS & LaRS dataset
<table style="width: 100%; border-collapse: collapse;">
    <tr>
        <td style="padding: 10px; text-align: center;">
            <img src="./results_gif/Water%20surface%20reflection%20and%20Glare.gif?raw=true" 
                 style="width: 600px; height: auto;" 
                 alt="Sunlight Glare">
            <div style="font-style: italic; margin-top: 5px;">Fig. 1. Sunlight Glare</div>
        </td>
        <td style="padding: 10px; text-align: center;">
            <img src="./results_gif/Water%20Surface%20Reflection.gif?raw=true" 
                 style="width: 600px; height: auto;" 
                 alt="Reflection & Irregular Waterline">
            <div style="font-style: italic; margin-top: 5px;">Fig. 2. Reflection & Irregular Waterline</div>
        </td>
    </tr>
    <tr>
        <td style="padding: 10px; text-align: center;">
            <img src="./results_gif/Multi-scale%20Obstacles.gif?raw=true" 
                 style="width: 600px; height: auto;" 
                 alt="Multi-scale Obstacles">
            <div style="font-style: italic; margin-top: 5px;">Fig. 3. Multi-scale Obstacles</div>
        </td>
        <td style="padding: 10px; text-align: center;">
            <img src="./results_gif/Wake%20and%20Water%20Surface%20Reflection.gif?raw=true" 
                 style="width: 600px; height: auto;" 
                 alt="Wake & Reflection">
            <div style="font-style: italic; margin-top: 5px;">Fig. 4. Wake & Reflection</div>
        </td>
    </tr>
    <tr>
        <td style="padding: 10px; text-align: center;">
            <img src="./results_gif/glare.gif?raw=true" 
                 style="width: 600px; height: auto;" 
                 alt="Sunlight Glare & low light">
            <div style="font-style: italic; margin-top: 5px;">Fig. 5. Sunlight Glare & Low Light</div>
        </td>
        <td style="padding: 10px; text-align: center;">
            <img src="./results_gif/dark.gif?raw=true" 
                 style="width: 600px; height: auto;" 
                 alt="Multi-scale Obstacles & low light">
            <div style="font-style: italic; margin-top: 5px;">Fig. 6. Multi-scale Obstacles & Low Light</div>
        </td>
    </tr>
</table>

### Citation
If you use RSOS-Net in your research, please cite the following paper. A star on my GitHub repository would also be greatly appreciated:

```
@ARTICLE{11267114,
  author={Wang, Ning and Feng, Yuan and Tian, Lixin and Wei, Yi},
  journal={IEEE Transactions on Intelligent Transportation Systems}, 
  title={RSOS-Net: Real-Time Surface Obstacle Segmentation Network for Uncrewed Waterborne Vehicles}, 
  year={2025},
  volume={},
  number={},
  pages={1-14},
  doi={10.1109/TITS.2025.3628677}
}

```