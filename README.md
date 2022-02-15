# selfmono-sceneflow (NOT Official)
[Self-Supervised Monocular Scene Flow Estimation](https://openaccess.thecvf.com/content_CVPR_2020/papers/Hur_Self-Supervised_Monocular_Scene_Flow_Estimation_CVPR_2020_paper.pdf)  
[Official Github](https://github.com/visinf/self-mono-sf)  
Reproducing of Self-Supervised Monocular Scene Flow Estimation (CVPR 2020)  
Reorganizing the training pipeline again (For me)    
The backbone and module are from the official code    
I annotated and checked the function to understrand this pipeline
## Visualization
![disp](https://github.com/Doyosae/Self-Supervised-Monocular-Sceneflow-Estimation/blob/main/demo/disp.png)
![flow](https://github.com/Doyosae/Self-Supervised-Monocular-Sceneflow-Estimation/blob/main/demo/flow.png)
![scene](https://github.com/Doyosae/Self-Supervised-Monocular-Sceneflow-Estimation/blob/main/demo/scene.png)
## Requirements
```
Installation problem in some environments.
../model_layer/correlation_package/setup.py
../model_layer/forwardwarp_package/setup.py

cxx_args = ['-std=c++11'] -> cxx_args = ['-std=c++14']

bash install_modules.sh
```
## Dataset
[KITTI raw data](http://www.cvlibs.net/datasets/kitti/raw_data.php)  
[KITTI Scene Flow 2015 data](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=flow)  
Dataset follows convention of [monodepth](https://github.com/mrharicot/monodepth)  
```
Convert KITTI RAW png to jpeg

find (data_folder)/ -name '*.png' | parallel 'convert {.}.png {.}.jpg && rm {}'
```
## Training
```
This repository only support eigen split training

bash train1.sh
```
## Inference
```
note-evaluation.ipynb
```
## Metric
Metric only reported for eigen split  
```
Depth Estimation

Paper report
0.125   0.978   4.877   0.208   0.851   0.950   0.978

This repo + official eigen weight
0.125   0.975   4.878   0.209   0.850   0.950   0.978

This repo + retrain eigen weight
0.131   1.000   4.923   0.214   0.842   0.945   0.978



Sceneflow Estimation

Paper report
None

This repo + official eigen weight
d1 0.230   d2 0.277   f1 0.188   sf 0.377

This repo + retrain eigen weight
d1 0.220   d2 0.268   f1 0.232   sf 0.388 
```