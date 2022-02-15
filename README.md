# selfmono-sceneflow (NOT Official)
[Self-Supervised Monocular Scene Flow Estimation](https://openaccess.thecvf.com/content_CVPR_2020/papers/Hur_Self-Supervised_Monocular_Scene_Flow_Estimation_CVPR_2020_paper.pdf)  
[Official Github](https://github.com/visinf/self-mono-sf)  
Reproducing code of Self-Supervised Monocular Scene Flow Estimation (CVPR 2020)  
Reorganizing the training pipeline again.  
The backbone and module are from the official code.  
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
This repository only eigen split training
## Inference
note-evaluation.ipynb
## Metric