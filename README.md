# E-3DTrack
This repository is for the **CVPR 2024** paper *"3D Feature Tracking via Event Camera"*. This work is inspired by the great success of "Data-Driven Feature Tracking for Event Cameras" (CVPR 2023 Best Paper Candidate) and extends it to 3D.

## Requirements
1. Python 3.8 with the following packages installed:
   * einops==0.4.1
   * kornia==0.6.7
   * opencv-python==4.6.0.66
   * torch==1.9.0
   * tqdm==4.64.0
3. cuda
   - **CUDA** enabled **GPUs** are required for training. We train our code with CUDA 11.1 V11.1.105 on A100 GPUs and test on NVIDIA 3090 GPUs.


## Data preparing
1. Our E-3DTrack dataset could be downloaded from https://github.com/lisiqi19971013/event-based-datasets.
2. Download the pre-trained model from https://drive.google.com/file/d/1Gx0zhIeciHGEqrRryPmAC-mqoNO1wuMQ/view?usp=sharing or from https://pan.baidu.com/s/1ONvkUyk2cqWM2XR_XwaKeg (extract code: 2024).


## Evaluation
1. Modify the variables "**ckpt_path**" and "**data_folder**" in the file "**eval.py**" accordingly.

Run the following code to generate output results.

   ```shell
   >>> python eval.py
   ```

The output predictions could be found at "./output"

2. Calculate metrics using the following code.

```shell
>>> python calMetric.py
```


## Citation
```bib
@inproceedings{e3dtrack,
    title={3D Feature Tracking via Event Camera}, 
    author={Li, Siqi and Zhou, Zhikuan and Xue Zhou and Li, Yipeng and Du, Shaoyi and Gao, Yue},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition}, 
    year={2024},
}
```
