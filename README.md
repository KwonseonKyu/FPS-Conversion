# FPS-Conversion

## Overview
Create an intermediate frame with the video frame interpolation onnx model and perform two options with the corresponding generated frame. (Slow_motion video generation & Frame rate increase)

The output of the resulting video depends on the scale factor (2x, 4x, 8x, 16x).

## Requirements
- CUDA 11.8.0
- cuDNN 8.9.6
- Python 3.8.19
- Pytorch


## Installation

Download repository:
```bash
git clone https://github.com/KwonseonKyu/FPS-Conversion.git
```

```bash
conda create -n fisf python=3.8.19
conda activate fisf
pip install -r requirements.txt
```

## Download ONNX model

Download pretrained onnx model [ONNX model](https://drive.google.com/file/d/1-MIVhCToz8_IKC1B9k47uHtg1cMNyXNO/view?usp=sharing)


## File Paths


## Test video

Run the following command for training:

```bash
python test_video.py --input_video_path <path to video> --model_path <path to onnx model> --scale_factor <choice: 2, 4, 8, 16> --mode <choice: slow_motion, frame_rate_increase> --output_dir <path to output folder> --use_gpu <If you use cpu, you don't have to write it>
```
