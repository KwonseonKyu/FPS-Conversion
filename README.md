# FPS-Conversion

## Overview
Create an intermediate frame with the video frame interpolation onnx model and perform two options with the corresponding generated frame. (Slow_motion video generation & Frame rate increase)

The output of the resulting video depends on the scale factor (2x, 4x, 8x, 16x).

## Requirements
- CUDA 11.8.0
- cuDNN 8.x
- Python 3.8.19
- onnxruntime-gpu 1.18.0

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

## Download 

Download pretrained [ONNX model](https://drive.google.com/file/d/1-MIVhCToz8_IKC1B9k47uHtg1cMNyXNO/view?usp=sharing)


## File Paths

The 'video_outputs' folder is automatically generated when a test is run.

```bash
.
├── video_outputs
│   ├── frame_rate_increase
│   │   ├── scale_factor_2
│   │   ├── scale_factor_4
│   │   ├── scale_factor_8
│   │   └── scale_factor_16
│   └── slow_motion
│   │   ├── scale_factor_2
│   │   ├── scale_factor_4
│   │   ├── scale_factor_8
│   │   └── scale_factor_16
├── video_inputs
│   └── Test_video.mp4
├── experiments
├── interpolation.onnx
├── requirements.txt
└── test_onnx_video.py
```

## Test video

```bash
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Frame interpolation video generation")
    parser.add_argument('--input_video_path', type=str, default='./video_inputs/Test_video.mp4', help="Path to the input video")
    parser.add_argument('--model_path', type=str, default='./interpolation.onnx' , help="Path to the ONNX model")
    parser.add_argument('--scale_factor', type=int, required=True, choices=[2, 4, 8, 16], help='Interpolation scale factor (2x, 4x, 8x, 16x)')
    parser.add_argument('--mode', type=str, required=True, choices=['slow_motion', 'frame_rate_increase'], help="Mode for video generation")
    parser.add_argument('--output_dir', type=str, help="Directory to save the output video")
    parser.add_argument('--use_gpu', action='store_true', help="Use GPU for inference if available")
```

Run the following command for test:

```bash
python test_video.py  --scale_factor <choice: 2, 4, 8, 16> --mode <choice: slow_motion, frame_rate_increase> --use_gpu <If you use cpu, you don't have to write it>
```


## Test example (2x)

### Input video
![Test_video_3](https://github.com/user-attachments/assets/8c25817a-3cdd-4c5f-ba6d-13286b3ca9be)


### Slow_motion video(2x)
![Test_video_3_slow_motion_2x](https://github.com/user-attachments/assets/bc5ff3d4-6d05-483e-8798-d6db84d82a91)


### Frame rate increase(2x)
![Test_video_3_frame_rate_increase_2x](https://github.com/user-attachments/assets/231c1053-f9e7-4da0-be2c-5f34895066ae)


