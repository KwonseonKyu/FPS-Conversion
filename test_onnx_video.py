import cv2
import numpy as np
import onnxruntime as ort
from tqdm import tqdm
import os
import argparse

def split_into_tiles(frame, tile_size=512):
    tiles = []
    h, w, _ = frame.shape
    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
            tile = frame[y:min(y+tile_size, h), x:min(x+tile_size, w)]
            tiles.append((tile, x, y))
    return tiles

def merge_tiles(tiles, output_size):
    output_image = np.zeros(output_size, dtype=np.uint8)
    for tile, x, y in tiles:
        h, w, _ = tile.shape
        output_image[y:y+h, x:x+w] = tile
    return output_image

def interpolate_frames(frames, model_path, scale_factor, use_gpu=True):
    if use_gpu:
        providers = ['CUDAExecutionProvider']
    else:
        providers = ['CPUExecutionProvider']
    
    try:
        ort_session = ort.InferenceSession(model_path, providers=providers)
        provider_used = ort_session.get_providers()[0]
    except Exception as e:
        print(f"Failed to load with providers {providers}: {e}")
        print("Falling back to CPUExecutionProvider")
        ort_session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        provider_used = 'CPUExecutionProvider'
    
    print(f"Using provider: {provider_used}")
    
    
    num_iterations = int(np.log2(scale_factor))
    
    for iteration in range(num_iterations):
        new_frames = []
        for i in tqdm(range(len(frames) - 1), desc=f"Interpolating frames (Iteration {iteration + 1}/{num_iterations})"):
            frame1 = frames[i].astype(np.float32) / 255.0
            frame2 = frames[i + 1].astype(np.float32) / 255.0
            
            tiles1 = split_into_tiles(frame1)
            tiles2 = split_into_tiles(frame2)
            interpolated_tiles = []

            for (tile1, x, y), (tile2, _, _) in zip(tiles1, tiles2):
                resized_tile1 = cv2.resize(tile1, (512, 512))
                resized_tile2 = cv2.resize(tile2, (512, 512))
                
                resized_tile1 = np.transpose(resized_tile1, (2, 0, 1))[np.newaxis, :]
                resized_tile2 = np.transpose(resized_tile2, (2, 0, 1))[np.newaxis, :]

                inputs = {'input0': resized_tile1, 'input1': resized_tile2}
                
                interpolated_tile = ort_session.run(None, inputs)[0]
                interpolated_tile = np.transpose(interpolated_tile[0], (1, 2, 0))
                interpolated_tile = np.clip(interpolated_tile * 255.0, 0, 255).astype(np.uint8)
                
                # Resize the tile back to the original size
                interpolated_tile = cv2.resize(interpolated_tile, (tile1.shape[1], tile1.shape[0]))
                
                interpolated_tiles.append((interpolated_tile, x, y))
            
            merged_frame = merge_tiles(interpolated_tiles, frames[i].shape)
            new_frames.append(frames[i])
            new_frames.append(merged_frame)
        new_frames.append(frames[-1])
        frames = new_frames
    
    return frames

def generate_video(input_video_path, output_dir, model_path, scale_factor, mode, use_gpu=True):
    cap = cv2.VideoCapture(input_video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file {input_video_path}")
        return
    
    input_video_name = os.path.splitext(os.path.basename(input_video_path))[0]
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for _ in tqdm(range(total_frames), desc="Reading frames"):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    
    cap.release()

    if len(frames) == 0:
        print("Error: No frames were read from the video.")
        return

    interpolated_frames = interpolate_frames(frames, model_path, scale_factor, use_gpu=use_gpu)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_video_path = os.path.join(output_dir, f"{input_video_name}_{mode}_{scale_factor}x.mp4")
    
    if mode == "slow_motion":
        out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (original_width, original_height))
    elif mode == "frame_rate_increase":
        new_fps = fps * scale_factor
        out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), new_fps, (original_width, original_height))


    for frame in tqdm(interpolated_frames, desc=f"Writing frames ({mode})"):
        out.write(frame)

    out.release()
    print(f"Video saved to {output_video_path}")

def print_model_input_names(model_path):
    ort_session = ort.InferenceSession(model_path)
    input_names = [input.name for input in ort_session.get_inputs()]
    print(f"Model input names: {input_names}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Frame interpolation video generation")
    parser.add_argument('--input_video_path', type=str, default='./video_inputs/Test_video.mp4', help="Path to the input video")
    parser.add_argument('--model_path', type=str, default='./interpolation.onnx' , help="Path to the ONNX model")
    parser.add_argument('--scale_factor', type=int, required=True, choices=[2, 4, 8, 16], help='Interpolation scale factor (2x, 4x, 8x, 16x)')
    parser.add_argument('--mode', type=str, required=True, choices=['slow_motion', 'frame_rate_increase'], help="Mode for video generation")
    parser.add_argument('--output_dir', type=str, help="Directory to save the output video")
    parser.add_argument('--use_gpu', action='store_true', help="Use GPU for inference if available")

    args = parser.parse_args()
    
    if not args.output_dir:
        args.output_dir = f"./video_outputs/{args.mode}/scale_factor_{args.scale_factor}"
        
    print(f"Output directory is set to: {args.output_dir}")

    print_model_input_names(args.model_path)

    generate_video(args.input_video_path, args.output_dir, args.model_path, args.scale_factor, args.mode, use_gpu=args.use_gpu)
