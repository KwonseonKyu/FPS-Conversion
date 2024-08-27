##############################
### Convert 512 x 512 size ###
##############################


import torch
import torch.onnx
import argparse
import numpy as np
import os
from PIL import Image
from datas.utils import imread_rgb
from models import make_model, model_profile
from utils.config import make_config

# Pytorch tensor -> Numpy array
def to_numpy(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    return tensor 

# preprocess 
def preprocess(image_path, input_size):
    image = imread_rgb(image_path)
    image = np.array(image).astype(np.float32) / 255.0  
    
    image = Image.fromarray((image * 255).astype(np.uint8))  
    image = image.resize((input_size[3], input_size[2]))  # Resize to input size
    image = np.array(image).astype(np.float32) / 255.0  
    
    image = np.transpose(image, (2, 0, 1))  # (H, W, C) -> (C, H, W)
    image = np.expand_dims(image, axis=0)  # (C, H, W) -> (1, C, H, W)
    
    return image

# postprocess 
def postprocess(output_tensor):
    output = output_tensor[0]  # Remove batch dimension
    output = np.clip(output, 0, 1)  
    output = np.transpose(output, (1, 2, 0))  
    output = (output * 255).astype(np.uint8)  
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/test.yaml', help="Path to the config file")
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--im0', type=str, default='./inputs/images/0.png', help='First image')
    parser.add_argument('--im1', type=str, default='./inputs/images/2.png', help='Second image')
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--onnx_path', type=str, default='./onnx/interpolation.onnx', help='Path to save ONNX model')
    parser.add_argument('--scale_factor', type=int, default=2, choices=[2, 4, 8, 16], help='Interpolation scale factor (2x, 4x, 8x)')
    parser.add_argument('--input_size', type=str, default="[1, 3, 512, 512]", help='Input size as [batch_size, channels, height, width]')
    args = parser.parse_args()

    # Convert input_size from string to list of ints
    input_size = [int(dim) for dim in args.input_size.strip('[]').split(',')]

    cfg_file = args.config
    dev_id = args.gpu_id
    torch.cuda.set_device(dev_id)

    cfg = make_config(cfg_file, launch_experiment=False)
    print(model_profile(cfg.model))

    model = make_model(cfg.model)
    model.cuda()
    model.eval()

    im0_np = preprocess(args.im0, input_size)
    im1_np = preprocess(args.im1, input_size)

    print("Start ONNX export...")
    dummy_input = (torch.from_numpy(im0_np).cuda(), torch.from_numpy(im1_np).cuda())
    torch.onnx.export(model,
                      dummy_input,
                      args.onnx_path,
                      export_params=True,
                      opset_version=18,
                      do_constant_folding=True,
                      input_names=['input0', 'input1'],
                      output_names=['output'],
                      dynamic_axes={'input0': {0: 'batch_size', 2: 'height', 3: 'width'},
                                    'input1': {0: 'batch_size', 2: 'height', 3: 'width'},
                                    'output': {0: 'batch_size', 2: 'height', 3: 'width'}})
    print(f"Completed ONNX export >> {args.onnx_path}")

###########################################################################################################

#############################
### Convert original size ###
#############################

# import torch
# import torch.onnx
# import argparse
# import numpy as np
# import os
# from PIL import Image
# from datas.utils import imread_rgb
# from models import make_model, model_profile
# from utils.config import make_config

# # Pytorch tensor -> Numpy array
# def to_numpy(tensor):
#     if isinstance(tensor, torch.Tensor):
#         return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
#     return tensor 

# # preprocess 
# def preprocess(image_path):
#     image = imread_rgb(image_path)
#     image = np.array(image).astype(np.float32) / 255.0  
    
#     image = np.transpose(image, (2, 0, 1))  # (H, W, C) -> (C, H, W)
#     image = np.expand_dims(image, axis=0)  # (C, H, W) -> (1, C, H, W)
    
#     return image

# # postprocess 
# def postprocess(output_tensor):
#     output = output_tensor[0]  # Remove batch dimension
#     output = np.clip(output, 0, 1)  
#     output = np.transpose(output, (1, 2, 0))  
#     output = (output * 255).astype(np.uint8)  
#     return output

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--config', default='./configs/test.yaml', help="Path to the config file")
#     parser.add_argument('--batch_size', type=int, default=1)
#     parser.add_argument('--gpu_id', type=int, default=0)
#     parser.add_argument('--im0', type=str, default='./inputs/images/0.png', help='First image')
#     parser.add_argument('--im1', type=str, default='./inputs/images/2.png', help='Second image')
#     parser.add_argument('--output_dir', type=str)
#     parser.add_argument('--onnx_path', type=str, default='./onnx/interpolation_.onnx', help='Path to save ONNX model')
#     parser.add_argument('--scale_factor', type=int, default=2, choices=[2, 4, 8, 16], help='Interpolation scale factor (2x, 4x, 8x, 16x)')
#     args = parser.parse_args()

#     cfg_file = args.config
#     dev_id = args.gpu_id
#     torch.cuda.set_device(dev_id)

#     cfg = make_config(cfg_file, launch_experiment=False)
#     print(model_profile(cfg.model))

#     model = make_model(cfg.model)
#     model.cuda()
#     model.eval()

#     im0_np = preprocess(args.im0)
#     im1_np = preprocess(args.im1)

#     print("Start ONNX export...")
#     dummy_input = (torch.from_numpy(im0_np).cuda(), torch.from_numpy(im1_np).cuda())
#     torch.onnx.export(model,
#                       dummy_input,
#                       args.onnx_path,
#                       export_params=True,
#                       opset_version=18,
#                       do_constant_folding=True,
#                       input_names=['input0', 'input1'],
#                       output_names=['output'],
#                       dynamic_axes={'input0': {0: 'batch_size', 2: 'height', 3: 'width'},
#                                     'input1': {0: 'batch_size', 2: 'height', 3: 'width'},
#                                     'output': {0: 'batch_size', 2: 'height', 3: 'width'}})
#     print(f"Completed ONNX export >> {args.onnx_path}")
