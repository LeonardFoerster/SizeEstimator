import sys
import os
import torch

# Add current directory to path so we can import depth_anything_v2
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'Depth-Anything-V2'))

from depth_anything_v2.dpt import DepthAnythingV2

def export_onnx():
    encoder = 'vitl'
    checkpoint_path = 'Depth-Anything-V2/checkpoints/depth_anything_v2_vitl.pth'
    output_path = 'depth_anything_v2_vitl.onnx'
    
    print(f"Loading model {encoder} from {checkpoint_path}...")
    
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    }
    
    model = DepthAnythingV2(**model_configs[encoder])
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    
    # Input size 518x518 is the default used in infer_image
    dummy_input = torch.randn(1, 3, 518, 518)
    
    print(f"Exporting to {output_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size', 2: 'height', 3: 'width'},
            'output': {0: 'batch_size', 1: 'height', 2: 'width'}
        }
    )
    print("Export complete.")

if __name__ == "__main__":
    export_onnx()
