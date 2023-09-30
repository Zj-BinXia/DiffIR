import argparse
import torch
import torch.onnx
from DiffIR.archs.S2_arch import DiffIRS2


def main(args):
    # An instance of the model
    model = DiffIRS2( n_encoder_res= 9, dim= 64, scale=args.scale,num_blocks= [13,1,1,1],num_refinement_blocks= 13,heads= [1,2,4,8], ffn_expansion_factor= 2.2,LayerNorm_type= "BiasFree")
    loadnet = torch.load(args.model_path, map_location=torch.device('cpu'))
    model.load_state_dict(loadnet['params_ema'], strict=True)
    # set the train mode to false since we will only run the forward pass.
    model.train(False)
    model.cpu().eval()

    # An example input
    x = torch.rand(1, 3, 64, 64)
    # Export the model
    with torch.no_grad():
        torch_out = torch.onnx._export(model, x, args.output, opset_version=11, export_params=True)
    print(torch_out.shape)


if __name__ == '__main__':
    """Convert pytorch model to onnx models"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--scale', type=int, default=4)
    parser.add_argument('--model_path', type=str, default='./experiments/DiffIRS2-GANv2.pth')
    parser.add_argument('--output', type=str, default='DiffIRS2-GANv2-x4.onnx', help='Output onnx path')
    args = parser.parse_args()

    main(args)
