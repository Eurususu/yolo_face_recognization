#coding=utf-8
import torch
import onnx
from onnx2pytorch import ConvertModel
from ptflops import get_model_complexity_info

# 加载 ONNX 模型
model_path = '/home/jia/.insightface/models/buffalo_l/adaface_r50.onnx'
onnx_model = onnx.load(model_path)

# 转换为 PyTorch 模型
pytorch_model = ConvertModel(onnx_model)

# 输入张量的大小
input_size = (3, 112, 112)  # 根据你的模型输入形状调整

# 计算 FLOPs
flops, params = get_model_complexity_info(pytorch_model, input_size, as_strings=True, print_per_layer_stat=True)
print(f"FLOPs: {flops}")
print(f"Params: {params}")
