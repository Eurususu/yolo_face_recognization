#coding=utf-8
import onnxruntime as ort

# 打印所有可用的执行提供者
print("Available providers:", ort.get_all_providers())

# 创建一个会话，并指定使用 CUDA Execution Provider
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
sess_options = ort.SessionOptions()
session = ort.InferenceSession("/home/jia/.insightface/models/buffalo_l/w600k_r50.onnx", sess_options, providers=providers)

# 检查当前会话使用的提供者
print("Session providers:", session.get_providers())
