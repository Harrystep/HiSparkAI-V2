import onnx
from onnx import helper, numpy_helper
from onnx import TensorProto
import onnxruntime as ort
import numpy as np
import argparse
import os
from datetime import datetime


# ======================
# 构建 Gelu ONNX 模型
# ======================
def build_gelu_onnx_model(input_shape, approximate, save_path):
    """
    Gelu (Gaussian Error Linear Unit) 激活函数
    公式: GELU(x) = x * Φ(x), 其中 Φ(x) 是标准正态分布的累积分布函数
    近似公式: GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))

    Args:
        input_shape: 输入张量的 shape
        approximate: 近似模式，"none" 表示使用精确公式(erf)，"tanh" 表示使用 tanh 近似
        save_path: ONNX 模型保存路径
    """
    # 创建输入
    input_tensor = helper.make_tensor_value_info(
        'input', TensorProto.FLOAT, input_shape
    )

    # 创建输出
    output_tensor = helper.make_tensor_value_info(
        'output', TensorProto.FLOAT, input_shape
    )

    # 创建 Gelu 节点
    # ONNX Gelu 算子: https://github.com/onnx/onnx/blob/main/docs/Operators.md#Gelu
    # approximate: "none" = 精确公式(erf), "tanh" = tanh 近似公式
    gelu_node = helper.make_node(
        'Gelu',
        inputs=['input'],
        outputs=['output'],
        approximate=approximate
    )

    # 创建图
    graph = helper.make_graph(
        [gelu_node],
        'Gelu_Model',
        [input_tensor],
        [output_tensor]
    )

    # 创建模型 (使用 opset 20 以支持 Gelu)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 20)])
    model.ir_version = 8

    # 保存模型
    onnx.save(model, save_path)
    print(f"[INFO] 成功导出 Gelu ONNX 单算子模型: {save_path}")

    return model


# ======================
# 通用工具函数
# ======================
def load_onnx_model(model_path):
    print(f"[INFO] 加载 ONNX 模型: {model_path}")
    session = ort.InferenceSession(model_path)
    return session


def get_model_io_details(session):
    input_details = session.get_inputs()
    output_details = session.get_outputs()
    print("\n===== 模型输入信息 =====")
    for d in input_details:
        print(f"  name: {d.name}, shape: {d.shape}, dtype: {d.type}")
    print("\n===== 模型输出信息 =====")
    for d in output_details:
        print(f"  name: {d.name}, shape: {d.shape}, dtype: {d.type}")
    return input_details, output_details


def run_inference(session, input_data, input_details, output_details):
    input_name = input_details[0].name
    outputs = session.run(None, {input_name: input_data})
    for i, out in enumerate(outputs):
        print(f"[OUTPUT] {output_details[i].name} shape={out.shape}, dtype={out.dtype}")
    return outputs


def save_output_bin(outputs, output_details, output_path):
    outputs[0].tofile(output_path)
    print(f"[INFO] 输出保存为 {output_path}")


# ======================
# 主函数入口
# ======================
def main():
    parser = argparse.ArgumentParser(description="构建 Gelu ONNX 单算子模型")
    parser.add_argument("--input_shape", type=int, nargs="+", default=[1, 2, 3],
                        help="输入 shape，例如: 1 2 3 表示 [1, 2, 3]")
    parser.add_argument("--approximate", type=str, choices=["none", "tanh"], default="none",
                        help="Gelu 近似模式: none=精确公式(erf), tanh=tanh 近似公式，默认为 none")
    parser.add_argument("--input_bin", help="输入 bin 文件路径（可选）")

    args = parser.parse_args()

    # ===== Step 1: 输入参数处理 =====
    input_shape = list(args.input_shape)
    approximate = args.approximate
    print(f"[INFO] input_shape = {input_shape}")
    print(f"[INFO] approximate = {approximate}")

    # ===== Step 2: 目录和路径 =====
    input_shape_str = "x".join(map(str, input_shape))
    dir_name = f"gelu_onnx_input_{input_shape_str}_approx_{approximate}"
    os.makedirs(dir_name, exist_ok=True)
    model_path = os.path.join(dir_name, f"onnx_input_{input_shape_str}_gelu_approx_{approximate}.onnx")

    # ===== Step 3: 构建模型 =====
    build_gelu_onnx_model(input_shape, approximate, model_path)

    # ===== Step 4: 加载模型 =====
    session = load_onnx_model(model_path)
    input_details, output_details = get_model_io_details(session)

    # ===== Step 5: 输入准备 =====
    shape = list(input_details[0].shape)
    dtype = np.float32

    if args.input_bin and os.path.exists(args.input_bin):
        print(f"[INFO] 从 {args.input_bin} 加载输入")
        input_data = np.fromfile(args.input_bin, dtype=dtype).reshape(input_shape)
    else:
        input_data = np.random.rand(*input_shape).astype(dtype)
        np.array(input_data).tofile(os.path.join(dir_name, "input.bin"))
        print(f"[INFO] 生成随机输入 shape={input_data.shape}")

    # ===== Step 6: 推理 =====
    outputs = run_inference(session, input_data, input_details, output_details)

    # ===== Step 7: 保存输出 =====
    save_output_bin(outputs, output_details, os.path.join(dir_name, "golden_output.bin"))


if __name__ == "__main__":
    main()
