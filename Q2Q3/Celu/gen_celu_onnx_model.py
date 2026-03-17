import onnx
from onnx import helper, numpy_helper
from onnx import TensorProto
import onnxruntime as ort
import numpy as np
import argparse
import os
from datetime import datetime


# ======================
# 构建 Celu ONNX 模型
# ======================
def build_celu_onnx_model(input_shape, alpha, save_path):
    """
    Celu (Continuously Differentiable Exponential Linear Unit) 激活函数
    公式: max(0, x) + min(0, alpha * (exp(x/alpha) - 1))

    Args:
        input_shape: 输入张量的 shape
        alpha: 缩放参数，alpha > 0，默认为 1.0
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

    # 创建 Celu 节点
    celu_node = helper.make_node(
        'Celu',
        inputs=['input'],
        outputs=['output'],
        alpha=alpha
    )

    # 创建图
    graph = helper.make_graph(
        [celu_node],
        'Celu_Model',
        [input_tensor],
        [output_tensor]
    )

    # 创建模型 (opset 18)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 8

    # 保存模型
    onnx.save(model, save_path)
    print(f"[INFO] 成功导出 Celu ONNX 单算子模型: {save_path}")

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
    parser = argparse.ArgumentParser(description="构建 Celu ONNX 单算子模型")
    parser.add_argument("--input_shape", type=int, nargs="+", default=[1, 2, 3],
                        help="输入 shape，例如: 1 2 3 表示 [1, 2, 3]")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Celu 的 alpha 参数（缩放因子，必须>0），默认为 1.0")
    parser.add_argument("--input_bin", help="输入 bin 文件路径（可选）")

    args = parser.parse_args()

    # ===== Step 1: 输入参数处理 =====
    input_shape = list(args.input_shape)
    alpha = args.alpha

    if alpha <= 0:
        raise ValueError(f"alpha 必须大于 0，当前值为 {alpha}")

    print(f"[INFO] input_shape = {input_shape}")
    print(f"[INFO] alpha = {alpha}")

    # ===== Step 2: 目录和路径 =====
    input_shape_str = "x".join(map(str, input_shape))
    dir_name = f"celu_onnx_input_{input_shape_str}_alpha_{alpha}"
    os.makedirs(dir_name, exist_ok=True)
    model_path = os.path.join(dir_name, f"onnx_input_{input_shape_str}_celu.onnx")

    # ===== Step 3: 构建模型 =====
    build_celu_onnx_model(input_shape, alpha, model_path)

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
