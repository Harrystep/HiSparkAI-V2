import tensorflow as tf
import numpy as np
import argparse
import os
from datetime import datetime


# ======================
# 构建 Gelu 模型
# ======================
def build_gelu_model(input_shape, approximate, save_path):
    """
    Gelu (Gaussian Error Linear Unit) 激活函数
    公式: GELU(x) = x * Φ(x), 其中 Φ(x) 是标准正态分布的累积分布函数

    Args:
        input_shape: 输入张量的 shape
        approximate: 近似模式
            - "none" 或 False: 使用精确公式 erf(x / sqrt(2))
            - "tanh" 或 True: 使用 tanh 近似公式
        save_path: TFLite 模型保存路径
    """
    class GeluModel(tf.Module):
        def __init__(self, approximate):
            super().__init__()
            # TensorFlow 的 approximate 参数: False=精确, True=tanh近似
            self.approximate = (approximate == "tanh")

        @tf.function(input_signature=[tf.TensorSpec(shape=input_shape, dtype=tf.float32, name="input")])
        def __call__(self, x):
            # 使用 tf.nn.gelu 来生成 Gelu 算子
            # approximate: False=精确公式(erf), True=tanh近似
            return tf.nn.gelu(x, approximate=self.approximate, name="Gelu")

    model = GeluModel(approximate)
    concrete_func = model.__call__.get_concrete_function()
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    tflite_model = converter.convert()

    with open(save_path, "wb") as f:
        f.write(tflite_model)
    print(f"[INFO] 成功导出 Gelu 单算子模型: {save_path}")


# ======================
# 通用工具函数
# ======================
def load_tflite_model(model_path):
    print(f"[INFO] 加载 TFLite 模型: {model_path}")
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter


def get_model_io_details(interpreter):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("\n===== 模型输入信息 =====")
    for d in input_details:
        print(d)
    print("\n===== 模型输出信息 =====")
    for d in output_details:
        print(d)
    return input_details, output_details


def run_inference(interpreter, input_data, input_details, output_details):
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    outputs = [interpreter.get_tensor(o['index']) for o in output_details]
    for i, out in enumerate(outputs):
        print(f"[OUTPUT] {output_details[i]['name']} shape={out.shape}, dtype={out.dtype}")
    return outputs


def save_output_bin(outputs, output_details, output_path):
    outputs[0].tofile(output_path)
    print(f"[INFO] 输出保存为 {output_path}")


# ======================
# 主函数入口
# ======================
def main():
    parser = argparse.ArgumentParser(description="构建 Gelu TFLite 单算子模型")
    parser.add_argument("--input_shape", type=int, nargs="+", default=[1, 2, 3],
                        help="输入 shape，例如: 1 2 3 表示 [1, 2, 3]")
    parser.add_argument("--approximate", type=str, choices=["none", "tanh"], default="none",
                        help="Gelu 近似模式: none=精确公式(erf), tanh=tanh 近似公式，默认为 none")
    parser.add_argument("--input_bin", help="输入 bin 文件路径（可选）")

    args = parser.parse_args()

    # ===== Step 1: 输入参数处理 =====
    input_shape = tuple(args.input_shape)
    approximate = args.approximate
    print(f"[INFO] input_shape = {input_shape}")
    print(f"[INFO] approximate = {approximate}")

    # ===== Step 2: 目录和路径 =====
    input_shape_str = "x".join(map(str, input_shape))
    dir_name = f"gelu_input_{input_shape_str}_approx_{approximate}"
    os.makedirs(dir_name, exist_ok=True)
    model_path = os.path.join(dir_name, f"tf_input_{input_shape_str}_approx_{approximate}_gelu.tflite")

    # ===== Step 3: 构建模型 =====
    build_gelu_model(input_shape, approximate, model_path)

    # ===== Step 4: 加载模型 =====
    interpreter = load_tflite_model(model_path)
    input_details, output_details = get_model_io_details(interpreter)

    # ===== Step 5: 输入准备 =====
    shape = input_details[0]['shape']
    dtype = input_details[0]['dtype']

    if args.input_bin and os.path.exists(args.input_bin):
        print(f"[INFO] 从 {args.input_bin} 加载输入")
        input_data = np.fromfile(args.input_bin, dtype=dtype).reshape(shape)
    else:
        input_data = np.random.rand(*shape).astype(dtype)
        np.array(input_data).tofile(os.path.join(dir_name, "input.bin"))
        print(f"[INFO] 生成随机输入 shape={input_data.shape}")

    # ===== Step 6: 推理 =====
    outputs = run_inference(interpreter, input_data, input_details, output_details)

    # ===== Step 7: 保存输出 =====
    save_output_bin(outputs, output_details, os.path.join(dir_name, "golden_output.bin"))


if __name__ == "__main__":
    main()
