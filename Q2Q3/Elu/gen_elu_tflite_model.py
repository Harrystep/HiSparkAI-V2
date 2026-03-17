import tensorflow as tf
import numpy as np
import argparse
import os
from datetime import datetime


# ======================
# 构建 Elu 模型
# ======================
def build_elu_model(input_shape, alpha, save_path):
    """
    Elu (Exponential Linear Unit) 激活函数
    公式: f(x) = x when x > 0, else alpha * (exp(x) - 1)

    Args:
        input_shape: 输入张量的 shape
        alpha: 负区间的缩放因子，默认为 1.0
        save_path: TFLite 模型保存路径
    """
    class EluModel(tf.Module):
        def __init__(self, alpha):
            super().__init__()
            self.alpha = tf.constant(alpha, dtype=tf.float32)

        @tf.function(input_signature=[tf.TensorSpec(shape=input_shape, dtype=tf.float32, name="input")])
        def __call__(self, x):
            # 使用 tf.raw_ops.Elu 来生成 Elu 算子
            return tf.raw_ops.Elu(features=x, name="Elu")

    model = EluModel(alpha)
    concrete_func = model.__call__.get_concrete_function()
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    tflite_model = converter.convert()

    with open(save_path, "wb") as f:
        f.write(tflite_model)
    print(f"[INFO] 成功导出 Elu 单算子模型: {save_path}")


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
    parser = argparse.ArgumentParser(description="构建 Elu TFLite 单算子模型")
    parser.add_argument("--input_shape", type=int, nargs="+", default=[1, 2, 3],
                        help="输入 shape，例如: 1 2 3 表示 [1, 2, 3]")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Elu 激活函数的 alpha 参数（负区间缩放因子），默认为 1.0")
    parser.add_argument("--input_bin", help="输入 bin 文件路径（可选）")

    args = parser.parse_args()

    # ===== Step 1: 输入参数处理 =====
    input_shape = tuple(args.input_shape)
    alpha = args.alpha
    print(f"[INFO] input_shape = {input_shape}")
    print(f"[INFO] alpha = {alpha}")

    # ===== Step 2: 目录和路径 =====
    current_time = datetime.now().strftime("%Y%m%d")
    input_shape_str = "x".join(map(str, input_shape))
    dir_name = f"{current_time}_elu_input_{input_shape_str}_alpha_{alpha}"
    os.makedirs(dir_name, exist_ok=True)
    model_path = os.path.join(dir_name, "elu.tflite")

    # ===== Step 3: 构建模型 =====
    build_elu_model(input_shape, alpha, model_path)

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
