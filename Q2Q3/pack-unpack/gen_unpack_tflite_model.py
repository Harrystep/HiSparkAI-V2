import tensorflow as tf
import numpy as np
import argparse
import os
from datetime import datetime


# ======================
# 构建 Unpack 模型
# ======================
def build_unpack_model(input_shape, axis, num, save_path):
    """
    Unpack 算子：将一个张量沿着指定轴解包成多个张量

    Args:
        input_shape: 输入张量的 shape
        axis: 解包的轴（0-based）
        num: 输出张量的数量
        save_path: TFLite 模型保存路径
    """
    class UnpackModel(tf.Module):
        def __init__(self, axis, num):
            super().__init__()
            self.axis = axis
            self.num = num

        @tf.function(input_signature=[tf.TensorSpec(shape=input_shape, dtype=tf.float32, name="input")])
        def __call__(self, x):
            # 使用 tf.unstack/Unpack 来解包张量
            return tf.unstack(x, axis=self.axis, num=self.num, name="Unpack")

    model = UnpackModel(axis, num)
    concrete_func = model.__call__.get_concrete_function()
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    tflite_model = converter.convert()

    with open(save_path, "wb") as f:
        f.write(tflite_model)
    print(f"[INFO] 成功导出 Unpack 单算子模型: {save_path}")


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


def save_output_bin(outputs, output_details, dir_path):
    # 保存每个输出
    for i, out in enumerate(outputs):
        output_path = os.path.join(dir_path, f"golden_output_{i}.bin")
        out.tofile(output_path)
        print(f"[INFO] 输出 {i} 保存为 {output_path}")


# ======================
# 主函数入口
# ======================
def main():
    parser = argparse.ArgumentParser(description="构建 Unpack TFLite 单算子模型")
    parser.add_argument("--input_shape", type=int, nargs="+", default=[3, 2, 3],
                        help="输入张量的 shape，例如: 3 2 3 表示 [3, 2, 3]")
    parser.add_argument("--axis", type=int, default=0,
                        help="解包的轴（0-based），默认为 0")
    parser.add_argument("--num", type=int, default=3,
                        help="输出张量的数量，默认为 3")

    args = parser.parse_args()

    # ===== Step 1: 输入参数处理 =====
    input_shape = tuple(args.input_shape)
    axis = args.axis
    num = args.num

    # 验证 axis 维度大小是否等于 num
    if input_shape[axis] != num:
        raise ValueError(f"input_shape[{axis}]={input_shape[axis]} 必须等于 num={num}")

    print(f"[INFO] input_shape = {input_shape}")
    print(f"[INFO] axis = {axis}")
    print(f"[INFO] num = {num}")

    # ===== Step 2: 目录和路径 =====
    input_shape_str = "x".join(map(str, input_shape))
    dir_name = f"unpack_input_{input_shape_str}_axis_{axis}_num_{num}"
    os.makedirs(dir_name, exist_ok=True)
    model_path = os.path.join(dir_name, f"tf_unpack_axis_{axis}_num_{num}.tflite")

    # ===== Step 3: 构建模型 =====
    build_unpack_model(input_shape, axis, num, model_path)

    # ===== Step 4: 加载模型 =====
    interpreter = load_tflite_model(model_path)
    input_details, output_details = get_model_io_details(interpreter)

    # ===== Step 5: 输入准备 =====
    shape = input_details[0]['shape']
    dtype = input_details[0]['dtype']

    input_data = np.random.rand(*shape).astype(dtype)
    np.array(input_data).tofile(os.path.join(dir_name, "input.bin"))
    print(f"[INFO] 生成随机输入 shape={input_data.shape}")

    # ===== Step 6: 推理 =====
    outputs = run_inference(interpreter, input_data, input_details, output_details)

    # ===== Step 7: 保存输出 =====
    save_output_bin(outputs, output_details, dir_name)


if __name__ == "__main__":
    main()
