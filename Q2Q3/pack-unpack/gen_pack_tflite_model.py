import tensorflow as tf
import numpy as np
import argparse
import os
from datetime import datetime


# ======================
# 构建 Pack 模型
# ======================
def build_pack_model(input_shape, axis, values_count, save_path):
    """
    Pack 算子：将多个张量沿着指定轴打包成一个张量

    Args:
        input_shape: 单个输入张量的 shape
        axis: 打包的轴（0-based）
        values_count: 输入张量的数量
        save_path: TFLite 模型保存路径
    """
    class PackModel(tf.Module):
        def __init__(self, axis, values_count):
            super().__init__()
            self.axis = axis
            self.values_count = values_count

        # 创建多个输入的 signature
        def get_input_signature(self):
            input_signatures = []
            for i in range(self.values_count):
                input_signatures.append(
                    tf.TensorSpec(shape=input_shape, dtype=tf.float32, name=f"input_{i}")
                )
            return input_signatures

        @tf.function
        def __call__(self, *args):
            # 使用 tf.stack/Pack 来打包张量
            return tf.stack(args, axis=self.axis, name="Pack")

    model = PackModel(axis, values_count)

    # 创建输入签名
    input_signatures = model.get_input_signature()
    concrete_func = tf.function(
        lambda *args: tf.stack(args, axis=axis, name="Pack")
    ).get_concrete_function(*input_signatures)

    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    tflite_model = converter.convert()

    with open(save_path, "wb") as f:
        f.write(tflite_model)
    print(f"[INFO] 成功导出 Pack 单算子模型: {save_path}")


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


def run_inference(interpreter, input_data_list, input_details, output_details):
    # 设置所有输入
    for i, input_data in enumerate(input_data_list):
        interpreter.set_tensor(input_details[i]['index'], input_data)
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
    parser = argparse.ArgumentParser(description="构建 Pack TFLite 单算子模型")
    parser.add_argument("--input_shape", type=int, nargs="+", default=[2, 3],
                        help="单个输入张量的 shape，例如: 2 3 表示 [2, 3]")
    parser.add_argument("--axis", type=int, default=0,
                        help="打包的轴（0-based），默认为 0")
    parser.add_argument("--values_count", type=int, default=3,
                        help="输入张量的数量，默认为 3")

    args = parser.parse_args()

    # ===== Step 1: 输入参数处理 =====
    input_shape = tuple(args.input_shape)
    axis = args.axis
    values_count = args.values_count
    print(f"[INFO] input_shape = {input_shape}")
    print(f"[INFO] axis = {axis}")
    print(f"[INFO] values_count = {values_count}")

    # ===== Step 2: 目录和路径 =====
    input_shape_str = "x".join(map(str, input_shape))
    dir_name = f"pack_input_{input_shape_str}_axis_{axis}_count_{values_count}"
    os.makedirs(dir_name, exist_ok=True)
    model_path = os.path.join(dir_name, f"tf_pack_axis_{axis}_count_{values_count}.tflite")

    # ===== Step 3: 构建模型 =====
    build_pack_model(input_shape, axis, values_count, model_path)

    # ===== Step 4: 加载模型 =====
    interpreter = load_tflite_model(model_path)
    input_details, output_details = get_model_io_details(interpreter)

    # ===== Step 5: 输入准备 =====
    dtype = input_details[0]['dtype']

    # 为每个输入生成随机数据
    input_data_list = []
    for i in range(len(input_details)):
        input_data = np.random.rand(*input_shape).astype(dtype)
        input_data_list.append(input_data)
        # 保存每个输入文件
        np.array(input_data).tofile(os.path.join(dir_name, f"input_{i}.bin"))
        print(f"[INFO] 生成随机输入 input_{i} shape={input_data.shape}")

    # ===== Step 6: 推理 =====
    outputs = run_inference(interpreter, input_data_list, input_details, output_details)

    # ===== Step 7: 保存输出 =====
    save_output_bin(outputs, output_details, os.path.join(dir_name, "golden_output.bin"))


if __name__ == "__main__":
    main()
