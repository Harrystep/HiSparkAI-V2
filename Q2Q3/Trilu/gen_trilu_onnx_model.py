import onnx
from onnx import helper, numpy_helper
from onnx import TensorProto
import onnxruntime as ort
import numpy as np
import argparse
import os
from datetime import datetime


# ======================
# 构建 Trilu ONNX 模型
# ======================
def build_trilu_onnx_model(input_shape, upper, k, save_path, k_as_input=True):
    """
    Trilu (Truncate matrix upper or lower triangular) 函数
    保留矩阵的上三角或下三角部分

    Args:
        input_shape: 输入张量的 shape（至少是 2D）
        upper: True=保留上三角，False=保留下三角
        k: 对角线偏移量，默认为 0
        save_path: ONNX 模型保存路径
        k_as_input: True=k作为外部输入(动态), False=k作为图内常量(静态)
    """
    # 创建输入（矩阵）
    input_tensor = helper.make_tensor_value_info(
        'input', TensorProto.FLOAT, input_shape
    )

    # 创建输出
    output_tensor = helper.make_tensor_value_info(
        'output', TensorProto.FLOAT, input_shape
    )

    # 根据 k_as_input 决定 k 的处理方式
    if k_as_input:
        # k 作为外部输入（动态）
        k_tensor = helper.make_tensor_value_info(
            'k', TensorProto.INT64, []  # 标量
        )
        # 创建 Trilu 节点，k 作为输入
        trilu_node = helper.make_node(
            'Trilu',
            inputs=['input', 'k'],
            outputs=['output'],
            upper=upper
        )
        # 图的输入包含 k
        graph_inputs = [input_tensor, k_tensor]
        k_mode_str = "k_dynamic"
        nodes = [trilu_node]
    else:
        # k 作为图内常量（静态），使用 Constant 节点
        k_const_tensor = helper.make_tensor(
            'k_const',
            TensorProto.INT64,
            [],
            [k]
        )
        k_const_node = helper.make_node(
            'Constant',
            inputs=[],
            outputs=['k'],
            value=k_const_tensor
        )
        # 创建 Trilu 节点，k 来自 Constant 节点
        trilu_node = helper.make_node(
            'Trilu',
            inputs=['input', 'k'],
            outputs=['output'],
            upper=upper
        )
        # 图的输入不包含 k
        graph_inputs = [input_tensor]
        k_mode_str = f"k_const_{k}"
        nodes = [k_const_node, trilu_node]

    # 创建图
    graph = helper.make_graph(
        nodes,
        f'Trilu_Model_{k_mode_str}',
        graph_inputs,
        [output_tensor]
    )

    # 创建模型 (opset 18)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 8

    # 保存模型
    onnx.save(model, save_path)
    k_mode_desc = "k作为外部输入(动态)" if k_as_input else f"k作为图内常量(静态={k})"
    print(f"[INFO] 成功导出 Trilu ONNX 单算子模型 ({k_mode_desc}): {save_path}")

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


def run_inference(session, input_data, k_value, input_details, output_details, k_as_input=True):
    """
    推理函数

    Args:
        session: ONNX Runtime 会话
        input_data: 输入数据
        k_value: k 的值
        input_details: 输入详情
        output_details: 输出详情
        k_as_input: True=k作为输入传入, False=k已在模型中作为常量
    """
    if k_as_input:
        # k 作为输入传入
        outputs = session.run(None, {
            input_details[0].name: input_data,
            input_details[1].name: np.array(k_value, dtype=np.int64)
        })
    else:
        # k 已作为常量属性，只需传入 input
        outputs = session.run(None, {
            input_details[0].name: input_data
        })

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
    parser = argparse.ArgumentParser(description="构建 Trilu ONNX 单算子模型")
    parser.add_argument("--input_shape", type=int, nargs="+", default=[3, 4],
                        help="输入 shape（至少 2D），例如: 3 4 表示 [3, 4]")
    parser.add_argument("--upper", type=int, choices=[0, 1], default=1,
                        help="1=保留上三角，0=保留下三角，默认为 1")
    parser.add_argument("--k", type=int, default=0,
                        help="对角线偏移量，默认为 0")
    parser.add_argument("--k_as_input", type=int, choices=[0, 1], default=1,
                        help="1=k作为输入变量(动态), 0=k作为常量属性(静态)，默认为 1")
    parser.add_argument("--input_bin", help="输入 bin 文件路径（可选）")

    args = parser.parse_args()

    # ===== Step 1: 输入参数处理 =====
    input_shape = list(args.input_shape)
    upper = bool(args.upper)
    k = args.k
    k_as_input = bool(args.k_as_input)

    if len(input_shape) < 2:
        raise ValueError(f"input_shape 必须至少是 2D，当前为 {len(input_shape)}D")

    print(f"[INFO] input_shape = {input_shape}")
    print(f"[INFO] upper = {upper}")
    print(f"[INFO] k = {k}")
    print(f"[INFO] k_as_input = {k_as_input} ({'k作为输入变量(动态)' if k_as_input else 'k作为常量属性(静态)'})")

    # ===== Step 2: 目录和路径 =====
    input_shape_str = "x".join(map(str, input_shape))
    mode_str = "upper" if upper else "lower"
    k_mode_str = "k_dynamic" if k_as_input else f"k_const_{k}"
    dir_name = f"trilu_onnx_input_{input_shape_str}_{mode_str}_{k_mode_str}"
    os.makedirs(dir_name, exist_ok=True)
    model_path = os.path.join(dir_name, f"onnx_input_{input_shape_str}_trilu.onnx")

    # ===== Step 3: 构建模型 =====
    build_trilu_onnx_model(input_shape, upper, k, model_path, k_as_input)

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
    outputs = run_inference(session, input_data, k, input_details, output_details, k_as_input)

    # ===== Step 7: 保存输出 =====
    save_output_bin(outputs, output_details, os.path.join(dir_name, "golden_output.bin"))


if __name__ == "__main__":
    main()
