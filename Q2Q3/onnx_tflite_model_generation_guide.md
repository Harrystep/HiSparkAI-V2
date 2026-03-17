# ONNX/TFLite 算子模型导出工具使用说明

本文档描述了如何使用提供的 Python 脚本导出各种算子的 ONNX 和 TFLite 单算子模型。

---

## 目录

- [ONNX 算子导出脚本](#onnx-算子导出脚本)
  - [激活函数算子](#激活函数算子)
  - [矩阵操作算子](#矩阵操作算子)
- [TFLite 算子导出脚本](#tflite-算子导出脚本)
- [通用参数说明](#通用参数说明)

---

## ONNX 算子导出脚本

所有 ONNX 脚本使用 **opset 18** 版本，生成的模型可用于 ONNX Runtime 推理。

### 激活函数算子

#### 1. HardSigmoid

**脚本**: `gen_hardsigmoid_onnx_model.py`

**公式**: `max(0, min(1, alpha * x + beta))`

```bash
python gen_hardsigmoid_onnx_model.py \
    --input_shape 1 4 5 6 \
    --alpha 0.2 \
    --beta 0.5
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--input_shape` | 输入张量 shape | `[1, 2, 3]` |
| `--alpha` | 斜率参数 | `0.2` |
| `--beta` | 截距参数 | `0.5` |

---

#### 2. Celu (Continuously Differentiable ELU)

**脚本**: `gen_celu_onnx_model.py`

**公式**: `max(0, x) + min(0, alpha * (exp(x/alpha) - 1))`

```bash
python gen_celu_onnx_model.py \
    --input_shape 1 4 5 6 \
    --alpha 1.0
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--input_shape` | 输入张量 shape | `[1, 2, 3]` |
| `--alpha` | 缩放参数（必须 > 0） | `1.0` |

---

#### 3. Erf (Gauss Error Function)

**脚本**: `gen_erf_onnx_model.py`

**公式**: 高斯误差函数

```bash
python gen_erf_onnx_model.py \
    --input_shape 1 4 5 6
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--input_shape` | 输入张量 shape | `[1, 2, 3]` |

---

#### 4. Elu (Exponential Linear Unit)

**脚本**: `gen_elu_onnx_model.py`

**公式**: `f(x) = x when x > 0, else alpha * (exp(x) - 1)`

```bash
python gen_elu_onnx_model.py \
    --input_shape 1 4 5 6 \
    --alpha 1.0
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--input_shape` | 输入张量 shape | `[1, 2, 3]` |
| `--alpha` | 负区间缩放因子 | `1.0` |

---

#### 5. Gelu (Gaussian Error Linear Unit)

**脚本**: `gen_gelu_onnx_model.py`

**公式**: `GELU(x) = x * Φ(x)`，其中 Φ(x) 是标准正态分布的 CDF

```bash
# 精确模式 (erf)
python gen_gelu_onnx_model.py \
    --input_shape 1 4 5 6 \
    --approximate none

# tanh 近似模式
python gen_gelu_onnx_model.py \
    --input_shape 1 4 5 6 \
    --approximate tanh
```

| 参数 | 说明 | 默认值 | 可选值 |
|------|------|--------|--------|
| `--input_shape` | 输入张量 shape | `[1, 2, 3]` | - |
| `--approximate` | 近似模式 | `none` | `none`, `tanh` |

---

### 矩阵操作算子

#### 6. Trilu (Truncate matrix upper/lower triangular)

**脚本**: `gen_trilu_onnx_model.py`

**功能**: 保留矩阵的上三角或下三角部分

```bash
# 上三角，k 作为动态输入
python gen_trilu_onnx_model.py \
    --input_shape 3 4 \
    --upper 1 \
    --k 0 \
    --k_as_input 1

# 下三角，k 作为静态常量 (k=1)
python gen_trilu_onnx_model.py \
    --input_shape 3 4 \
    --upper 0 \
    --k 1 \
    --k_as_input 0
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--input_shape` | 输入张量 shape（至少 2D） | `[3, 4]` |
| `--upper` | `1`=上三角, `0`=下三角 | `1` |
| `--k` | 对角线偏移量 | `0` |
| `--k_as_input` | `1`=k作为输入(动态), `0`=k作为常量(静态) | `1` |

**k 参数说明**:
- `k = 0`: 保留主对角线
- `k > 0`: 保留主对角线上方第 k 条对角线
- `k < 0`: 保留主对角线下方第 |k| 条对角线

---

## TFLite 算子导出脚本

### 激活函数算子

#### 1. Elu

**脚本**: `gen_elu_tflite_model.py`

```bash
python gen_elu_tflite_model.py \
    --input_shape 1 4 5 6 \
    --alpha 1.0
```

---

#### 2. Gelu

**脚本**: `gen_gelu_tflite_model.py`

```bash
# 精确模式
python gen_gelu_tflite_model.py \
    --input_shape 1 4 5 6 \
    --approximate none

# tanh 近似模式
python gen_gelu_tflite_model.py \
    --input_shape 1 4 5 6 \
    --approximate tanh
```

---

### 张量操作算子

#### 3. Pack

**脚本**: `gen_pack_tflite_model.py`

**功能**: 将多个张量沿指定轴打包成一个张量

```bash
python gen_pack_tflite_model.py \
    --input_shape 2 3 \
    --axis 0 \
    --values_count 3
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--input_shape` | 单个输入张量的 shape | `[2, 3]` |
| `--axis` | 打包的轴（0-based） | `0` |
| `--values_count` | 输入张量数量 | `3` |

**输出**: 3 个 `[2, 3]` 的张量沿 axis 0 打包成 `[3, 2, 3]`

---

#### 4. Unpack

**脚本**: `gen_unpack_tflite_model.py`

**功能**: 将一个张量沿指定轴解包成多个张量

```bash
python gen_unpack_tflite_model.py \
    --input_shape 3 2 3 \
    --axis 0 \
    --num 3
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--input_shape` | 输入张量的 shape | `[3, 2, 3]` |
| `--axis` | 解包的轴（0-based） | `0` |
| `--num` | 输出张量数量 | `3` |

**注意**: `input_shape[axis]` 必须等于 `num`

---

## 通用参数说明

所有脚本支持以下通用参数：

| 参数 | 说明 |
|------|------|
| `--input_bin` | 输入 bin 文件路径（可选）。如果指定，将从文件加载输入数据；否则生成随机输入 |

---

## 输出文件结构

每个脚本运行后会生成一个目录，包含以下文件：

```
<算子名>_input_<shape>_<属性>/
├── <算子名>.onnx 或 <算子名>.tflite    # 导出的模型文件
├── input.bin                           # 输入数据（随机生成或指定）
└── golden_output.bin                   # 推理输出（金标准）
```

**示例**:
```
hardsigmoid_onnx_input_1x4x5x6_alpha_0.2_beta_0.5/
├── onnx_input_1x4x5x6_hardsigmoid.onnx
├── input.bin
└── golden_output.bin
```

---

## 依赖环境

### ONNX 脚本
```bash
pip install onnx onnxruntime numpy
```

### TFLite 脚本
```bash
pip install tensorflow numpy
```

---

## 示例：完整工作流

```bash
# 1. 生成 HardSigmoid ONNX 模型
python gen_hardsigmoid_onnx_model.py \
    --input_shape 1 4 5 6 \
    --alpha 0.2 \
    --beta 0.5

# 2. 生成的文件位置
# hardsigmoid_onnx_input_1x4x5x6_alpha_0.2_beta_0.5/

# 3. 使用 ONNX Runtime 进行推理验证
python -c "
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession(
    'hardsigmoid_onnx_input_1x4x5x6_alpha_0.2_beta_0.5/onnx_input_1x4x5x6_hardsigmoid.onnx'
)
input_data = np.fromfile(
    'hardsigmoid_onnx_input_1x4x5x6_alpha_0.2_beta_0.5/input.bin',
    dtype=np.float32
).reshape(1, 4, 5, 6)

outputs = session.run(None, {'input': input_data})
print(f'Output shape: {outputs[0].shape}')
"
```

---

## 算子速查表

| 算子 | 框架 | 可配置属性 |
|------|------|-----------|
| HardSigmoid | ONNX | alpha, beta |
| Celu | ONNX | alpha |
| Erf | ONNX | - |
| Elu | ONNX | alpha |
| Gelu | ONNX | approximate |
| Trilu | ONNX | upper, k, k_as_input |
| Elu | TFLite | alpha |
| Gelu | TFLite | approximate |
| Pack | TFLite | axis, values_count |
| Unpack | TFLite | axis, num |

---

*文档生成日期: 2025-03-17*
