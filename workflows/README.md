# ACE-Step Type Adapter 使用指南

## 概述

`ACE_STEP_TypeAdapter` 节点允许您将 ComfyUI 标准类型（MODEL、VAE、CLIP）转换为 ACE-Step 专用的 `ACE_STEP_MODEL` 类型。

## 使用场景

1. **现有工作流集成**：当您已使用 `CheckpointLoaderSimple` 加载模型时
2. **模型复用**：避免重复加载同一模型
3. **统一工作流**：将标准 ComfyUI 节点与 ACE-Step 生成节点结合使用

## 节点说明

### ACE_STEP_TypeAdapter

**输入**：
| 参数 | 类型 | 描述 |
|------|------|------|
| model | MODEL | DiT 模型（来自 CheckpointLoaderSimple） |
| vae | VAE | VAE 模型 |
| clip | CLIP | 文本编码器 |
| lm_model_path | 选择 | 语言模型路径（如 acestep-5Hz-lm-1.7B） |
| device | 选择 | 计算设备（auto/cuda/cpu/mps/xpu） |
| prefer_download_source | 选择 | LM 模型下载源（auto/huggingface/modelscope） |
| offload_to_cpu | 布尔 | 是否将模型卸载到 CPU |
| silence_latent_path | 字符串 | silence_latent.pt 文件路径（可选） |

**输出**：
- `ACE_STEP_MODEL` - 可与所有 ACE-Step 生成节点兼容的模型对象

## 工作流示例

### 示例 1：简单文本生成

```
CheckpointLoaderSimple → ACE_STEP_TypeAdapter → ACE_STEP_SimpleMode
```

这是最简单的用法，直接从标准加载器生成音乐。

### 示例 2：结合 CreateSample

```
CheckpointLoaderSimple → ACE_STEP_TypeAdapter → ACE_STEP_TextToMusic
                                                    ↑
                                          ACE_STEP_CreateSample
```

先通过 LLM 生成结构化信息，再生成音乐。

### 示例 3：配合 LoRA 使用

```
ACE_STEP_LoRALoader → [lora_info] ─┐
                                   ├→ ACE_STEP_TextToMusic
CheckpointLoaderSimple → ACE_STEP_TypeAdapter ─┘
```

使用 LoRA 进行风格微调。

## 工作流文件

- `type_adapter_simple.json` - 简单示例（AIO 格式，3 个节点）
- `type_adapter_example.json` - 完整示例（AIO 格式，带 CreateSample）
- `split_format_example.json` - 分片格式示例（Comfy-Org 分片格式）
- `type_adapter_with_lora.json` - LoRA 兼容性示例

## 模型格式

### AIO 格式（All-in-One）

单个 `.safetensors` 文件包含所有组件（DiT、VAE、Text Encoder）。

**加载方式**：
```json
CheckpointLoaderSimple
  ckpt_name: "acestep_v15_turbo.safetensors"
```

### 分片格式（Split Format）

来自 Comfy-Org/ace_step_1.5_ComfyUI_files，需要分别加载三个组件。

**组件**：
- `diffusion_models/acestep_v1.5_turbo.safetensors` - DiT 模型
- `vae/ace_1.5_vae.safetensors` - VAE 模型
- `text_encoders/qwen_1.7b_ace15.safetensors` - 文本编码器

**加载方式**：
```json
UNET Loader: diffusion_models/acestep_v1.5_turbo.safetensors
VAE Loader: vae/ace_1.5_vae.safetensors
CLIP Loader: text_encoders/qwen_1.7b_ace15.safetensors
```

**下载命令**：
```bash
huggingface-cli download Comfy-Org/ace_step_1.5_ComfyUI_files \
  split_files/diffusion_models/acestep_v1.5_turbo.safetensors \
  --local-dir ComfyUI/models/

huggingface-cli download Comfy-Org/ace_step_1.5_ComfyUI_files \
  split_files/vae/ace_1.5_vae.safetensors \
  --local-dir ComfyUI/models/

huggingface-cli download Comfy-Org/ace_step_1.5_ComfyUI_files \
  split_files/text_encoders/qwen_1.7b_ace15.safetensors \
  --local-dir ComfyUI/models/
```

## 注意事项

1. **模型兼容性**：输入的 MODEL/VAE/CLIP 必须是 ACE-Step 1.5 架构兼容的
2. **silence_latent**：如果未找到，会自动生成默认值（可能影响质量）
3. **语言模型**：需要单独下载 LM 模型（如 acestep-5Hz-lm-1.7B）
4. **设备一致性**：确保所有组件在同一设备上运行
5. **LoRA 兼容性**：TypeAdapter 完全支持 LoRA，可与 `ACE_STEP_LoRALoader` 配合使用

## LoRA 兼容性

**TypeAdapter 完全支持 LoRA！**

所有 ACE-Step 生成节点（TextToMusic、Cover、Repaint、SimpleMode）都有 `lora_info` 输入参数，可以接收 `ACE_STEP_LoRALoader` 的输出。

**使用方式**：
1. 添加 `ACE_STEP_LoRALoader` 节点
2. 将其 `lora_info` 输出连接到生成节点的 `lora_info` 输入
3. 调整 LoRA 强度（scale 参数）

**注意**：
- LoRA 与量化（quantization）不兼容，使用 LoRA 时需将 quantization 设为 "None"
- LoRA 可以在模型加载后动态应用和卸载
- scale=0 会禁用 LoRA 但不卸载，方便快速对比

## 与 ModelLoader 的区别

| 特性 | ModelLoader | TypeAdapter |
|------|-------------|-------------|
| 输入 | 模型路径 | MODEL/VAE/CLIP 对象 |
| 加载方式 | 直接加载 | 提取已加载的组件 |
| 使用场景 | 新工作流 | 集成现有工作流 |
| 模型复用 | 每次加载 | 可复用已加载模型 |

## 颜色编码说明

工作流使用深色蓝色主题：
- 深蓝 (#1e3a5f/#4a90e2) - 初始化/设置
- 中深蓝 (#2c5282/#5ca0d3) - 工作流
- 中蓝 (#3a5f7d/#6b9bd1) - 命令
- 灰蓝 (#3d5a80/#6a95c5) - 适配
- 浅中蓝 (#4a6fa5/#7ca6d4) - 输出

所有节点使用白色文字 (#fff) 以确保在深色背景上的可读性。
