# ACE-Step Model Setup Guide

## Model Directory Structure

The `checkpoint_dir` parameter in the nodes **scans all subdirectories** directly under the root `ComfyUI/models/` folder.

This allows you to choose any folder in the models directory as your ACE-Step base.

### Example Layout

```
ComfyUI/models/
└── My-ACE-Step-Models/         # This name will appear in the dropdown
    ├── acestep-v15-turbo/      # DiT Model
    │   ├── config.json
    │   └── model.safetensors
    ├── acestep-5Hz-lm-1.7B/    # LM Model
    ├── vae/                    # VAE Model
    └── Qwen3-Embedding-0.6B/   # Text Encoder
```

## 安装步骤

### 1. 下载模型

从 HuggingFace 下载 ACE-Step 模型：

```bash
# 使用 huggingface-cli
huggingface-cli download ACE-Step/Ace-Step1.5 --local-dir ComfyUI/models/Ace-Step1.5
```

或手动下载：
- https://huggingface.co/ACE-Step/Ace-Step1.5

### 2. 使用符号链接（如果你已有模型）

如果你已经下载了模型到其他位置，可以创建符号链接：

```bash
# Linux/macOS
ln -s /path/to/your/Ace-Step1.5/* ComfyUI/models/Ace-Step1.5/
```

### 3. 验证安装

在 ComfyUI 中加载 ACE-Step 节点，保持 `checkpoint_dir` 参数为空即可使用。
