# ComfyUI-Kaola-ACE-Step

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![ComfyUI Registry](https://img.shields.io/badge/dynamic/json?color=blue&label=registry&prefix=v&query=version&url=https%3A%2F%2Fregistry.comfy.org%2Fnodes%2Fcomfyui-kaola-ace-step)](https://registry.comfy.org/packages/nodes/comfyui-kaola-ace-step/)

ComfyUI custom nodes for [ACE-Step 1.5](https://github.com/ace-step/ACE-Step-1.5) - A powerful open-source music generation model that achieves commercial-grade quality.

基于 [ACE-Step 1.5](https://github.com/ace-step/ACE-Step-1.5) 的 ComfyUI 自定义节点 - 商业级的开源音乐生成模型。

## Features / 功能特性

- 🎵 **Text to Music** - Generate music from text descriptions / 从文本描述生成音乐
- 🎭 **Cover Generation** - Transform audio to different styles / 风格转换和翻唱
- 🎨 **Audio Repaint** - Regenerate specific segments / 局部重绘音频
- 💡 **Simple Mode** - Natural language to music with auto-metadata / 自然语言生成
- 📝 **Format Sample** - Enhance and format user input / 格式化输入
- 🔍 **Understand Audio** - Analyze audio codes / 音频分析

## Quick Start / 快速开始

### Prerequisites / 前置要求

1. **Python 3.11** (必需) / Python 3.11 is required:
```bash
# 创建 conda 环境
conda create -n ace-step python=3.11 -y
conda activate ace-step
```

2. **Install ACE-Step Python Package** / 安装 ACE-Step Python 包:
```bash
# 从 GitHub 安装 acestep 包
pip install git+https://github.com/ace-step/ACE-Step-1.5.git
```

3. **Install PyTorch Dependencies** / 安装 PyTorch 依赖:
```bash
pip install torch torchvision torchaudio soundfile numpy
```

4. **Download Models** / 下载模型 (可选 / Optional):
```bash
# 方式 1: 使用 huggingface-cli (推荐)
huggingface-cli download ACE-Step/Ace-Step1.5 --local-dir /path/to/models

# 方式 2: 浏览器下载
# https://huggingface.co/ACE-Step/Ace-Step1.5
```

### Installation / 安装

#### Method 1: ComfyUI Manager (Recommended) / 方式 1: ComfyUI Manager（推荐）

Coming soon to ComfyUI Registry!

即将在 ComfyUI Registry 上线！

#### Method 2: Manual Install / 方式 2: 手动安装

```bash
# Navigate to ComfyUI custom_nodes directory
cd ComfyUI/custom_nodes

# Clone this repository
git clone https://github.com/kana112233/ComfyUI-kaola-ace_step.git

# Restart ComfyUI
```

### Configuration / 配置

**Option 1: Use Existing Models / 使用已有模型**

如果你已经下载了 ACE-Step 模型，直接放置到 ComfyUI 模型目录：

```bash
# 假设你的模型在 /path/to/Ace-Step1.5/
# 创建符号链接或复制文件
ln -s /path/to/Ace-Step1.5/* ComfyUI/models/acestep/
```

**Option 2: Download Models / 下载模型**

使用 huggingface-cli 下载模型到 ComfyUI 目录：

```bash
# 创建模型目录
mkdir -p ComfyUI/models/acestep

# 下载模型
huggingface-cli download ACE-Step/Ace-Step1.5 --local-dir ComfyUI/models/acestep
```

**Expected Structure / 期望目录结构:**

```
ComfyUI/models/acestep/
├── acestep-v15-turbo/          # DiT 模型
│   ├── config.json
│   ├── model.safetensors
│   └── ...
├── acestep-5Hz-lm-1.7B/        # LM 模型
├── vae/                         # VAE 模型
└── Qwen3-Embedding-0.6B/        # 文本编码器
```

**Node Settings / 节点设置:**

- **checkpoint_dir**: 留空 (自动使用 `ComfyUI/models/acestep/`)
- **config_path**: `acestep-v15-turbo` (快速) 或 `acestep-v15-base` (高质量)
- **lm_model_path**: `acestep-5Hz-lm-1.7B` (推荐)(MODEL_SETUP.md)

**Node Parameters / 节点参数**

- **checkpoint_dir**: Leave empty to use default ComfyUI model directory
  - 留空以使用 ComfyUI 默认模型目录 (`ComfyUI/models/acestep/`)
  - Or specify custom path if needed / 或指定自定义路径

- **config_path**: DiT model to use
  - `acestep-v15-turbo` (fast, recommended / 快速，推荐)
  - `acestep-v15-base` (high quality / 高质量)

- **lm_model_path**: Language model to use
  - `acestep-5Hz-lm-0.6B` (6-12GB VRAM / 显存)
  - `acestep-5Hz-lm-1.7B` (12-16GB VRAM, recommended / 推荐)
  - `acestep-5Hz-lm-4B` (16GB+ VRAM / 显存)

## Usage Examples / 使用示例

### Quick Examples / 快速示例

#### Example 1: Text to Music / 文本生成音乐
```
ACE-Step Text to Music Node:
├── caption: "upbeat electronic dance music with heavy bass"
├── duration: 30
├── bpm: 128
└── Execute → Generate audio
```

#### Example 2: Cover Generation / 翻唱生成
```
Load Audio → ACE-Step Cover Node:
├── caption: "jazz piano arrangement with swing feel"
└── audio_cover_strength: 0.7
```

#### Example 3: Simple Mode / 简单模式
```
ACE-Step Simple Mode Node:
└── query: "energetic K-pop dance track with catchy hooks"
```

### Full Workflow Examples / 完整工作流示例

For ready-to-use ComfyUI workflows, see the [examples/](examples/) directory:

完整的 ComfyUI 工作流示例，请查看 [examples/](examples/) 目录：

- 📝 **[Text to Music](examples/text_to_music.json)** - Generate music from text / 从文本生成音乐
- 💡 **[Simple Mode](examples/simple_mode.json)** - Natural language to music / 自然语言生成音乐
- 🎭 **[Cover Generation](examples/cover_generation.json)** - Style transfer / 风格转换
- 📦 **[Batch Generation](examples/batch_generation.json)** - Multiple variations / 批量生成
- 🎵 **[Music with Lyrics](examples/music_with_lyrics.json)** - Complete songs / 完整歌曲

See [examples/README.md](examples/README.md) for detailed usage instructions.

详细使用说明请参考 [examples/README.md](examples/README.md)。

## Nodes / 节点列表

| Node | Description |
|------|-------------|
| **ACE_STEP_TextToMusic** | Generate music from text with full parameter control |
| **ACE_STEP_Cover** | Transform existing audio to new style |
| **ACE_STEP_Repaint** | Regenerate specific segments of audio |
| **ACE_STEP_SimpleMode** | Generate from natural language (auto-metadata) |
| **ACE_STEP_FormatSample** | Format and enhance user input |
| **ACE_STEP_Understand** | Analyze audio semantic codes |

## System Requirements / 系统要求

- **GPU**: NVIDIA GPU with 6GB+ VRAM recommended / 推荐使用 6GB+ 显存的 NVIDIA GPU
- **CPU**: Works on CPU but slower / 支持 CPU 但速度较慢
- **Disk**: ~8GB for models / 约 8GB 磁盘空间用于模型
- **Python**: 3.10+ / Python 3.10 或更高版本

## Documentation / 文档

For detailed usage instructions, see [USAGE.md](USAGE.md).

详细使用说明请参考 [USAGE.md](USAGE.md)。

## Performance / 性能

- **Ultra-Fast**: ~10s per song on RTX 3090 (turbo model) / RTX 3090 上约 10 秒一首歌
- **Low VRAM**: <4GB for basic use / 基础使用小于 4GB 显存
- **High Quality**: Commercial-grade output / 商业级质量输出

## Troubleshooting / 故障排除

**Problem**: "ACE-Step is not installed"
- **Solution**: Install ACE-Step following the Quick Start guide / 按照快速开始指南安装

**Problem**: Out of memory errors
- **Solution**: Reduce `batch_size` or use smaller `lm_model_path` / 减少批量大小或使用更小的语言模型

**Problem**: Poor quality results
- **Solution**: Increase `inference_steps`, use base model / 增加推理步数，使用基础模型

## Acknowledgments / 致谢

Based on [ACE-Step 1.5](https://github.com/ace-step/ACE-Step-1.5) by ACE Studio and StepFun.

基于 ACE Studio 和 StepFun 的 [ACE-Step 1.5](https://github.com/ace-step/ACE-Step-1.5)。

## License / 许可证

MIT License - see [LICENSE](LICENSE) for details.

## Links / 链接

- [ACE-Step GitHub](https://github.com/ace-step/ACE-Step-1.5)
- [ACE-Step Hugging Face](https://huggingface.co/ACE-Step)
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
- [ComfyUI Registry](https://registry.comfy.org)

---

**Made with ❤️ by kana112233**
