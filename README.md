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

1. **Install ACE-Step** / 安装 ACE-Step:
```bash
# Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install ACE-Step
git clone https://github.com/ace-step/ACE-Step-1.5.git acestep_repo
cd acestep_repo
uv sync

# Download models (requires ~8GB disk space)
uv run acestep-download
```

2. **Install Python Dependencies** / 安装 Python 依赖:
```bash
pip install torch torchvision torchaudio soundfile
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

In any ACE-Step node, set the following paths:

在任何 ACE-Step 节点中，设置以下路径：

- **checkpoint_dir**: Path to ACE-Step checkpoints
  - Example: `/path/to/acestep_repo/checkpoints`

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
