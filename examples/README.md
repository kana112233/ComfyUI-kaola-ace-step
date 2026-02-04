# Workflow Examples / 工作流示例

This directory contains example ComfyUI workflows demonstrating various ACE-Step node capabilities.

本目录包含展示 ACE-Step 节点各种功能的 ComfyUI 工作流示例。

## Available Examples / 可用示例

### 1. Text to Music / 文本生成音乐
**File**: `text_to_music.json`

**Description**: Generate music from a simple text description.

**描述**: 从简单的文本描述生成音乐。

**Key Features / 关键特性**:
- Caption-based generation / 基于标题的生成
- Fixed seed for reproducibility / 固定种子以实现可重现性
- Batch generation (2 variations) / 批量生成（2个变体）
- 30-second duration / 30秒时长

**Use Case / 用例**: Quick background music generation / 快速生成背景音乐

---

### 2. Simple Mode / 简单模式
**File**: `simple_mode.json`

**Description**: Generate music from natural language with automatic metadata.

**描述**: 使用自然语言和自动元数据生成音乐。

**Key Features / 关键特性**:
- Natural language input / 自然语言输入
- LM-generated metadata (BPM, key, duration) / LM生成的元数据
- Random seed for variety / 随机种子以产生变化
- Sample info output / 示例信息输出

**Use Case / 用例**: Getting musical ideas from simple descriptions / 从简单描述获取音乐创意

---

### 3. Cover Generation / 翻唱生成
**File**: `cover_generation.json`

**Description**: Transform existing audio to a different style.

**描述**: 将现有音频转换为不同风格。

**Key Features / 关键特性**:
- Loads input audio / 加载输入音频
- Style transformation / 风格转换
- Cover strength control (0.7) / 翻唱强度控制
- Jazz piano arrangement example / 爵士钢琴编曲示例

**Use Case / 用例**: Creating covers or style transfers / 创建翻唱或风格转换

---

### 4. Batch Generation / 批量生成
**File**: `batch_generation.json`

**Description**: Generate multiple variations at once.

**描述**: 一次生成多个变体。

**Key Features / 关键特性**:
- Batch size of 4 / 批量大小为4
- Random seeds for variety / 随机种子以产生变化
- Cinematic music example / 电影音乐示例
- Efficient parallel generation / 高效并行生成

**Use Case / 用例**: Exploring multiple variations quickly / 快速探索多个变体

---

### 5. Music with Lyrics / 带歌词的音乐
**File**: `music_with_lyrics.json`

**Description**: Generate a complete song with lyrics.

**描述**: 生成完整的带歌词歌曲。

**Key Features / 关键特性**:
- Full lyrics with structure tags / 完整歌词带结构标签
- Vocal language specification / 人声语言规范
- Manual BPM and key / 手动 BPM 和调性
- 3-minute song / 3分钟歌曲
- Verse and chorus structure / 主歌和副歌结构

**Use Case / 用例**: Creating complete songs / 创建完整歌曲

---

## How to Use These Workflows / 如何使用这些工作流

### Method 1: Drag & Drop / 方法 1: 拖放

1. Open ComfyUI / 打开 ComfyUI
2. Drag the JSON file into the ComfyUI window / 将 JSON 文件拖入 ComfyUI 窗口
3. Adjust parameters as needed / 根据需要调整参数
4. Click "Queue Prompt" to execute / 点击"Queue Prompt"执行

### Method 2: Load Menu / 方法 2: 加载菜单

1. Open ComfyUI / 打开 ComfyUI
2. Click the "Load" button / 点击"Load"按钮
3. Select the workflow JSON file / 选择工作流 JSON 文件
4. Adjust parameters and execute / 调整参数并执行

---

## Configuration Required / 需要配置

Before running these workflows, make sure to:

在运行这些工作流之前，请确保：

1. **Install ACE-Step** / **安装 ACE-Step**
   ```bash
   git clone https://github.com/ace-step/ACE-Step-1.5.git acestep_repo
   cd acestep_repo
   uv sync
   uv run acestep-download
   ```

2. **Update Checkpoint Path** / **更新检查点路径**
   - In each workflow, update `checkpoint_dir` to point to your ACE-Step checkpoints
   - 在每个工作流中，更新 `checkpoint_dir` 指向你的 ACE-Step 检查点
   - Example: `/absolute/path/to/acestep_repo/checkpoints`

3. **Verify GPU Requirements** / **验证 GPU 要求**
   - Minimum 6GB VRAM for turbo model / Turbo 模型最少 6GB 显存
   - Recommended 12GB+ for LM features / LM 功能推荐 12GB+ 显存

---

## Customization Tips / 自定义提示

### Changing Music Style / 改变音乐风格

Edit the `caption` parameter to describe your desired music style:

编辑 `caption` 参数以描述你想要的音乐风格：

```
"upbeat electronic dance music with heavy bass"
"calm ambient piano music with soft strings"
"energetic rock music with electric guitar solos"
```

### Adjusting Duration / 调整时长

Change the `duration` parameter (in seconds):

更改 `duration` 参数（以秒为单位）：

- Short clips: 10-30 seconds / 短片段：10-30 秒
- Full songs: 180-300 seconds / 完整歌曲：180-300 秒
- Long form: Up to 600 seconds / 长形式：最多 600 秒

### Controlling Creativity / 控制创造力

Adjust these parameters:

调整这些参数：

- `seed`: Use -1 for random, fixed number for reproducible / 使用 -1 随机，固定数字可重现
- `lm_temperature`: 0.7-0.85 for conservative, 0.9-1.1 for creative / 0.7-0.85 保守，0.9-1.1 创意
- `thinking`: Enable/disable LM reasoning / 启用/禁用 LM 推理

### Quality vs Speed / 质量与速度

**Fast** (Turbo model):
- `inference_steps`: 8
- `thinking`: false
- Generation time: ~10s on RTX 3090

**Balanced**:
- `inference_steps`: 8-12
- `thinking`: true
- Generation time: ~15-20s on RTX 3090

**Quality** (Base model):
- `config_path`: `acestep-v15-base`
- `inference_steps`: 32-64
- `thinking`: true
- Generation time: ~40-80s on RTX 3090

---

## Troubleshooting / 故障排除

**Problem**: "ACE-Step is not installed"
- **Solution**: Follow the installation steps above / 按照上面的安装步骤操作

**Problem**: Out of memory errors
- **Solution**: Reduce `batch_size` or use smaller `lm_model_path` / 减少批量大小或使用更小的语言模型

**Problem**: Results don't match prompt
- **Solution**: Be more specific in caption, increase `guidance_scale` / 在标题中更具体，增加引导比例

---

## Creating Your Own Workflows / 创建自己的工作流

1. Start with an example that matches your use case / 从符合你用例的示例开始
2. Modify the caption and parameters / 修改标题和参数
3. Add/remove nodes as needed / 根据需要添加/删除节点
4. Save your custom workflow / 保存自定义工作流

For more detailed documentation, see [USAGE.md](../USAGE.md).

更多详细文档请参考 [USAGE.md](../USAGE.md)。

---

## Examples Gallery / 示例画廊

Want to share your workflow? Submit a PR or open an issue!

想分享你的工作流吗？提交 PR 或打开 issue！
