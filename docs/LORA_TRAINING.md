# ACE-Step LoRA Training for ComfyUI

本文档介绍如何在ComfyUI中使用ACE-Step的LoRA训练功能。

## 概述

LoRA (Low-Rank Adaptation) 训练允许你使用自定义音频数据微调ACE-Step模型，创建特定风格的适配器。

### 训练流程

```mermaid
flowchart LR
    A[收集音频文件] --> B[准备训练数据]
    B --> C[预处理成张量]
    C --> D[训练LoRA]
    D --> E[测试训练后的模型]

    style A fill:#1e3a5f,color:#fff
    style B fill:#2c5282,color:#fff
    style C fill:#3a5f7d,color:#fff
    style D fill:#4a6fa5,color:#fff
    style E fill:#5c7caf,color:#fff
```

## 新增节点

### 1. ACE-Step Create Training Sample

创建单个训练样本，将音频和元数据保存到文件。

**输入参数：**
- `audio`: 音频文件 (AUDIO类型)
- `caption`: 音乐风格描述
- `output_dir`: 样本保存目录
- `lyrics`: 歌词 (纯音乐使用 "[Instrumental]")
- `bpm`: 拍子 (0表示自动检测)
- `keyscale`: 调性 (如 "C Major", "Am")
- `language`: 语言 (en, zh, instrumental等)

**输出：**
- `sample_info`: 保存的样本信息

### 2. ACE-Step Collect Training Samples

从目录收集所有训练样本。

**输入参数：**
- `samples_dir`: 包含训练样本的目录

**输出：**
- `audio_files`: 逗号分隔的音频文件路径
- `captions`: 逗号分隔的描述
- `lyrics_list`: 逗号分隔的歌词
- `bpm_list`: 逗号分隔的BPM值
- `keyscale_list`: 逗号分隔的调性
- `language_list`: 逗号分隔的语言

### 3. ACE-Step LoRA Prepare Training Data

将音频文件和元数据预处理成训练用的张量文件。

**输入参数：**
- `audio_files`: 逗号分隔的音频文件路径
- `captions`: 逗号分隔的描述
- `lyrics_list`: 逗号分隔的歌词
- `bpm_list`: 逗号分隔的BPM值
- `keyscale_list`: 逗号分隔的调性
- `language_list`: 逗号分隔的语言
- `output_dir`: 张量文件输出目录
- `checkpoint_dir`: ACE-Step模型目录
- `config_path`: 模型配置 (推荐 acestep-v15-turbo)
- `device`: 计算设备
- `audio_input_1~4`: 可选的音频输入连接
- `custom_tag`: 自定义标签 (添加到所有描述)
- `tag_position`: 标签位置 (append/prepend/replace)
- `max_duration`: 最大音频时长 (秒)

**输出：**
- `tensor_dir`: 预处理张量目录
- `status`: 状态信息

### 4. ACE-Step LoRA Train

使用预处理的张量文件训练LoRA适配器。

**输入参数：**
- `tensor_dir`: 预处理张量目录
- `checkpoint_dir`: ACE-Step模型目录
- `config_path`: 模型配置
- `output_dir`: LoRA输出目录
- `lora_rank`: LoRA秩 (4-256，推荐64)
- `lora_alpha`: LoRA缩放因子 (推荐为rank的2倍)
- `lora_dropout`: Dropout概率 (0.0-0.5)
- `learning_rate`: 学习率 (推荐3e-4)
- `train_epochs`: 训练轮数 (100-4000)
- `batch_size`: 批大小 (1-8)
- `gradient_accumulation`: 梯度累积步数
- `save_every_n_epochs`: 每N轮保存一次
- `seed`: 随机种子
- `resume_from`: 从检查点恢复训练
- `target_modules`: 目标模块 (逗号分隔)

**输出：**
- `lora_info`: 训练好的LoRA信息
- `training_log`: 训练日志

## 使用示例

### 方法1: 使用文件路径

1. 准备音频文件和对应的元数据
2. 在 "LoRA Prepare Training Data" 节点中输入：
   - audio_files: `path/to/song1.wav,path/to/song2.wav,path/to/song3.wav`
   - captions: `Electronic pop with synth,Upbeat dance track,Chill lo-fi beat`
   - lyrics_list: `[Instrumental],[Instrumental],[Instrumental]`
   - bpm_list: `120,128,90`
   - keyscale_list: `C Major,D Minor,F Major`

### 方法2: 使用音频输入连接

1. 使用 ComfyUI 的 Load Audio 节点加载音频
2. 连接到 "LoRA Prepare Training Data" 节点的 audio_input_1~4
3. 填写对应的元数据

## 训练配置建议

### 快速测试 (10个样本)
```
lora_rank: 32
lora_alpha: 64
train_epochs: 500
batch_size: 1
learning_rate: 3e-4
```

### 标准训练 (50-100个样本)
```
lora_rank: 64
lora_alpha: 128
train_epochs: 1000
batch_size: 1
learning_rate: 3e-4
```

### 高质量训练 (200+个样本)
```
lora_rank: 128
lora_alpha: 256
train_epochs: 2000
batch_size: 2
gradient_accumulation: 2
learning_rate: 1e-4
```

## 训练约束

开始训练前，请确保以下设置已禁用：
- ❌ CPU卸载 (会大幅降低训练速度)
- ❌ 模型编译 (与LoRA不兼容)
- ❌ INT8量化 (与LoRA不兼容)
- ❌ LLM初始化 (占用训练所需GPU显存)

## 训练后使用

训练完成后，`lora_info` 输出可以直接连接到任何ACE-Step生成节点：
- Text to Music
- Cover
- Repaint
- Simple Mode

## 故障排除

### 内存不足
- 减小 `batch_size`
- 减小 `lora_rank`
- 使用更少的训练样本

### 训练速度慢
- 确保使用 GPU (cuda/mps)
- 禁用所有不兼容选项
- 减小 `max_duration` 限制音频长度

### 训练质量不佳
- 增加训练样本数量
- 提高 `train_epochs`
- 调整 `learning_rate`
- 确保数据质量 (音频清晰、描述准确)
