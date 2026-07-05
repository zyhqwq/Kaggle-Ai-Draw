# AI Drawing Platform — Kaggle GPU 版

[![Kaggle](https://img.shields.io/badge/Run%20on-Kaggle-20BEFF?style=for-the-badge&logo=kaggle)](https://www.kaggle.com/code/yalsdd/kaggle-ai-draw)  [![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github)](https://github.com/zyhqwq/Kaggle-Ai-Draw)   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/104yO2a6p8_Fd7Cr_G70O_oq-5zYRtVss?usp=sharing)

> **💡 前排说明：** 此脚本当作玩具就好，不要指望能生成像 GPT-5.5 一样的图像，但不影响预选的这些模型生图是挺优秀的。


## 功能

- 文生图（Text-to-Image）
- 正向 / 负向提示词
- 可调参数：步数、引导强度、尺寸、随机种子、生成数量
- 自动保存图片到 `generated_images/`
- 历史记录浏览
- Gradio 网页界面（可分享链接）

## 当前模型效果展示 — `cagliostrolab/animagine-xl-4.0`

| | | |
|---|---|---|
| ![示例1](image.webp) | ![示例2](image%20(1).webp) | ![示例3](image%20(2).webp) |

> 也就抽了不到10次卡嘻嘻。
---


## 支持的模型

在 `Kaggle.py` 中修改 `MODEL_ID` 即可切换：

| 模型 | 类型 | HF 链接 | 说明 |
|------|------|---------|------|
| `cagliostrolab/animagine-xl-4.0` | SDXL | [🔗](https://huggingface.co/cagliostrolab/animagine-xl-4.0) | ⭐ 当前模型 |
| `cagliostrolab/animagine-xl-3.1` | SDXL | [🔗](https://huggingface.co/cagliostrolab/animagine-xl-3.1) | 另一个SDXL 模型 |
| `CompVis/stable-diffusion-v1-4` | SD 1.5 | [🔗](https://huggingface.co/CompVis/stable-diffusion-v1-4) | 基础模型 |

> 部分模型需在 Hugging Face 上同意授权协议并设置 `HF_TOKEN`。

## Kaggle 使用方式

### 1. 直接运行（推荐）


Cell 保持运行，终端会输出 Gradio 公共链接：

```
Running on public URL: https://xxxxx.gradio.live
```

在浏览器中打开即可使用。

### 2. 直接调函数生图（无需网页）

```python
from Kaggle import generate, list_generated

images, msg = generate(
    prompt="masterpiece, best quality, 1girl, cat ears, white hair",
    negative_prompt="lowres, bad anatomy",
    seed=42
)
list_generated()
```

### 3. 设置 HF Token（如需授权模型）

```python
import os
os.environ['HF_TOKEN'] = 'Huggingface Token在这里设置'
```

Token 在 https://huggingface.co/settings/tokens 创建。

## Web UI 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| Prompt | — | 正向提示词，描述想要的内容 |
| Negative Prompt | — | 负向提示词，描述不想要的内容 |
| Width / Height | 1024 | 图片尺寸（SDXL 推荐 1024×1024） |
| Steps | 25 | 推理步数，越高越精细（15-50） |
| Guidance | 7.0 | 提示词引导强度（5-12） |
| Seed | -1 | 随机种子，-1 为随机 |
| N | 1 | 一次生成图片数 |

## 环境清理

脚本启动时自动执行：

- 屏蔽多余日志警告
- 删除临时缓存目录
- 释放 GPU 显存
- 检测并安装缺失依赖

## 输出目录

默认保存在 `/kaggle/working/generated_images/`（Kaggle 环境）或 `./generated_images/`（本地），文件名格式为 `img_YYYYMMDD_HHMMSS_seedXXX_0.png`。

## 📚 学习用途声明

**本项目仅供学习、研究和教育目的使用。**
- 用于了解 Stable Diffusion XL 模型推理流程
- 用于学习 Gradio Web UI 搭建
- 用于实践 Kaggle GPU 加速
- **禁止**用于商业用途、生产环境或任何形式的盈利行为

## ⚠️ 重要：禁止使用 NSFW 模型

**本项目严格遵守内容安全准则，严禁用于生成任何 NSFW（Not Safe For Work）内容**

**所有使用的模型均为安全审核通过的版本：**
- 所有模型加载时设置 `safety_checker=None` 仅用于性能优化，不代表去除安全限制
- 用户应遵守 Kaggle 社区准则和 Hugging Face 模型许可协议
- 如发现违规使用，将被Kaggle平台封号处理，本人不负责

**请合理合法使用本脚本，仅用于创作健康、积极、合规的图片内容。**
