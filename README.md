# AI Drawing Platform — Kaggle GPU 版

基于 **Stable Diffusion XL (SDXL)** 的 AI 绘图平台，在 **Kaggle Tesla T4 (14.6GB)** 上运行，提供 **Gradio Web UI**。

## 功能

- 文生图（Text-to-Image）
- 正向 / 负向提示词
- 可调参数：步数、引导强度、尺寸、随机种子、生成数量
- 自动保存图片到 `generated_images/`
- 历史记录浏览
- Gradio 网页界面（可分享链接）

## 支持的模型

在 `ai_image_platform.py` 中修改 `MODEL_ID` 即可切换：

| 模型 | 类型 | 说明 |
|------|------|------|
| `Laxhar/noobai-XL-1.0` | SDXL | ⭐ 当前模型，2024 顶级动漫模型 |
| `cagliostrolab/animagine-xl-3.1` | SDXL | 高质量动漫 SDXL 模型 |
| `CompVis/stable-diffusion-v1-4` | SD 1.5 | 基础模型，兼容性最好 |

> 部分模型需在 Hugging Face 上同意授权协议并设置 `HF_TOKEN`。

## Kaggle 使用方式

### 1. 直接运行（推荐）

```python
!python /kaggle/working/ai_image_platform.py
```

Cell 保持运行，终端会输出 Gradio 公共链接：

```
Running on public URL: https://xxxxx.gradio.live
```

在浏览器中打开即可使用。

### 2. 直接调函数生图（无需网页）

```python
from ai_image_platform import generate, list_generated

images, msg = generate(
    prompt="masterpiece, best quality, 1girl, cat ears, white hair",
    negative_prompt="nsfw, lowres, bad anatomy",
    seed=42
)
list_generated()
```

### 3. 设置 HF Token（如需授权模型）

```python
import os
os.environ['HF_TOKEN'] = 'hf_your_token_here'
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
