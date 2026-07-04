import os, sys, gc, warnings, shutil, subprocess, importlib, platform, time, random
from pathlib import Path
from datetime import datetime

# ============================================================
# 1. 环境清理：屏蔽日志、删除缓存、释放显存
# ============================================================

warnings.filterwarnings('ignore')
for _k in ['TF_CPP_MIN_LOG_LEVEL', 'CUDA_LAUNCH_BLOCKING', 'PYTHONWARNINGS',
           'DIFFUSERS_VERBOSITY', 'HUGGINGFACE_HUB_VERBOSITY', 'TRANSFORMERS_VERBOSITY']:
    os.environ[_k] = 'error' if 'VERBOSITY' in _k else '3'

for _d in ['/tmp/hf_cache', '/tmp/torch_cache', '/kaggle/working/tmp', '/tmp/__pycache__']:
    if os.path.exists(_d):
        shutil.rmtree(_d, ignore_errors=True)
gc.collect()

import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    print(f'GPU memory: {free/1024**3:.1f}/{total/1024**3:.1f} GB')
else:
    print('WARNING: CUDA not available.')

print(f'Python {sys.version.split()[0]} | {platform.system()} {platform.release()}')
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f'{p.name} | VRAM: {p.total_memory/1024**3:.1f}GB | CUDA {torch.version.cuda}')

# ============================================================
# 2. 依赖检测：缺失则自动 pip install
# ============================================================

_REQUIRED = ['diffusers', 'transformers', 'accelerate', 'safetensors', 'matplotlib', 'gradio']
_missing = [p for p in _REQUIRED if not importlib.util.find_spec(p.replace('-', '_'))]
if _missing:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', '--upgrade'] + _missing)
# 锁定 Pillow 版本避免与 torchvision 冲突
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', '--upgrade', 'pillow>=8.0,<12.0'])

from diffusers import StableDiffusionXLPipeline, AutoencoderKL
from huggingface_hub import login
from PIL import Image
import gradio as gr

# ============================================================
# 3. Hugging Face 登录（如需 gated model）
# ============================================================

HF_TOKEN = os.environ.get('HF_TOKEN', '') or 'Huggingface Token填这里'
if HF_TOKEN:
    login(token=HF_TOKEN, add_to_git_credential=False)
    print('HF login OK')

# ============================================================
# 4. 加载 SDXL 模型
# ============================================================

MODEL_ID = 'Laxhar/noobai-XL-1.0'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE = torch.float16 if DEVICE == 'cuda' else torch.float32

print(f'Loading {MODEL_ID} on {DEVICE}...')
pipe = StableDiffusionXLPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=DTYPE,
    use_safetensors=True,
    token=HF_TOKEN or None,
)
if DEVICE == 'cuda':
    pipe.enable_attention_slicing()       # 降低显存占用
    pipe.enable_model_cpu_offload()       # 不用的模块卸载到 CPU
    torch.cuda.empty_cache()
print(f'Model loaded. UNet params: {sum(p.numel() for p in pipe.unet.parameters())/1e6:.1f}M')

# ============================================================
# 5. 输出目录
# ============================================================

OUTPUT_DIR = '/kaggle/working/generated_images' if os.path.exists('/kaggle/working') else './generated_images'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 6. 核心函数：文生图
# ============================================================

def generate(prompt, negative_prompt='', width=1024, height=1024, steps=25,
             guidance=7.0, seed=-1, num_images=1):
    prompt = (prompt or '').strip()
    negative_prompt = (negative_prompt or '').strip()
    if not prompt:
        return [], 'Please enter a prompt.'
    seed = random.randint(0, 2**32 - 1) if seed == -1 else seed
    generator = torch.Generator(device='cpu').manual_seed(seed)
    start = time.time()
    with torch.no_grad():
        result = pipe(prompt=prompt,
                      negative_prompt=negative_prompt if negative_prompt else None,
                      width=width, height=height,
                      num_inference_steps=steps,
                      guidance_scale=guidance,
                      generator=generator,
                      num_images_per_prompt=num_images)
    elapsed = time.time() - start
    for i, img in enumerate(result.images):
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        path = os.path.join(OUTPUT_DIR, f'img_{ts}_seed{seed}_{i}.png')
        img.save(path)
    return result.images, f'Done in {elapsed:.1f}s (seed={seed})'

# ============================================================
# 7. 辅助函数：列出历史图片
# ============================================================

def list_generated():
    files = sorted(Path(OUTPUT_DIR).glob('*.png'), key=os.path.getmtime, reverse=True)
    if not files:
        return [], 'No images yet.'
    return [Image.open(f) for f in files[:12]], f'{len(files)} total, showing {min(12, len(files))}'

# ============================================================
# 8. Gradio Web UI
# ============================================================

def build_ui():
    with gr.Blocks(title='AI Drawing Platform SDXL', theme=gr.themes.Soft(),
                   css='footer{display:none !important}') as ui:
        gr.Markdown('# AI Drawing Platform SDXL')
        gr.Markdown(f'**{MODEL_ID}** | {DEVICE.upper()} | `{OUTPUT_DIR}`')
        with gr.Row():
            with gr.Column(scale=2):
                p = gr.Textbox(label='Prompt', placeholder='masterpiece, best quality, 1girl, ...', lines=3)
                n = gr.Textbox(label='Negative Prompt', placeholder='nsfw, lowres, bad anatomy, ...', lines=2)
                with gr.Row():
                    w = gr.Dropdown(label='Width', choices=[512, 640, 768, 896, 1024], value=1024)
                    h = gr.Dropdown(label='Height', choices=[512, 640, 768, 896, 1024], value=1024)
                with gr.Row():
                    s = gr.Slider(label='Steps', minimum=10, maximum=50, value=25, step=1)
                    g = gr.Slider(label='Guidance', minimum=1.0, maximum=15.0, value=7.0, step=0.5)
                with gr.Row():
                    sd = gr.Number(label='Seed (-1=random)', value=-1, precision=0)
                    ni = gr.Dropdown(label='N', choices=[1, 2, 3, 4], value=1)
                btn = gr.Button('Generate', variant='primary', size='lg')
                msg = gr.Textbox(label='Status', interactive=False)
            with gr.Column(scale=3):
                gal = gr.Gallery(label='Output', columns=2, object_fit='contain', height='auto')
        with gr.Row():
            ref = gr.Button('Refresh History')
            his = gr.Gallery(label='Recent 12', columns=4, object_fit='contain', height=280)
        btn.click(fn=generate, inputs=[p, n, w, h, s, g, sd, ni], outputs=[gal, msg])
        ref.click(fn=list_generated, inputs=[], outputs=[his, msg])
    return ui

# ============================================================
# 9. 启动服务器（自动查找空闲端口）
# ============================================================

def launch_web():
    import socket
    port = 7860
    for _ in range(10):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(('127.0.0.1', port)) != 0:
                break
        port += 1
    ui = build_ui()
    print(f'\nStarting web server on port {port}...')
    ui.launch(server_name='0.0.0.0', server_port=port, share=True, debug=False)
    print('Server stopped unexpectedly.')
    while True:
        time.sleep(1)

if __name__ == '__main__':
    launch_web()
