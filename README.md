Diffusers on macOS Quickstart
====

开发环境
----
1. VSCode + Python Extension Pack（含 Python & Jupyter Notebook 支持）
2. Homebrew (https://brew.sh)
3. Command Line Tools for Xcode (安装 Homebrew 时，按提示即可自动安装)
4. Python 3.11.0+（Homebrew 安装）

安装 Python 3.11.0+
----
```bash
brew install python@3.11
```

初始化 venv
----
```bash
mkdir -p ~/Projects/diffusers
cd ~/Projects/diffusers
python3 -m venv venv
source venv/bin/activate
```

安装依赖包
----
包括了 Pytorch、Huggingface Transformers 等必备包。

```bash
pip install -r requirements.txt
```

快速尝试
----
使用 Hugginface Pokemon 数据集直接训练：

```bash
./train.sh
```

训练中断时，可以使用 `./train.sh --resume_from_checkpoint=latest` 继续训练。

使用模型
----
在 VSCode 中，打开 `sd15.ipynb` 或 `sd15_lora.ipynb`，按 `Shift + Enter` 逐步执行代码块。

1. `sd15.ipynb` 使用基线 SD-1.5 模型出图
2. `sd15_lora.ipynb` 使用训练出的 Lora 模型出图

其他
----
训练时间较长，在 macOS 上，可以使用 `caffeinate` 命令防止电脑休眠和禁止屏幕保护：

```bash
caffeinate -d ./train.sh
```

参考资料
----
1. https://huggingface.co/blog/lora Using LoRA for Efficient Stable Diffusion Fine-Tuning
2. https://huggingface.co/docs/diffusers/optimization/mps macOS 配置 GPU 加速 diffusers

One more thing
----
1. https://github.com/AUTOMATIC1111/stable-diffusion-webui 一个基于 Streamlit 的 Web UI，可以在浏览器中傻瓜式直接训练和使用 SD-1.5 模型。
