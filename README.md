# mini RVC

mini RVC 是对 [Retrieval-based-Voice-Conversion-WebUI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) 的重新实现。

## 安装

项目要求 Python 3.10，推荐使用 `uv`：

```bash
git clone <this-repo>
cd minirvc
uv venv --python 3.10
uv sync
```

运行命令时使用：

```bash
uv run <command>
```

当前 Python 依赖只有：

```text
av
numpy
sounddevice
torch
mlx（仅 Apple Silicon 上安装，用于 MLX 后端）
```

## 下载和放置模型

本仓库不提交预训练权重。请从原 RVC 项目的模型发布位置下载 HuBERT、RMVPE 和预训练 G/D 权重，常用来源是 [lj1995/VoiceConversionWebUI](https://huggingface.co/lj1995/VoiceConversionWebUI)。

## 数据预处理

假设输入音频放在 `wav/`，实验目录使用 `logs/test/`。

### 1. 音频预处理

```bash
uv run minirvc-preprocess-audio wav logs/test --sample-rate 40000 --workers 14
```

输出：

```text
logs/test/0_gt_wavs
logs/test/1_16k_wavs
```

`0_gt_wavs` 是训练采样率音频，`1_16k_wavs` 用于 F0 和 HuBERT。

### 2. 提取 F0

f0 模型需要这一步；nof0 模型可以跳过。

```bash
uv run minirvc-extract-f0 logs/test --model assets/rmvpe/rmvpe.pt --workers 1 --device cuda:0 --batch-size 8
```

Apple Silicon 上可使用 MLX 后端：

```bash
uv run minirvc-extract-f0 logs/test --model assets/rmvpe/rmvpe.pt --backend mlx --device gpu --workers 1 --batch-size 8
```

输出：

```text
logs/test/2a_f0
logs/test/2b-f0nsf
```

### 3. 提取 HuBERT 特征

v2：

```bash
uv run minirvc-extract-hubert logs/test --model assets/hubert/hubert_base.pt --version v2 --device cuda:0 --batch-size 16
```

Apple Silicon 上可使用 MLX 后端：

```bash
uv run minirvc-extract-hubert logs/test --model assets/hubert/hubert_base.pt --version v2 --backend mlx --device gpu --batch-size 16
```

v1：

```bash
uv run minirvc-extract-hubert logs/test --model assets/hubert/hubert_base.pt --version v1 --device cuda:0 --batch-size 16
```

输出：

```text
logs/test/3_feature768
logs/test/3_feature256
```

## 准备 mute 样本

训练 filelist 需要 mute 样本。可以从旧 RVC 目录复制，也可以生成最小 mute 资产：

```bash
uv run python scripts/prepare_mute_assets.py
```

输出目录：

```text
logs/mute
```

## 构建训练 filelist

v2 40k f0：

```bash
uv run minirvc-build-filelist logs/test --version v2 --sample-rate 40k --f0 --mute-root logs/mute --output logs/test/filelist.txt
```

v2 40k nof0：

```bash
uv run minirvc-build-filelist logs/test --version v2 --sample-rate 40k --mute-root logs/mute --output logs/test/filelist.txt
```

## 训练

v2 40k f0：

```bash
uv run minirvc-train logs/test --version v2 --sample-rate 40k --f0 --batch-size 20 --epochs 20 --save-every-epoch 20
```

Apple Silicon 上可使用 MLX 后端：

```bash
uv run minirvc-train logs/test --version v2 --sample-rate 40k --f0 --batch-size 4 --epochs 20 --save-every-epoch 20 --backend mlx --device gpu --precision bf16
```

MLX 训练默认使用 `fp32`；`--precision bf16` 会把模型和训练输入切到 BF16，mel/loss 关键计算仍保持 fp32。

v2 40k nof0：

```bash
uv run minirvc-train logs/test --version v2 --sample-rate 40k --batch-size 20 --epochs 20 --save-every-epoch 20
```

训练默认会从下面路径推断预训练 G/D：

```text
assets/pretrained
assets/pretrained_v2
```

也可以显式指定：

```bash
uv run minirvc-train logs/test --version v2 --sample-rate 40k --f0 --batch-size 20 --epochs 20 --pretrain-g assets/pretrained_v2/f0G40k.pth --pretrain-d assets/pretrained_v2/f0D40k.pth
```

训练结束后会导出小模型：

```text
logs/test/test.pth
logs/test/test.mlx.npz  # MLX 后端
```

## 构建检索索引

v2：

```bash
uv run minirvc-build-index logs/test --version v2 --output logs/test/feature_v2.npz
```

v1：

```bash
uv run minirvc-build-index logs/test --version v1 --output logs/test/feature_v1.npz
```

索引文件是 `.npz`，内部保存 feature matrix。Torch 推理使用 torch exact top-k，MLX 推理使用 MLX exact top-k，不需要 FAISS。

## 推理

单文件推理：

```bash
uv run minirvc-infer input.wav output.wav --model logs/test/test.pth
```

Apple Silicon 上可使用 MLX 后端加载 `.mlx.npz`：

```bash
uv run minirvc-infer input.wav output.wav --model logs/test/test.mlx.npz --backend mlx --device gpu --precision bf16
```

目录批量推理：

```bash
uv run minirvc-infer wav_test outputs --model logs/test/test.pth
```

启用检索增强：

```bash
uv run minirvc-infer input.wav output.wav --model logs/test/test.pth --index logs/test/feature_v2.npz --index-rate 0.5
```

常用参数：

```text
--sid                 说话人 ID，默认 0
--f0-up-key           半音升降调，默认 0
--protect             F0 模型无声区特征保护，默认 0.33
--index-rate          检索混合比例，0 到 1，默认 0
--index-top-k         检索 top-k，默认 8
--split-pad-seconds   推理分段前后 pad 秒数，默认按设备精度沿用原版
--split-query-seconds 推理切点搜索窗口秒数，默认按设备精度沿用原版
--split-center-seconds 推理切点间隔秒数，默认按设备精度沿用原版
--split-max-seconds   超过该长度启用推理分段，默认按设备精度沿用原版
--device              例如 cuda:0 或 cpu
--backend             torch 或 mlx，默认 torch
--precision           MLX 推理精度：fp32、bf16、fp16，默认 fp32
--no-half             禁用半精度推理
```

## 实时推理

列出音频设备：

```bash
uv run minirvc-realtime --list-devices
```

启动实时变声：

```bash
uv run minirvc-realtime --model logs/test/test.pth --device cuda:0 --input-device 0 --output-device 1
```

离线模拟实时 block 处理：

```bash
uv run minirvc-realtime --model logs/test/test.pth --offline-input input.wav --offline-output realtime.wav
```

常用实时参数：

```text
--block-time          每次转换的音频块时长，默认 0.25
--crossfade-time      SOLA/crossfade 时长，默认 0.05
--extra-time          HuBERT/F0 上下文时长，默认 2.5
--f0-up-key           半音升降调，默认 0
--index-rate          检索混合比例，默认 0
```

## ckpt 工具

查看模型信息：

```bash
uv run minirvc-model-info logs/test/test.pth
```

融合两个同架构小模型：

```bash
uv run minirvc-merge-models model_a.pth model_b.pth merged.pth --alpha 0.5
```

`alpha` 是 `model_a` 的权重比例，`1 - alpha` 是 `model_b` 的权重比例。两个模型必须有相同的 `version`、`sr`、`f0` 和 config。

## 命令列表

```text
minirvc-preprocess-audio
minirvc-extract-f0
minirvc-extract-hubert
minirvc-build-filelist
minirvc-train
minirvc-build-index
minirvc-infer
minirvc-realtime
minirvc-model-info
minirvc-merge-models
```

## 参考项目

+ [RVC-Project/Retrieval-based-Voice-Conversion-WebUI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)
+ [ContentVec](https://github.com/auspicious3000/contentvec/)
+ [VITS](https://github.com/jaywalnut310/vits)
+ [HIFIGAN](https://github.com/jik876/hifi-gan)
+ [Gradio](https://github.com/gradio-app/gradio)
+ [FFmpeg](https://github.com/FFmpeg/FFmpeg)
+ [Ultimate Vocal Remover](https://github.com/Anjok07/ultimatevocalremovergui)
+ [audio-slicer](https://github.com/openvpi/audio-slicer)
+ [Vocal pitch extraction:RMVPE](https://github.com/Dream-High/RMVPE)
  + The pretrained model is trained and tested by [yxlllc](https://github.com/yxlllc/RMVPE) and [RVC-Boss](https://github.com/RVC-Boss).
