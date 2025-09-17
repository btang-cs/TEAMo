# TEAMo Demo 项目说明 / TEAMo Demo Overview

本仓库演示了论文《TEAMo: Trait and Emotion Aware Motion Generation in 3D Human》中提到的核心逻辑，包括数据处理、模型组件与训练推理流程。This repository showcases the core logic described in the paper *TEAMo: Trait and Emotion Aware Motion Generation in 3D Human*, covering data handling, model components, and end-to-end training/inference.

## 功能概览 / Features
- **数据处理管线 Data Pipeline**：按元数据加载多模态特征，完成音频、文本、动作的预处理，并构建 PyTorch DataModule。Supports metadata-driven loading of multimodal features, preprocessing for audio/text/motion, and DataModule construction.
- **核心模型组件 Core Modules**：实现 Mamba Tagger、Trait & Emotion Aware Denoising Module (TEADM)，以及受到 TEA 理论约束的损失函数。Implements Mamba Tagger, TEADM, and the TEA-theory-inspired loss.
- **训练与推理脚本 Training & Inference**：提供配置化训练入口（支持调度器与 EMA）以及推理引擎，用于载入检查点并生成个性化动作。Offers configurable training (with scheduler/EMA) and an inference engine to load checkpoints and synthesize trait-aware motions.

## 快速开始 / Quick Start
1. **准备数据 Prepare Data**：根据 `data/schema.py` 组织元数据与特征路径，确保音频、文本、动作等模态可被读取。Arrange metadata and feature paths as defined in `data/schema.py` so all modalities are accessible.
2. **创建配置 Create Config**：参考 `train.py`，编写 YAML/JSON 配置文件，填写数据集、预处理、模型与训练超参。Write a YAML/JSON config (see `train.py`) specifying dataset, preprocessing, model, and training hyperparameters.
3. **启动训练 Run Training**：
   ```bash
   python -m teamo.train --config path/to/config.yaml
   ```
4. **执行推理 Run Inference**：
   ```python
   from pathlib import Path
   from teamo.inference import TEAMoInferenceEngine

   engine = TEAMoInferenceEngine.from_run_directory(Path("runs/experiment_001"))
   outputs = engine.generate(latent, conditions, tagger_inputs)
   ```

## 注意事项 / Notes
- 本 Demo 主要用于展示关键逻辑，未包含完整的数据准备、RVQ-VAE 编码器或评测脚本。This demo focuses on core logic; full data preparation, RVQ-VAE encoders, and evaluation scripts are not included.
- 部分模块依赖外部库（如 `torchaudio`），请根据需求自行安装。Some components require optional dependencies (e.g., `torchaudio`); install as needed.
- 在真实项目中，请结合具体场景扩展数据加载、模型规模及评估流程。Adapt the pipeline with real datasets, larger models, and evaluation protocols for production use.
