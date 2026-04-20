# MindStudio Modeling Scene Generator

MindStudio Modeling Scene Generator 是一个基于 Gradio 的图形化界面工具，旨在简化使用 `msmodeling` 框架对 ModelScope 上的大语言模型进行性能优化和仿真的流程。

## 核心功能

- **自动化模型准备**：通过输入 ModelScope 模型 ID，自动同步并下载模型权重。
- **吞吐量优化**：图形化配置参数，自动执行 `throughput_optimizer` 并展示最佳配置方案。
- **性能仿真**：模拟特定输入长度下的推理性能，获取延迟等关键指标。
- **参数校验**：内置严格的输入校验（如正整数限制），防止无效配置导致脚本崩溃。

## 快速开始

### 1. 环境准备

确保您的系统已安装 `uv` 和 Git。

```bash
# 克隆 msmodeling 仓库
git clone https://gitcode.com/Ascend/msmodeling.git

# 安装 Python 依赖
pip install -r msmodeling/requirements.txt

# 复制项目配置
cp ms_pyproject.toml msmodeling/pyproject.toml

# 同步虚拟环境
uv sync
```

### 2. 启动应用

```bash
uv run python main.py
```

启动后，在浏览器中打开显示的本地 URL（通常为 `http://127.0.0.1:7860`）。

## 使用指南

### 模式选择

- **吞吐量优化**：用于寻找在满足时延约束（TTFT/TPOT）下的最大吞吐量配置。
- **性能仿真**：用于测试特定模型和长度下的推理性能。

### 关键参数说明

- **模型 ID**：ModelScope 上的模型标识符（例如 `qwen/Qwen2.5-7B-Instruct`）。
- **设备类型**：目标硬件平台（如 `910B4`）。
- **设备数量**：用于推理的并行设备数（必须为正整数）。
- **量化方案**：可选 `W8A8` 或 `W8A16`。

## 硬件要求

- 推荐在 **Ascend NPU** 环境下运行，以获得准确的性能反馈和优化建议。

## 技术架构

- **UI 框架**：Gradio
- **核心引擎**：`msmodeling` (liuren-modeling)
- **权重来源**：ModelScope Hub

# TODO

- 模型列表要有所限制
- stdout 加工，把重要信息抽取出来
- 免责声明
- mcp tool化
