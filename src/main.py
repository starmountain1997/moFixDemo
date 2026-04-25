import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import gradio as gr

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# Make src/ importable regardless of where the script is invoked from
_SRC_DIR = Path(__file__).parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from arg_reflector import CLIReflector  # noqa: E402

_CLI_DIR = Path(__file__).parent.parent / "msmodeling" / "cli"
_COMMON_FILE = _CLI_DIR / "utils.py"

MODEL_KEYWORDS = {
    "DeepSeek": ["deepseek", "deepseek_v3", "deepseek_v32"],
    "Qwen": ["qwen", "qwen3", "qwen2.5", "qwen3_vl", "qwen3_5"],
    "GLM": ["glm", "glm4", "glm4v", "chatglm"],
    "InternVL": ["internvl"],
    "ERNIE": ["ernie"],
    "Bailing": ["bailing"],
    "MiniMax": ["minimax", "m2"],
    "MIMO": ["mimo"],
    "Wan": ["wan"],
    "Hunyuan": ["hunyuan"],
}


def validate_model_id(model_id: str) -> bool:
    normalized = model_id.lower().replace("-", "_")
    return any(
        kw.lower() in normalized
        for keywords in MODEL_KEYWORDS.values()
        for kw in keywords
    )


def _make_handler(reflector: CLIReflector) -> Any:
    """
    Wrap CLIReflector with model-ID validation.
    Must be called after build_accordions() so component order is fixed.
    """
    dests = list(reflector.components.keys())

    def handler(*args: Any, progress: gr.Progress = gr.Progress()) -> str:
        values: dict[str, Any] = dict(zip(dests, args))
        model_id = str(values.get("model_id", "")).strip()

        if not model_id:
            return "## ❌ 错误\n**模型 ID** 是必选参数，请填写后再试。"

        if not validate_model_id(model_id):
            supported = ", ".join(MODEL_KEYWORDS.keys())
            return (
                f"## ❌ 预期不支持\n模型 ID `{model_id}` 可能不属于 `msmodeling` 目前支持的架构系列。\n\n"
                f"**当前支持的主流系列包括:**\n{supported}\n\n"
                "*如果您确认该模型架构兼容，请联系开发人员或直接尝试运行。*"
            )

        if progress:
            progress(0.2, desc="正在分析参数并生成命令...")

        cmd = reflector.build_command(values)

        if progress:
            progress(0.8, desc="正在执行 msmodeling...")

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
            out = "### 执行结果\n\n"
            out += f"**执行命令:** `{' '.join(cmd)}`\n\n"
            if result.stdout:
                out += f"#### 标准输出 (stdout):\n```text\n{result.stdout}\n```\n"
            if result.stderr:
                out += f"#### 标准错误 (stderr):\n```text\n{result.stderr}\n```\n"
            if result.returncode != 0:
                out += f"\n> ⚠️ **命令执行失败，退出码: {result.returncode}**"
            return out
        except Exception as exc:
            return f"## ❌ 执行错误\n无法运行命令：{exc}"

    return handler


# Build reflectors at import time (pure AST parsing, no tensor_cast import needed)
_sim_reflector = CLIReflector(
    _CLI_DIR / "inference" / "text_generate.py",
    common_file=_COMMON_FILE,
    cli_module="cli.inference.text_generate",
)
_opt_reflector = CLIReflector(
    _CLI_DIR / "inference" / "throughput_optimizer.py",
    common_file=_COMMON_FILE,
    cli_module="cli.inference.throughput_optimizer",
)

custom_css = """
#header {
    text-align: center;
    padding: 24px 0;
    margin-bottom: 24px;
}
#header h1 {
    font-size: 28px;
    color: #1890ff;
    margin-bottom: 12px;
    font-weight: 600;
}
#header p {
    font-size: 15px;
    color: #666;
    line-height: 1.5;
}
.feature-box {
    background: linear-gradient(135deg, #f0f7ff 0%, #e6f4ff 100%);
    border-left: 4px solid #1890ff;
    border-radius: 8px;
    padding: 16px 20px;
    margin-bottom: 20px;
}
.feature-box h4 {
    font-size: 15px;
    margin-bottom: 10px;
    color: #333;
}
.feature-box p {
    font-size: 13px;
    color: #666;
    margin-bottom: 6px;
}
.feature-box ul {
    margin-left: 16px;
    margin-top: 6px;
    font-size: 13px;
    color: #666;
}
.submit-btn {
    background: #1890ff !important;
    border: none !important;
    box-shadow: 0 2px 4px rgba(24, 144, 255, 0.2) !important;
}
.submit-btn:hover {
    background: #40a9ff !important;
    box-shadow: 0 4px 8px rgba(24, 144, 255, 0.3) !important;
}
"""

theme = gr.themes.Default(
    primary_hue="blue",
    secondary_hue="gray",
    neutral_hue="gray",
).set(
    body_background_fill="#f7f9fc",
    body_background_fill_dark="#1a1a2e",
)  # type: ignore[reportPrivateImportUsage]

with gr.Blocks(title="msModeling 推理仿真与参数寻优工具", theme=theme, css=custom_css) as demo:
    gr.HTML("""
    <div id="header">
        <h1>msModeling 推理仿真与参数寻优工具</h1>
        <p>基于 Ascend 平台的 AI 模型性能评估与优化工具，支持推理仿真和参数寻优两大核心功能</p>
    </div>
    """)

    with gr.Tabs():
        with gr.TabItem("🎯 推理仿真"):
            gr.HTML("""
            <div class="feature-box">
                <h4>推理仿真</h4>
                <p>用于评估模型在不同配置下的推理性能，包括并发测试和单次forward时间测量。</p>
                <p><strong>主要用途：</strong></p>
                <ul>
                    <li>测试不同并发数下的模型性能</li>
                    <li>评估单次forward时间</li>
                    <li>分析不同硬件设备的表现</li>
                    <li>验证量化对性能的影响</li>
                </ul>
            </div>
            """)

            comps_sim = _sim_reflector.build_accordions()

            gr.HTML("""
            <div style="background: #f6ffed; border: 1px solid #b7eb8f; border-radius: 6px; padding: 14px 16px; margin: 16px 0;">
                <h4 style="font-size: 13px; font-weight: 600; margin-bottom: 8px; color: #389e0d;">💡 智能推荐</h4>
                <p style="font-size: 12px; color: #52c41a; margin-bottom: 4px;">• 并发数：2（适合测试基础性能）</p>
                <p style="font-size: 12px; color: #52c41a; margin-bottom: 4px;">• 输入长度：3500（标准测试长度）</p>
                <p style="font-size: 12px; color: #52c41a;">• 图编译：启用（可提升性能）</p>
            </div>
            """)

            sim_output = gr.Markdown(label="执行结果")
            sim_btn = gr.Button("▶️ 执行仿真", variant="primary", elem_classes="submit-btn")
            sim_btn.click(
                _make_handler(_sim_reflector),
                inputs=list(comps_sim.values()),
                outputs=sim_output,
            )

        with gr.TabItem("🎨 参数寻优"):
            gr.HTML("""
            <div class="feature-box">
                <h4>参数寻优</h4>
                <p>在给定 SLO 约束下搜索最优吞吐配置，支持三种模式：</p>
                <ul>
                    <li><strong>aggregation</strong>：PD混部，联合优化 Prefill 和 Decode</li>
                    <li><strong>disaggregation</strong>：PD分离，分别评估各自性能</li>
                    <li><strong>pd-ratio</strong>：寻找最优 Prefill:Decode 实例配比</li>
                </ul>
                <p><strong>主要用途：</strong>寻找最佳批次大小、评估量化策略影响、分析 PD 分离部署性能</p>
            </div>
            """)

            comps_opt = _opt_reflector.build_accordions()

            gr.HTML("""
            <div style="background: #f6ffed; border: 1px solid #b7eb8f; border-radius: 6px; padding: 14px 16px; margin: 16px 0;">
                <h4 style="font-size: 13px; font-weight: 600; margin-bottom: 8px; color: #389e0d;">💡 智能推荐</h4>
                <p style="font-size: 12px; color: #52c41a; margin-bottom: 4px;">• 输入长度：3500，输出长度：1500（标准测试长度）</p>
                <p style="font-size: 12px; color: #52c41a; margin-bottom: 4px;">• TPOT约束：50ms，TTFT约束：7000ms（合理的性能约束）</p>
                <p style="font-size: 12px; color: #52c41a;">• 最大预填充令牌数：8192（标准设置）</p>
            </div>
            """)

            opt_output = gr.Markdown(label="执行结果")
            opt_btn = gr.Button("▶️ 执行寻优", variant="primary", elem_classes="submit-btn")
            opt_btn.click(
                _make_handler(_opt_reflector),
                inputs=list(comps_opt.values()),
                outputs=opt_output,
            )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")
