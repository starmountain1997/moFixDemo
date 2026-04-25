import os
import subprocess
from typing import cast

import gradio as gr

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


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
    normalized_id = model_id.lower().replace("-", "_")
    for _category, keywords in MODEL_KEYWORDS.items():
        for kw in keywords:
            if kw.lower() in normalized_id:
                return True
    return False


def execute_inference_simulation(
    model_id: str,
    device: str | None,
    num_queries: int | float | None,
    query_length: int | float | None,
    context_length: int | float | None,
    num_devices: int | float | None,
    compile: str | None,
    quantize_linear_action: str | None,
    quantize_attention_action: str | None,
    progress: gr.Progress | None = None,
) -> str:
    if not model_id or not model_id.strip():
        return "## ❌ 错误\n**ModelScope 模型 ID** 是必选参数，请填写后再试。"

    if not validate_model_id(model_id):
        supported_categories = ", ".join(MODEL_KEYWORDS.keys())
        return (
            f"## ❌ 预期不支持\n模型 ID `{model_id}` 可能不属于 `msmodeling` 目前支持的架构系列。\n\n"
            f"**当前支持的主流系列包括:**\n{supported_categories}\n\n"
            "*如果您确认该模型架构兼容，请联系开发人员或直接尝试运行。*"
        )

    if progress:
        progress(0.2, desc="正在分析参数并生成命令...")

    q_val = (
        quantize_linear_action.split(" ")[0] if quantize_linear_action else "DISABLED"
    )

    cmd: list[str] = [
        "python",
        "-m",
        "cli.inference.text_generate",
        cast(str, model_id),
    ]
    if device and device != "无 (None)":
        cmd.extend(["--device", device])
    if num_devices is not None and num_devices > 0:
        cmd.extend(["--num-devices", str(int(num_devices))])
    if query_length is not None and query_length > 0:
        cmd.extend(["--query-length", str(int(query_length))])
    if num_queries is not None and num_queries > 0:
        cmd.extend(["--num-queries", str(int(num_queries))])
    if context_length is not None and context_length > 0:
        cmd.extend(["--context-length", str(int(context_length))])
    if compile == "true":
        cmd.append("--compile")
    if q_val != "DISABLED":
        cmd.extend(["--quantize-linear-action", q_val])
    if quantize_attention_action and quantize_attention_action != "DISABLED":
        att_val = quantize_attention_action.split(" ")[0]
        cmd.extend(["--quantize-attention-action", att_val])

    if progress:
        progress(0.8, desc="正在执行 msmodeling...")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)

        output_text = "### 执行结果\n\n"
        output_text += f"**执行命令:** `{' '.join(cmd)}`\n\n"

        if result.stdout:
            output_text += (
                "#### 标准输出 (stdout):\n```text\n" + result.stdout + "\n```\n"
            )

        if result.stderr:
            output_text += (
                "#### 标准错误 (stderr):\n```text\n" + result.stderr + "\n```\n"
            )

        if result.returncode != 0:
            output_text += f"\n> ⚠️ **命令执行失败，退出码: {result.returncode}**"

        return output_text

    except Exception as e:
        return f"## ❌ 执行错误\n无法运行命令：{str(e)}"


def execute_parameter_optimization(
    model_id: str,
    device: str | None,
    input_length: int | float | None,
    output_length: int | float | None,
    num_devices: int | float | None,
    tpot_limits: int | float | None,
    ttft_limits: int | float | None,
    max_prefill_tokens: int | float | None,
    compile: str | None,
    quantize_linear_action: str | None,
    progress: gr.Progress | None = None,
) -> str:
    if not model_id or not model_id.strip():
        return "## ❌ 错误\n**ModelScope 模型 ID** 是必选参数，请填写后再试。"

    if not validate_model_id(model_id):
        supported_categories = ", ".join(MODEL_KEYWORDS.keys())
        return (
            f"## ❌ 预期不支持\n模型 ID `{model_id}` 可能不属于 `msmodeling` 目前支持的架构系列。\n\n"
            f"**当前支持的主流系列包括:**\n{supported_categories}\n\n"
            "*如果您确认该模型架构兼容，请联系开发人员或直接尝试运行。*"
        )

    if (
        input_length is None
        or input_length <= 0
        or output_length is None
        or output_length <= 0
    ):
        return "## ❌ 错误\n**输入长度** 和 **预期输出长度** 是必选参数，且必须大于 0。"

    if progress:
        progress(0.2, desc="正在分析参数并生成命令...")

    q_val = (
        quantize_linear_action.split(" ")[0] if quantize_linear_action else "DISABLED"
    )

    cmd: list[str] = [
        "python",
        "-m",
        "cli.inference.throughput_optimizer",
        cast(str, model_id),
    ]
    if device and device != "无 (None)":
        cmd.extend(["--device", device])
    if num_devices is not None and num_devices > 0:
        cmd.extend(["--num-devices", str(int(num_devices))])
    cmd.extend(["--input-length", str(int(input_length))])
    cmd.extend(["--output-length", str(int(output_length))])
    if ttft_limits is not None and ttft_limits > 0:
        cmd.extend(["--ttft-limits", str(int(ttft_limits))])
    if tpot_limits is not None and tpot_limits > 0:
        cmd.extend(["--tpot-limits", str(int(tpot_limits))])
    if max_prefill_tokens is not None and max_prefill_tokens > 0:
        cmd.extend(["--max-prefill-tokens", str(int(max_prefill_tokens))])
    if compile == "true":
        cmd.append("--compile")
    if q_val != "DISABLED":
        cmd.extend(["--quantize-linear-action", q_val])

    if progress:
        progress(0.8, desc="正在执行 msmodeling...")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)

        output_text = "### 执行结果\n\n"
        output_text += f"**执行命令:** `{' '.join(cmd)}`\n\n"

        if result.stdout:
            output_text += (
                "#### 标准输出 (stdout):\n```text\n" + result.stdout + "\n```\n"
            )

        if result.stderr:
            output_text += (
                "#### 标准错误 (stderr):\n```text\n" + result.stderr + "\n```\n"
            )

        if result.returncode != 0:
            output_text += f"\n> ⚠️ **命令执行失败，退出码: {result.returncode}**"

        return output_text

    except Exception as e:
        return f"## ❌ 执行错误\n无法运行命令：{str(e)}"


devices = [
    "无 (None)",
    "TEST_DEVICE",
    "ATLAS_800_A2_376T_64G",
    "ATLAS_800_A2_313T_64G",
    "ATLAS_800_A2_280T_64G",
    "ATLAS_800_A2_280T_64G_PCIE",
    "ATLAS_800_A2_280T_32G_PCIE",
    "ATLAS_800_A3_752T_128G_DIE",
    "ATLAS_800_A3_560T_128G_DIE",
]

quant_options = [
    "DISABLED (禁用)",
    "W8A16_STATIC",
    "W8A8_STATIC",
    "W4A8_STATIC",
    "W8A16_DYNAMIC",
    "W8A8_DYNAMIC",
    "W4A8_DYNAMIC",
    "FP8",
    "MXFP4",
]

attention_quant_options = [
    "DISABLED (禁用)",
    "INT8",
    "FP8",
]

compile_options = [
    "无 (None)",
    "true",
    "false",
]

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
.parameter-group {
    background: #fafafa;
    border: 1px solid #e8e8e8;
    border-radius: 10px;
    padding: 0;
    margin-bottom: 16px;
    overflow: hidden;
}
.parameter-group > .header {
    background: #fff;
    padding: 14px 20px;
    border-bottom: 1px solid #e8e8e8;
    font-weight: 600;
    font-size: 14px;
    color: #333;
    cursor: pointer;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.parameter-group > .header:hover {
    background: #f5f5f5;
}
.parameter-group > .content {
    padding: 20px;
}
.section-divider {
    height: 1px;
    background: #e8e8e8;
    margin: 16px 0;
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
.result-box {
    background: #fafafa;
    border: 1px solid #e8e8e8;
    border-radius: 8px;
    padding: 20px;
    margin-top: 20px;
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

            with gr.Accordion("⚙️ 核心参数", open=True):
                with gr.Group():
                    model_id_sim = gr.Textbox(
                        label="模型ID (--model) *",
                        placeholder="例如：Qwen/Qwen3-32B",
                        info="HuggingFace 模型 ID（如 Qwen/Qwen3-32B），不支持本地路径",
                    )
                    gr.HTML('<div class="section-divider"></div>')
                    with gr.Row():
                        num_queries_sim = gr.Number(
                            label="并发数 (--num-queries)",
                            value=2,
                            precision=0,
                            info="并发数，影响吞吐量",
                        )
                        query_length_sim = gr.Number(
                            label="输入长度 (--query-length)",
                            value=3500,
                            precision=0,
                            info="提示的输入长度，decode一次就设为1",
                        )
                    with gr.Row():
                        context_length_sim = gr.Number(
                            label="上下文长度 (--context-length)",
                            value=3000,
                            precision=0,
                            info="应该是(输入+输出/2)，不指定就是prefill",
                        )
                        device_sim = gr.Dropdown(
                            devices,
                            label="设备类型 (--device)",
                            value="TEST_DEVICE",
                            info="选择运行硬件型号",
                        )
                    num_devices_sim = gr.Number(
                        label="部署卡数 (--num-devices)",
                        value=1,
                        precision=0,
                        info="使用的设备数量",
                    )

            with gr.Accordion("🔧 编译优化参数", open=False):
                with gr.Group():
                    with gr.Row():
                        compile_sim = gr.Dropdown(
                            compile_options,
                            label="图编译 (--compile)",
                            value="false",
                            info="启用后性能会更好",
                        )
                        compile_allow_graph_break_sim = gr.Dropdown(
                            ["false", "true"],
                            label="允许图中断 (--compile-allow-graph-break)",
                            value="false",
                            info="允许在torch.compile()中出现图断点",
                        )

            with gr.Accordion("📊 量化参数", open=False):
                with gr.Group():
                    with gr.Row():
                        quantize_linear_sim = gr.Dropdown(
                            quant_options,
                            label="线性层量化 (--quantize-linear-action)",
                            value="DISABLED (禁用)",
                            info="线性层量化方式",
                        )
                        quantize_attention_sim = gr.Dropdown(
                            attention_quant_options,
                            label="Attention量化 (--quantize-attention-action)",
                            value="DISABLED (禁用)",
                            info="KV缓存量化方式",
                        )

            with gr.Accordion("🔀 并行策略参数", open=False):
                with gr.Group():
                    with gr.Row():
                        tp_size_sim = gr.Number(
                            label="TP大小 (--tp-size)",
                            value=1,
                            precision=0,
                            info="张量并行大小",
                        )
                        dp_size_sim = gr.Number(
                            label="DP大小 (--dp-size)",
                            value=1,
                            precision=0,
                            info="数据并行大小",
                        )
                    with gr.Row():
                        ep_sim = gr.Dropdown(
                            ["false", "true"],
                            label="专家并行 (--ep)",
                            value="false",
                            info="启用专家并行",
                        )
                        word_embedding_tp_sim = gr.Dropdown(
                            ["false", "true"],
                            label="词嵌入TP (--word-embedding-tp)",
                            value="false",
                            info="是否开启embedding的tp",
                        )

            with gr.Accordion("🐛 调试参数", open=False):
                with gr.Group():
                    with gr.Row():
                        dump_input_shapes_sim = gr.Dropdown(
                            ["false", "true"],
                            label="基于输入shape分组 (--dump-input-shapes)",
                            value="false",
                            info="基于输入shape给输出表格分组",
                        )
                        chrome_trace_sim = gr.Dropdown(
                            ["false", "true"],
                            label="生成trace (--chrome-trace)",
                            value="false",
                            info="方便看各算子具体运行情况",
                        )
                    with gr.Row():
                        graph_log_url_sim = gr.Textbox(
                            label="新图输出路径 (--graph-log-url)",
                            value="./graph.log",
                            info="调测用",
                        )
                        num_hidden_layers_override_sim = gr.Number(
                            label="覆盖模型层数 (--num-hidden-layers-override)",
                            value=0,
                            precision=0,
                            info="调测用，0表示使用模型默认值",
                        )

            gr.HTML("""
            <div style="background: #f6ffed; border: 1px solid #b7eb8f; border-radius: 6px; padding: 14px 16px; margin: 16px 0;">
                <h4 style="font-size: 13px; font-weight: 600; margin-bottom: 8px; color: #389e0d;">💡 智能推荐</h4>
                <p style="font-size: 12px; color: #52c41a; margin-bottom: 4px;">• 并发数：2（适合测试基础性能）</p>
                <p style="font-size: 12px; color: #52c41a; margin-bottom: 4px;">• 输入长度：3500（标准测试长度）</p>
                <p style="font-size: 12px; color: #52c41a;">• 图编译：启用（可提升性能）</p>
            </div>
            """)

            sim_output = gr.Markdown(label="执行结果")
            sim_submit_btn = gr.Button("▶️ 执行仿真", variant="primary", elem_classes="submit-btn")

            sim_submit_btn.click(
                execute_inference_simulation,
                inputs=[
                    model_id_sim,
                    device_sim,
                    num_queries_sim,
                    query_length_sim,
                    context_length_sim,
                    num_devices_sim,
                    compile_sim,
                    quantize_linear_sim,
                    quantize_attention_sim,
                ],
                outputs=sim_output,
            )

        with gr.TabItem("🎨 参数寻优"):
            gr.HTML("""
            <div class="feature-box">
                <h4>参数寻优</h4>
                <p>支持三种性能优化模式：</p>
                <ul>
                    <li><strong>PD混部</strong>：Prefill和Decode联合优化，搜索整体吞吐最大的配置</li>
                    <li><strong>分离模式</strong>：Prefill和Decode分离建模，分别评估各自性能</li>
                    <li><strong>PD比例优化</strong>：寻找最优的Prefill:Decode实例配比</li>
                </ul>
                <p><strong>主要用途：</strong>寻找最佳批次大小、优化模型配置、评估量化策略影响、分析PD分离部署性能</p>
            </div>
            """)

            optimization_mode = gr.Dropdown(
                ["aggregation", "disaggregation", "pd-ratio"],
                label="寻优模式",
                value="aggregation",
                info="选择优化模式",
            )

            with gr.Accordion("⚙️ 核心参数", open=True):
                with gr.Group():
                    model_id_opt = gr.Textbox(
                        label="模型ID (--model) *",
                        placeholder="例如：Qwen/Qwen3-32B",
                        info="HuggingFace 模型 ID（如 Qwen/Qwen3-32B），不支持本地路径",
                    )
                    gr.HTML('<div class="section-divider"></div>')
                    with gr.Row():
                        input_length_opt = gr.Number(
                            label="输入长度 (--input-length) *",
                            value=3500,
                            precision=0,
                            info="提示的输入长度 (必需)",
                        )
                        output_length_opt = gr.Number(
                            label="输出长度 (--output-length) *",
                            value=1500,
                            precision=0,
                            info="预期输出长度 (必需)",
                        )
                    with gr.Row():
                        device_opt = gr.Dropdown(
                            devices,
                            label="设备类型 (--device)",
                            value="TEST_DEVICE",
                            info="目标硬件型号",
                        )
                        num_devices_opt = gr.Number(
                            label="部署卡数 (--num-devices)",
                            value=8,
                            precision=0,
                            info="使用的设备数量",
                        )

            with gr.Accordion("🔧 编译优化参数", open=False):
                with gr.Group():
                    with gr.Row():
                        compile_opt = gr.Dropdown(
                            compile_options,
                            label="图编译 (--compile)",
                            value="false",
                            info="对模型启用 torch.compile()",
                        )
                        compile_allow_graph_break_opt = gr.Dropdown(
                            ["false", "true"],
                            label="允许图断点 (--compile-allow-graph-break)",
                            value="false",
                            info="允许在 torch.compile() 中出现图断点",
                        )

            with gr.Accordion("📊 量化参数", open=False):
                with gr.Group():
                    with gr.Row():
                        quantize_linear_opt = gr.Dropdown(
                            quant_options,
                            label="线性层量化方式 (--quantize-linear-action)",
                            value="DISABLED (禁用)",
                            info="线性层量化方式",
                        )
                        mxfp4_group_size_opt = gr.Number(
                            label="MXFP4量化分组大小 (--mxfp4-group-size)",
                            value=32,
                            precision=0,
                            info="MXFP4 量化分组大小",
                        )
                    quantize_attention_opt = gr.Dropdown(
                        attention_quant_options,
                        label="KV缓存量化方式 (--quantize-attention-action)",
                        value="DISABLED (禁用)",
                        info="KV 缓存量化方式",
                    )

            with gr.Accordion("🔮 MTP与Prefix Cache参数", open=False):
                with gr.Group():
                    with gr.Row():
                        num_mtp_tokens_opt = gr.Number(
                            label="MTP令牌数量 (--num-mtp-tokens)",
                            value=0,
                            precision=0,
                            info="MTP 令牌数量",
                        )
                        mtp_acceptance_rate_opt = gr.Textbox(
                            label="MTP接受率 (--mtp-acceptance-rate)",
                            value="[0.9, 0.6, 0.4, 0.2]",
                            info="MTP 接受率列表",
                        )

            with gr.Accordion("📦 Aggregation mode 参数 (PD混部)", open=False):
                with gr.Group():
                    with gr.Row():
                        ttft_limits_opt = gr.Number(
                            label="TTFT约束 (--ttft-limits)",
                            value=7000,
                            precision=0,
                            info="Time To First Token 时间约束",
                        )
                        tpot_limits_opt = gr.Number(
                            label="TPOT约束 (--tpot-limits)",
                            value=50,
                            precision=0,
                            info="Token Per Output Token 时间约束",
                        )
                    max_prefill_tokens_opt = gr.Number(
                        label="最大预填充令牌数 (--max-prefill-tokens)",
                        value=8192,
                        precision=0,
                        info="最大预填充令牌数",
                    )

            with gr.Accordion("📦 Disaggregation mode 参数 (分离模式)", open=False, visible=False):
                with gr.Group():
                    with gr.Row():
                        ttft_limits_disagg = gr.Number(
                            label="TTFT约束 (--ttft-limits)",
                            value=7000,
                            precision=0,
                            info="Prefill模式下的时间约束",
                        )
                        tpot_limits_disagg = gr.Number(
                            label="TPOT约束 (--tpot-limits)",
                            value=50,
                            precision=0,
                            info="Decode模式下的时间约束",
                        )
                    with gr.Row():
                        serving_cost_opt = gr.Number(
                            label="服务成本 (--serving-cost)",
                            value=0,
                            precision=0,
                            info="服务交付成本，参与优化结果评估",
                        )
                        disagg_opt = gr.Dropdown(
                            ["false", "true"],
                            label="启用PD分离模式 (--disagg)",
                            value="false",
                            info="启用PD分离模式",
                        )

            with gr.Accordion("📦 PD Ratio Optimization mode 参数", open=False, visible=False):
                with gr.Group():
                    with gr.Row():
                        prefill_devices_per_instance = gr.Number(
                            label="Prefill实例设备数 (--prefill-devices-per-instance)",
                            value=4,
                            precision=0,
                            info="每个Prefill实例的设备数",
                        )
                        decode_devices_per_instance = gr.Number(
                            label="Decode实例设备数 (--decode-devices-per-instance)",
                            value=4,
                            precision=0,
                            info="每个Decode实例的设备数",
                        )

            with gr.Accordion("📋 可选参数", open=False):
                with gr.Group():
                    with gr.Row():
                        batch_range_opt = gr.Textbox(
                            label="批次大小范围 (--batch-range)",
                            value="[1, 64]",
                            info="批次大小范围 [min, max]",
                        )
                        tp_sizes_opt = gr.Textbox(
                            label="TP大小列表 (--tp-sizes)",
                            value="[1, 2, 4, 8]",
                            info="搜索的 TP 大小列表",
                        )
                    with gr.Row():
                        jobs_opt = gr.Number(
                            label="并行作业数 (--jobs)",
                            value=8,
                            precision=0,
                            info="并行作业数",
                        )
                        dump_original_results_opt = gr.Dropdown(
                            ["false", "true"],
                            label="导出原始结果 (--dump-original-results)",
                            value="false",
                            info="导出原始搜索结果，便于分析",
                        )

            with gr.Accordion("📝 日志参数", open=False):
                with gr.Group():
                    log_level_opt = gr.Dropdown(
                        ["debug", "info", "warning", "error", "critical"],
                        label="日志级别 (--log-level)",
                        value="error",
                        info="日志级别 (默认: error)",
                    )

            gr.HTML("""
            <div style="background: #f6ffed; border: 1px solid #b7eb8f; border-radius: 6px; padding: 14px 16px; margin: 16px 0;">
                <h4 style="font-size: 13px; font-weight: 600; margin-bottom: 8px; color: #389e0d;">💡 智能推荐</h4>
                <p style="font-size: 12px; color: #52c41a; margin-bottom: 4px;">• 输入长度：3500，输出长度：1500（标准测试长度）</p>
                <p style="font-size: 12px; color: #52c41a; margin-bottom: 4px;">• TPOT约束：50ms，TTFT约束：7000ms（合理的性能约束）</p>
                <p style="font-size: 12px; color: #52c41a;">• 最大预填充令牌数：8192（标准设置）</p>
            </div>
            """)

            opt_output = gr.Markdown(label="执行结果")
            opt_submit_btn = gr.Button("▶️ 执行寻优", variant="primary", elem_classes="submit-btn")

            opt_submit_btn.click(
                execute_parameter_optimization,
                inputs=[
                    model_id_opt,
                    device_opt,
                    input_length_opt,
                    output_length_opt,
                    num_devices_opt,
                    tpot_limits_opt,
                    ttft_limits_opt,
                    max_prefill_tokens_opt,
                    compile_opt,
                    quantize_linear_opt,
                ],
                outputs=opt_output,
            )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")
