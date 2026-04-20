import subprocess
from typing import cast

import gradio as gr
from atomgit_hub import snapshot_download


def ensure_model_available(model_id: str) -> tuple[str | None, str, bool]:
    if not model_id or not model_id.strip():
        return None, "模型 ID 不能为空。", False

    try:
        download_path = snapshot_download(f"{model_id}")
        return download_path, "模型准备就绪", True
    except Exception as e:
        return None, f"自动化下载失败：{str(e)}", False


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
    # Validation
    if not model_id or not model_id.strip():
        return "## ❌ 错误\n**ModelScope 模型 ID** 是必选参数，请填写后再试。"

    if not validate_model_id(model_id):
        supported_categories = ", ".join(MODEL_KEYWORDS.keys())
        return (
            f"## ❌ 预期不支持\n模型 ID `{model_id}` 可能不属于 `msmodeling` 目前支持的架构系列。\n\n"
            f"**当前支持的主流系列包括:**\n{supported_categories}\n\n"
            "*如果您确认该模型架构兼容，请联系开发人员或直接尝试运行。*"
        )

    # Download model
    if progress:
        progress(0.1, desc="正在同步模型权重...")
    local_path, status_msg, success = ensure_model_available(model_id)
    if not success:
        return f"## ❌ 错误\n{status_msg}"

    if progress:
        progress(0.4, desc="正在分析参数并生成命令...")

    # Build command
    q_val = (
        quantize_linear_action.split(" ")[0] if quantize_linear_action else "DISABLED"
    )
    model_path = local_path

    cmd: list[str] = [
        "python",
        "-m",
        "cli.inference.text_generate",
        cast(str, model_path),
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
    if compile and compile != "无 (None)":
        cmd.extend(["--compile", compile])
    if q_val != "DISABLED":
        cmd.extend(["--quantize-linear-action", q_val])
    if quantize_attention_action and quantize_attention_action != "DISABLED":
        att_val = quantize_attention_action.split(" ")[0]
        cmd.extend(["--quantize-attention-action", att_val])

    if progress:
        progress(0.6, desc="正在执行 msmodeling...")

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
    # Validation
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

    # Download model
    if progress:
        progress(0.1, desc="正在同步模型权重...")
    local_path, status_msg, success = ensure_model_available(model_id)
    if not success:
        return f"## ❌ 错误\n{status_msg}"

    if progress:
        progress(0.4, desc="正在分析参数并生成命令...")

    q_val = (
        quantize_linear_action.split(" ")[0] if quantize_linear_action else "DISABLED"
    )
    model_path = local_path

    cmd: list[str] = [
        "python",
        "-m",
        "cli.inference.throughput_optimizer",
        cast(str, model_path),
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
    if compile and compile != "无 (None)":
        cmd.extend(["--compile", compile])
    if q_val != "DISABLED":
        cmd.extend(["--quantize-linear-action", q_val])

    if progress:
        progress(0.6, desc="正在执行 msmodeling...")

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
    "ATLAS_800_A2_376T_64G",
    "ATLAS_800_A2_313T_64G",
    "ATLAS_800_A2_280T_64G",
    "ATLAS_800_A2_280T_64G_PCIE",
    "ATLAS_800_A2_280T_32G_PCIE",
    "ATLAS_800_A3_752T_128G_DIE",
    "ATLAS_800_A3_560T_128G_DIE",
    "TEST_DEVICE",
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

theme = gr.themes.Default(
    primary_hue="blue",
    secondary_hue="gray",
    neutral_hue="gray",
).set(
    body_background_fill="*neutral_950",
    body_background_fill_dark="*neutral_950",
)  # type: ignore[reportPrivateImportUsage]

with gr.Blocks(title="msModeling 推理仿真与参数寻优工具", theme=theme) as demo:
    gr.Markdown("# msModeling 推理仿真与参数寻优工具")
    gr.Markdown(
        "基于 Ascend 平台的 AI 模型性能评估与优化工具，支持推理仿真和参数寻优两大核心功能"
    )

    with gr.Tabs():
        # ==================== 推理仿真 Tab ====================
        with gr.TabItem("推理仿真"):
            gr.Markdown("### 功能介绍")
            gr.Markdown(
                "用于评估模型在不同配置下的推理性能，包括并发测试和单次forward时间测量。\n\n"
                "**主要用途：** 测试不同并发数下的模型性能 | 评估单次forward时间 | "
                "分析不同硬件设备的表现 | 验证量化对性能的影响"
            )

            gr.Markdown("### 基本参数")

            with gr.Row():
                model_id_sim = gr.Textbox(
                    label="模型名称/路径 (--model) *",
                    placeholder="例如：Qwen3-32B 或本地模型路径",
                    scale=2,
                )

            with gr.Row():
                device_sim = gr.Dropdown(
                    devices,
                    label="硬件型号 (--device)",
                    value="无 (None)",
                )
                quantize_attention_sim = gr.Dropdown(
                    attention_quant_options,
                    label="Attention量化 (--quantize-attention-action)",
                    value="DISABLED (禁用)",
                )

            with gr.Row():
                num_queries_sim = gr.Number(
                    label="并发数 (--num-queries)",
                    value=2,
                    precision=0,
                )
                query_length_sim = gr.Number(
                    label="输入长度 (--query-length)",
                    value=3500,
                    precision=0,
                )

            with gr.Row():
                context_length_sim = gr.Number(
                    label="上下文长度 (--context-length)",
                    value=3000,
                    precision=0,
                )
                num_devices_sim = gr.Number(
                    label="部署卡数 (--num-devices)",
                    value=1,
                    precision=0,
                )

            with gr.Row():
                compile_sim = gr.Dropdown(
                    compile_options,
                    label="图编译 (--compile)",
                    value="无 (None)",
                )
                quantize_linear_sim = gr.Dropdown(
                    quant_options,
                    label="线性层量化 (--quantize-linear-action)",
                    value="DISABLED (禁用)",
                )

            with gr.Accordion("更多参数", open=False):
                gr.Markdown("高级配置选项（可根据需要调整）")

            sim_output = gr.Markdown(label="执行结果")
            sim_submit_btn = gr.Button("执行仿真", variant="primary")

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

        # ==================== 参数寻优 Tab ====================
        with gr.TabItem("参数寻优"):
            gr.Markdown("### 功能介绍")
            gr.Markdown(
                "用于评估PD混部场景在给定SLO下的Top n吞吐配置。\n\n"
                "**主要用途：** 给定输入输出长度寻找最佳批次大小 | 在TPOT和TTFT约束下优化模型配置 | "
                "评估不同量化策略对性能的影响 | 找到最优的并行策略"
            )

            gr.Markdown("### 基本参数")

            with gr.Row():
                model_id_opt = gr.Textbox(
                    label="模型名称/路径 (--model) *",
                    placeholder="例如：Qwen3-32B 或本地模型路径",
                    scale=2,
                )
                device_opt = gr.Dropdown(
                    devices,
                    label="硬件型号 (--device)",
                    value="无 (None)",
                )

            with gr.Row():
                input_length_opt = gr.Number(
                    label="输入长度 (--input-length) *",
                    value=3500,
                    precision=0,
                )
                output_length_opt = gr.Number(
                    label="输出长度 (--output-length) *",
                    value=1500,
                    precision=0,
                )

            with gr.Row():
                num_devices_opt = gr.Number(
                    label="部署卡数 (--num-devices)",
                    value=8,
                    precision=0,
                )
                tpot_limits_opt = gr.Number(
                    label="TPOT约束 (--tpot-limits)",
                    value=50,
                    precision=0,
                )

            with gr.Row():
                ttft_limits_opt = gr.Number(
                    label="TTFT约束 (--ttft-limits)",
                    value=7000,
                    precision=0,
                )
                max_prefill_tokens_opt = gr.Number(
                    label="最大预填充令牌数 (--max-prefill-tokens)",
                    value=8192,
                    precision=0,
                )

            with gr.Accordion("更多参数", open=False):
                with gr.Row():
                    compile_opt = gr.Dropdown(
                        compile_options,
                        label="图编译 (--compile)",
                        value="无 (None)",
                    )
                    quantize_linear_opt = gr.Dropdown(
                        quant_options,
                        label="线性层量化 (--quantize-linear-action)",
                        value="DISABLED (禁用)",
                    )

            opt_output = gr.Markdown(label="执行结果")
            opt_submit_btn = gr.Button("执行寻优", variant="primary")

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
