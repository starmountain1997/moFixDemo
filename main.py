import gradio as gr
import os
from modelscope.hub.api import HubApi
from modelscope.hub.snapshot_download import snapshot_download

def ensure_model_available(model_id):
    if not model_id or not model_id.strip():
        return None, "模型 ID 不能为空。", False
    
    api = HubApi()
    try:
        api.get_model(model_id)
    except Exception:
        return None, f"在 ModelScope 上未找到模型 '{model_id}'。", False

    try:
        downloaded_path = snapshot_download(model_id)
        return downloaded_path, "模型准备就绪", True
    except Exception as e:
        return None, f"自动化下载失败：{str(e)}", False

import subprocess

def generate_scene_description(mode, model_id, device, quantize, num_devices, input_len, output_len, ttft_limit, tpot_limit, progress=gr.Progress()):
    # 1. 必选参数校验
    if not model_id or not model_id.strip():
        return "## ❌ 错误\n**ModelScope 模型 ID** 是必选参数，请填写后再试。"

    # 2. 自动化下载/检查
    progress(0.1, desc="正在同步模型权重...")
    local_path, status_msg, success = ensure_model_available(model_id)
    if not success:
        return f"## ❌ 错误\n{status_msg}"
    
    progress(0.4, desc="正在分析参数并生成命令...")
    
    # 3. 构建命令行
    q_val = quantize.split(" ")[0] if quantize else "DISABLED"
    
    if mode == "吞吐量优化":
        cmd = ["python", "-m", "cli.inference.throughput_optimizer", model_id, "--remote-source", "modelscope"]
        if device and device != "无 (None)":
            cmd.extend(["--device", device])
        if num_devices is not None and num_devices > 0:
            cmd.extend(["--num-devices", str(int(num_devices))])
        if input_len is not None:
            cmd.extend(["--input-length", str(int(input_len))])
        if output_len is not None:
            cmd.extend(["--output-length", str(int(output_len))])
        if ttft_limit is not None:
            cmd.extend(["--ttft-limits", str(int(ttft_limit))])
        if tpot_limit is not None:
            cmd.extend(["--tpot-limits", str(int(tpot_limit))])
        if q_val != "DISABLED":
            cmd.extend(["--quantize-linear-action", q_val])
    else: # 性能仿真
        cmd = ["python", "-m", "cli.inference.text_generate", model_id, "--remote-source", "modelscope"]
        if device and device != "无 (None)":
            cmd.extend(["--device", device])
        if input_len is not None:
            cmd.extend(["--query-length", str(int(input_len))])
        cmd.extend(["--num-queries", "1"])
        if q_val != "DISABLED":
            cmd.extend(["--quantize-linear-action", q_val])

    progress(0.6, desc="正在执行 msmodeling...")
    
    try:
        # 运行命令并捕获输出
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        
        output_text = f"### 执行结果\n\n"
        output_text += f"**执行命令:** `{' '.join(cmd)}`\n\n"
        
        if result.stdout:
            output_text += "#### 标准输出 (stdout):\n```text\n" + result.stdout + "\n```\n"
        
        if result.stderr:
            output_text += "#### 标准错误 (stderr):\n```text\n" + result.stderr + "\n```\n"
            
        if result.returncode != 0:
            output_text += f"\n> ⚠️ **命令执行失败，退出码: {result.returncode}**"
            
        return output_text

    except Exception as e:
        return f"## ❌ 执行错误\n无法运行命令：{str(e)}"


devices = [
    "无 (None)", "TEST_DEVICE", "ATLAS_800_A2_376T_64G", "ATLAS_800_A2_313T_64G", 
    "ATLAS_800_A2_280T_64G", "ATLAS_800_A2_280T_64G_PCIE", 
    "ATLAS_800_A2_280T_32G_PCIE", "ATLAS_800_A3_752T_128G_DIE", 
    "ATLAS_800_A3_560T_128G_DIE"
]

quant_options = [
    "DISABLED (禁用)", "W8A16_STATIC", "W8A8_STATIC", "W4A8_STATIC", 
    "W8A16_DYNAMIC", "W8A8_DYNAMIC", "W4A8_DYNAMIC", "FP8", "MXFP4"
]

with gr.Blocks(title="MindStudio Modeling 场景生成器") as demo:
    gr.Markdown("# MindStudio Modeling 自动化配置生成")
    gr.Markdown("所有数值参数默认为空（不设置）。**模型 ID 为必填项**。")
    
    with gr.Row():
        mode = gr.Radio(["吞吐量优化", "文本生成 (性能仿真)"], label="运行模式", value="吞吐量优化")
        device = gr.Dropdown(devices, label="设备型号", value="无 (None)")
        num_devices = gr.Number(label="设备数量", value=None, precision=0)

    with gr.Row():
        quantize = gr.Dropdown(quant_options, label="量化方式", value="DISABLED (禁用)")
        model_id = gr.Textbox(label="ModelScope 模型 ID (必填)", placeholder="例如：qwen/Qwen2.5-7B-Instruct")

    with gr.Row():
        input_len = gr.Number(label="输入长度 (Tokens)", value=None, precision=0)
        output_len = gr.Number(label="预期输出长度 (Tokens)", value=None, precision=0)

    with gr.Group() as slo_group:
        gr.Markdown("### 性能限制 (SLO) - 留空表示不限制")
        with gr.Row():
            ttft_limit = gr.Number(label="TTFT 限制 (ms)", value=None, precision=0)
            tpot_limit = gr.Number(label="TPOT 限制 (ms)", value=None, precision=0)

    output = gr.Markdown(label="生成的配置与命令")
    submit_btn = gr.Button("生成配置并同步权重", variant="primary")

    def toggle_slo(m):
        return gr.update(visible=(m == "吞吐量优化"))

    mode.change(toggle_slo, mode, slo_group)

    submit_btn.click(
        generate_scene_description,
        inputs=[
            mode, model_id, device, quantize, num_devices, 
            input_len, output_len, ttft_limit, tpot_limit
        ],
        outputs=output
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")
