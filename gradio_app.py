import gradio as gr
import os
from modelscope.hub.api import HubApi
from modelscope.hub.snapshot_download import snapshot_download

# 模型权重基础目录
MODEL_WEIGHTS_BASE = "/home/model_weights"

def ensure_model_available(model_id):
    if not model_id or not model_id.strip():
        return None, "模型 ID 不能为空。", False
    
    if not os.path.exists(MODEL_WEIGHTS_BASE):
        try:
            os.makedirs(MODEL_WEIGHTS_BASE, exist_ok=True)
        except PermissionError:
            return None, f"权限拒绝：无法创建目录 '{MODEL_WEIGHTS_BASE}'", False

    api = HubApi()
    try:
        api.get_model(model_id)
    except Exception:
        return None, f"在 ModelScope 上未找到模型 '{model_id}'。", False

    try:
        downloaded_path = snapshot_download(model_id, cache_dir=MODEL_WEIGHTS_BASE)
        return downloaded_path, "模型准备就绪", True
    except Exception as e:
        return None, f"自动化下载失败：{str(e)}", False

def generate_scene_description(mode, model_id, device, quantize, num_devices, input_len, output_len, ttft_limit, tpot_limit, progress=gr.Progress()):
    # 1. 必选参数校验
    if not model_id or not model_id.strip():
        return "## ❌ 错误\n**ModelScope 模型 ID** 是必选参数，请填写后再试。"

    # 2. 自动化下载/检查
    progress(0.1, desc="正在同步模型权重...")
    local_path, status_msg, success = ensure_model_available(model_id)
    if not success:
        return f"## ❌ 错误\n{status_msg}"
    
    progress(0.8, desc="正在分析参数...")
    
    # 3. 构建描述信息
    description = f"### 推理场景配置分析\n\n"
    description += f"**模型 ID:** {model_id}\n"
    if local_path:
        description += f"**本地路径:** `{local_path}`\n"
    
    if device and device != "无 (None)":
        dev_str = f"{int(num_devices)}x " if num_devices is not None else ""
        description += f"**目标硬件:** {dev_str}{device}\n"
        
    if quantize and quantize != "DISABLED (禁用)":
        description += f"**量化方式:** {quantize}\n"
    
    description += "\n"

    # 4. 构建命令行
    q_val = quantize.split(" ")[0] if quantize else "DISABLED"
    
    if mode == "吞吐量优化":
        description += "#### 性能目标与约束 (SLO):\n"
        ttft_str = f"< {int(ttft_limit)} ms" if ttft_limit is not None else "未设置 (尽力而为)"
        tpot_str = f"< {int(tpot_limit)} ms" if tpot_limit is not None else "未设置 (尽力而为)"
        description += f"- **TTFT 限制 (首字延迟):** {ttft_str}\n"
        description += f"- **TPOT 限制 (字间延迟):** {tpot_str}\n"
            
        if input_len is not None or output_len is not None:
            description += f"\n#### 业务负载详情:\n"
            if input_len is not None: description += f"- **输入长度:** {int(input_len)} tokens\n"
            if output_len is not None: description += f"- **输出长度:** {int(output_len)} tokens\n"
        
        # 构建命令
        cmd = f"python -m cli.inference.throughput_optimizer {model_id} --remote-source modelscope"
        if device and device != "无 (None)": cmd += f" --device {device}"
        if num_devices is not None: cmd += f" --num-devices {int(num_devices)}"
        if input_len is not None: cmd += f" --input-length {int(input_len)}"
        if output_len is not None: cmd += f" --output-length {int(output_len)}"
        if ttft_limit is not None: cmd += f" --ttft-limits {int(ttft_limit)}"
        if tpot_limit is not None: cmd += f" --tpot-limits {int(tpot_limit)}"
        if q_val != "DISABLED": cmd += f" --quantize-linear-action {q_val}"
        
        description += f"\n---\n**优化器启动命令:**\n`{cmd}`"
        
    else: # 性能仿真
        if input_len is not None:
            description += "#### 业务负载详情:\n"
            description += f"- **输入长度:** {int(input_len)} tokens\n"
        
        cmd = f"python -m cli.inference.text_generate {model_id} --remote-source modelscope"
        if device and device != "无 (None)": cmd += f" --device {device}"
        if input_len is not None: cmd += f" --query-length {int(input_len)}"
        cmd += " --num-queries 1"
        if q_val != "DISABLED": cmd += f" --quantize-linear-action {q_val}"
        
        description += f"\n---\n**性能仿真命令:**\n`{cmd}`"

    return description

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
