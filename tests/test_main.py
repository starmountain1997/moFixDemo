import os
import sys
import subprocess
import itertools
from unittest.mock import MagicMock, patch

import pytest
import gradio as gr

# Ensure we can import main and cli modules
# The project structure is:
# /home/guozr/CODE/moFixDemo/
# ├── main.py
# ├── tests/test_main.py
# └── msmodeling/ (contains cli/ and tensor_cast/)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MSMODELING_DIR = os.path.abspath(os.path.join(PROJECT_ROOT, "msmodeling"))

sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, MSMODELING_DIR)

# Set PYTHONPATH for subprocesses
os.environ["PYTHONPATH"] = f"{MSMODELING_DIR}:{os.environ.get('PYTHONPATH', '')}"

import main

TEST_MODEL_ID = "Qwen/Qwen3-0.6B"
DUMMY_MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "qwen_dummy"))


@pytest.fixture(scope="session", autouse=True)
def setup_dummy_model():
    """Create a dummy model for real execution tests."""
    os.makedirs(DUMMY_MODEL_PATH, exist_ok=True)
    config_content = """{
  "architectures": ["Qwen2ForCausalLM"],
  "model_type": "qwen2",
  "hidden_size": 128,
  "intermediate_size": 256,
  "num_attention_heads": 4,
  "num_hidden_layers": 1,
  "num_key_value_heads": 2,
  "vocab_size": 1000
}"""
    with open(os.path.join(DUMMY_MODEL_PATH, "config.json"), "w") as f:
        f.write(config_content)
    yield
    # Cleanup after session
    import shutil
    if os.path.exists(DUMMY_MODEL_PATH):
        shutil.rmtree(DUMMY_MODEL_PATH)


class TestValidateModelId:
    """Test model ID validation."""

    @pytest.mark.parametrize("model_id,expected", [
        ("Qwen/Qwen3-0.6B", True),
        ("qwen3-0.6b", True),
        ("Qwen/Qwen3-32B", True),
        ("Qwen/Qwen2.5-72B", True),
        ("deepseek_v3", True),
        ("deepseek_v32", True),
        ("GLM/GLM-4V", True),
        ("InternVL/InternVL3", True),
        ("ERNIE/ERNIE-4", True),
        ("minimax/m2", True),
        ("MiniMax/M2.2", True),
        ("MIMO/mimo", True),
        ("wan/wan", True),
        ("hunyuan/hunyuan", True),
        ("invalid_model", False),
        ("", False),
        ("   ", False),
    ])
    def test_validate_model_id(self, model_id, expected):
        result = main.validate_model_id(model_id)
        assert result == expected


class TestEnsureModelAvailable:
    """Test model availability checking."""

    @patch("main.snapshot_download")
    def test_empty_model_id(self, mock_download):
        result = main.ensure_model_available("")
        assert result == (None, "模型 ID 不能为空。", False)

    @patch("main.snapshot_download")
    def test_whitespace_model_id(self, mock_download):
        result = main.ensure_model_available("   ")
        assert result == (None, "模型 ID 不能为空。", False)

    @patch("main.snapshot_download")
    def test_valid_model_download(self, mock_download):
        mock_download.return_value = "/path/to/model"
        result = main.ensure_model_available(TEST_MODEL_ID)
        assert result == ("/path/to/model", "模型准备就绪", True)
        mock_download.assert_called_once_with(TEST_MODEL_ID)

    @patch("main.snapshot_download")
    def test_download_failure(self, mock_download):
        mock_download.side_effect = Exception("Download failed")
        result = main.ensure_model_available(TEST_MODEL_ID)
        assert result == (None, "自动化下载失败：Download failed", False)


class TestCombinatorialInferenceExecution:
    """Test inference simulation by ACTUALLY running msmodeling for all combinations."""

    # Subset of options to keep test time reasonable but cover all types
    # Reducing a bit to avoid too many subprocesses in one test run, 
    # but still providing good coverage.
    DEVICES = ["TEST_DEVICE"]
    NUM_DEVICES = [1, 2]
    COMPILE_OPTS = ["false", "true"]
    QUANT_LINEAR = ["DISABLED (禁用)", "W8A16_STATIC", "FP8", "MXFP4"]
    QUANT_ATTN = ["DISABLED (禁用)", "INT8"]

    @pytest.mark.parametrize(
        "device, num_dev, compile, q_lin, q_attn",
        list(itertools.product(DEVICES, NUM_DEVICES, COMPILE_OPTS, QUANT_LINEAR, QUANT_ATTN))
    )
    @patch("main.snapshot_download")
    def test_inference_combinations_execution(self, mock_download, device, num_dev, compile, q_lin, q_attn):
        """This test actually calls the msmodeling CLI for each combination."""
        mock_download.return_value = DUMMY_MODEL_PATH
        
        # Use a model_id that passes validation
        model_id = "qwen-test"
        
        result = main.execute_inference_simulation(
            model_id=model_id,
            device=device,
            num_queries=1,
            query_length=1,
            context_length=0,
            num_devices=num_dev,
            compile=compile,
            quantize_linear_action=q_lin,
            quantize_attention_action=q_attn,
            progress=None
        )

        assert "### 执行结果" in result
        assert "cli.inference.text_generate" in result
        # If exit code is not 0, it means msmodeling failed for this combination
        assert "⚠️ 命令执行失败" not in result, f"Failed for {device}, {num_dev}, {compile}, {q_lin}, {q_attn}\nOutput: {result}"
        assert "标准输出 (stdout)" in result
        assert "TPS/Device" in result


class TestCombinatorialOptimizationExecution:
    """Test parameter optimization by ACTUALLY running msmodeling for all combinations."""

    DEVICES = ["TEST_DEVICE"]
    COMPILE_OPTS = ["false", "true"]
    QUANT_LINEAR = ["DISABLED (禁用)", "W4A8_DYNAMIC", "MXFP4"]

    @pytest.mark.parametrize(
        "device, compile, q_lin",
        list(itertools.product(DEVICES, COMPILE_OPTS, QUANT_LINEAR))
    )
    @patch("main.snapshot_download")
    def test_optimization_combinations_execution(self, mock_download, device, compile, q_lin):
        """This test actually calls the msmodeling CLI for each combination."""
        mock_download.return_value = DUMMY_MODEL_PATH

        result = main.execute_parameter_optimization(
            model_id="qwen-test",
            device=device,
            input_length=128,
            output_length=128,
            num_devices=1,
            tpot_limits=1000,
            ttft_limits=10000,
            max_prefill_tokens=256,
            compile=compile,
            quantize_linear_action=q_lin,
            progress=None
        )

        assert "### 执行结果" in result
        assert "cli.inference.throughput_optimizer" in result
        assert "⚠️ 命令执行失败" not in result, f"Failed for {device}, {compile}, {q_lin}\nOutput: {result}"
        assert "标准输出 (stdout)" in result


class TestErrorHandling:
    """Test error handling in execution functions."""

    @patch("main.ensure_model_available")
    def test_inference_missing_model_id(self, mock_ensure):
        result = main.execute_inference_simulation(
            model_id="", device="TEST_DEVICE", num_queries=1, query_length=1,
            context_length=0, num_devices=1, compile="false",
            quantize_linear_action="DISABLED", quantize_attention_action="DISABLED"
        )
        assert "❌ 错误" in result
        assert "模型 ID** 是必选参数" in result

    @patch("main.ensure_model_available")
    def test_optimization_invalid_lengths(self, mock_ensure):
        result = main.execute_parameter_optimization(
            model_id="qwen-test", device="TEST_DEVICE", input_length=0, output_length=128,
            num_devices=1, tpot_limits=50, ttft_limits=2000, max_prefill_tokens=1024,
            compile="false", quantize_linear_action="DISABLED"
        )
        assert "❌ 错误" in result
        assert "必须大于 0" in result


class TestDevicesAndOptions:
    """Test that devices and option lists are properly defined."""

    def test_devices_not_empty(self):
        assert len(main.devices) > 0
        assert "TEST_DEVICE" in main.devices

    def test_quant_options(self):
        assert "DISABLED (禁用)" in main.quant_options
        assert "W8A16_STATIC" in main.quant_options

    def test_compile_options(self):
        assert "true" in main.compile_options
        assert "false" in main.compile_options


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
