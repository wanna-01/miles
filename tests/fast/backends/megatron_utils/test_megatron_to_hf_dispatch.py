from unittest.mock import Mock, patch

import pytest

from miles.backends.megatron_utils.megatron_to_hf import _convert_to_hf_core


@pytest.mark.parametrize(
    "model_name",
    [
        "Qwen3.5-27B",
        "qwen3_6-27b",
        "Qwen3.8-27B",
    ],
)
def test_qwen3_5_family_uses_qwen3_5_converter(model_name):
    args = Mock()
    param = Mock()
    expected = [("converted", param)]

    with (
        patch(
            "miles.backends.megatron_utils.megatron_to_hf.convert_qwen3_5_to_hf",
            return_value=expected,
        ) as qwen3_5_converter,
        patch("miles.backends.megatron_utils.megatron_to_hf.convert_qwen2_to_hf") as qwen2_converter,
    ):
        assert _convert_to_hf_core(args, model_name, "parameter", param) == expected

    qwen3_5_converter.assert_called_once_with(args, "parameter", param)
    qwen2_converter.assert_not_called()


@pytest.mark.parametrize("model_name", ["Qwen3-32B", "Qwen3-8B"])
def test_original_qwen3_uses_qwen2_converter(model_name):
    args = Mock()
    param = Mock()
    expected = [("converted", param)]

    with (
        patch("miles.backends.megatron_utils.megatron_to_hf.convert_qwen3_5_to_hf") as qwen3_5_converter,
        patch(
            "miles.backends.megatron_utils.megatron_to_hf.convert_qwen2_to_hf",
            return_value=expected,
        ) as qwen2_converter,
    ):
        assert _convert_to_hf_core(args, model_name, "parameter", param) == expected

    qwen2_converter.assert_called_once_with(args, "parameter", param)
    qwen3_5_converter.assert_not_called()
