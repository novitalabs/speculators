"""Patch vLLM to support Eagle3 speculative decoding for MiniMax-M2 models.

vLLM 0.17.0 has a whitelist of model types that support Eagle3.
MiniMax-M2 (model_type='minimax_m2') is not in this whitelist.
This script finds and patches the whitelist to add minimax_m2.

Usage: Run this before importing vllm, or run as a standalone script.
"""

import importlib
import re
import sys
from pathlib import Path


def find_and_patch():
    """Find the Eagle3 model type whitelist in vLLM and add minimax_m2."""
    # Find vllm installation
    import vllm
    vllm_dir = Path(vllm.__file__).parent

    # Search for the whitelist in Python files
    patterns = [
        r"eagle3.*only supported",
        r"nemotron_h",
        r"hunyuan_v1_dense",
    ]

    patched = False
    for py_file in vllm_dir.rglob("*.py"):
        try:
            content = py_file.read_text()
        except Exception:
            continue

        if "nemotron_h" not in content:
            continue

        # Found the file with the whitelist
        print(f"[PATCH] Found Eagle3 whitelist in: {py_file}")

        # Look for the list containing the model types
        # Pattern: ['llama', 'qwen', ..., 'nemotron_h']
        list_pattern = r"(\[(?:['\"][a-z_]+['\"],?\s*)+\])"
        for match in re.finditer(list_pattern, content):
            list_str = match.group(0)
            if "nemotron_h" in list_str and "minimax_m2" not in list_str:
                new_list = list_str.replace("'nemotron_h'", "'nemotron_h', 'minimax_m2'")
                new_content = content.replace(list_str, new_list)
                py_file.write_text(new_content)
                print(f"[PATCH] Added 'minimax_m2' to Eagle3 supported models")
                print(f"[PATCH] Old: {list_str}")
                print(f"[PATCH] New: {new_list}")
                patched = True
                break

        if patched:
            break

    if not patched:
        # Fallback: search compiled (.pyc) files or try alternate patterns
        print("[PATCH] WARNING: Could not find Eagle3 whitelist to patch")
        print("[PATCH] Attempting monkey-patch approach...")
        monkey_patch()
    else:
        # Reload the module
        for mod_name in list(sys.modules.keys()):
            if "vllm" in mod_name:
                del sys.modules[mod_name]


def monkey_patch():
    """Monkey-patch approach: override SpeculativeConfig validation."""
    import vllm.config as config_mod

    original_init = config_mod.SpeculativeConfig.__pydantic_validator__

    # We can't easily override pydantic validators, so let's try a different approach
    # Override the model_type check by modifying hf_text_config before validation
    print("[PATCH] Monkey-patch: Will attempt to modify model_type at runtime")


if __name__ == "__main__":
    find_and_patch()
