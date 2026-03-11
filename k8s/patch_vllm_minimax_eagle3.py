"""Patch vLLM to support Eagle3 speculative decoding for MiniMax-M2 models.

Applies two patches:
1. Adds 'minimax_m2' to Eagle3 supported model types whitelist
2. Adds aux_hidden_state interface to MiniMaxM2Model/MiniMaxM2ForCausalLM

Run this script before starting vLLM with Eagle3 spec decode for MiniMax-M2.
"""

import re
import shutil
import sys
from pathlib import Path


def patch_speculative_config(vllm_dir: Path) -> bool:
    """Add minimax_m2 to Eagle3 supported models whitelist."""
    spec_config = vllm_dir / "config" / "speculative.py"
    if not spec_config.exists():
        print(f"[PATCH] WARNING: {spec_config} not found")
        return False

    content = spec_config.read_text()
    if "minimax_m2" in content:
        print("[PATCH] minimax_m2 already in whitelist, skipping")
        return True

    content = content.replace(
        '"nemotron_h",',
        '"nemotron_h",\n            "minimax_m2",'
    )
    spec_config.write_text(content)
    print("[PATCH] Added minimax_m2 to Eagle3 supported models whitelist")
    return True


def patch_minimax_model(vllm_dir: Path) -> bool:
    """Add Eagle3 interface to MiniMax-M2 model.

    Uses a fresh-copy approach: restores the original file from backup,
    then applies the complete patch. This avoids issues with partial patches.
    """
    model_file = vllm_dir / "model_executor" / "models" / "minimax_m2.py"
    backup_file = model_file.with_suffix(".py.orig")

    if not model_file.exists():
        print(f"[PATCH] WARNING: {model_file} not found")
        return False

    # If we have a backup, restore from it to get a clean starting point
    if backup_file.exists():
        shutil.copy2(backup_file, model_file)
        print("[PATCH] Restored original minimax_m2.py from backup")
    else:
        # First time: create backup
        shutil.copy2(model_file, backup_file)
        print("[PATCH] Created backup of original minimax_m2.py")

    content = model_file.read_text()

    # 1. Add aux_hidden_state_layers init to MiniMaxM2Model.__init__
    old_init = (
        '        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(\n'
        '            ["hidden_states", "residual"], config.hidden_size\n'
        '        )'
    )
    new_init = (
        '        self.aux_hidden_state_layers = tuple[int, ...]()\n'
        '\n'
        '        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(\n'
        '            ["hidden_states", "residual"], config.hidden_size\n'
        '        )'
    )
    if old_init not in content:
        print("[PATCH] WARNING: Could not find MiniMaxM2Model.__init__ target")
        return False
    content = content.replace(old_init, new_init)

    # 2. Patch MiniMaxM2Model.forward to collect aux hidden states
    old_forward_body = (
        '        for layer in self.layers[self.start_layer : self.end_layer]:\n'
        '            hidden_states, residual = layer(positions, hidden_states, residual)\n'
        '\n'
        '        if not get_pp_group().is_last_rank:\n'
        '            return IntermediateTensors(\n'
        '                {"hidden_states": hidden_states, "residual": residual}\n'
        '            )\n'
        '        hidden_states, _ = self.norm(hidden_states, residual)\n'
        '        return hidden_states'
    )
    new_forward_body = (
        '        aux_hidden_states = []\n'
        '        for idx, layer in enumerate(self.layers[self.start_layer : self.end_layer]):\n'
        '            hidden_states, residual = layer(positions, hidden_states, residual)\n'
        '            if idx in self.aux_hidden_state_layers:\n'
        '                aux_hidden_states.append(hidden_states + residual)\n'
        '\n'
        '        if not get_pp_group().is_last_rank:\n'
        '            return IntermediateTensors(\n'
        '                {"hidden_states": hidden_states, "residual": residual}\n'
        '            )\n'
        '        hidden_states, _ = self.norm(hidden_states, residual)\n'
        '\n'
        '        if len(aux_hidden_states) > 0:\n'
        '            return hidden_states, aux_hidden_states\n'
        '        return hidden_states'
    )
    if old_forward_body not in content:
        print("[PATCH] WARNING: Could not find MiniMaxM2Model.forward target")
        return False
    content = content.replace(old_forward_body, new_forward_body)

    # 3. Add supports_eagle3 class variable and Eagle3 methods to MiniMaxM2ForCausalLM
    old_causal_forward = (
        '    def forward(\n'
        '        self,\n'
        '        input_ids: torch.Tensor | None,\n'
        '        positions: torch.Tensor,\n'
        '        intermediate_tensors: IntermediateTensors | None = None,\n'
        '        inputs_embeds: torch.Tensor | None = None,\n'
        '        **kwargs,\n'
        '    ) -> torch.Tensor | IntermediateTensors:\n'
        '        hidden_states = self.model(\n'
        '            input_ids, positions, intermediate_tensors, inputs_embeds\n'
        '        )\n'
        '        return hidden_states'
    )
    new_causal_forward = (
        '    supports_eagle3 = True\n'
        '    has_own_lm_head = False\n'
        '    has_own_embed_tokens = False\n'
        '\n'
        '    def forward(\n'
        '        self,\n'
        '        input_ids: torch.Tensor | None,\n'
        '        positions: torch.Tensor,\n'
        '        intermediate_tensors: IntermediateTensors | None = None,\n'
        '        inputs_embeds: torch.Tensor | None = None,\n'
        '        **kwargs,\n'
        '    ) -> torch.Tensor | IntermediateTensors:\n'
        '        model_output = self.model(\n'
        '            input_ids, positions, intermediate_tensors, inputs_embeds\n'
        '        )\n'
        '        return model_output\n'
        '\n'
        '    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:\n'
        '        self.model.aux_hidden_state_layers = layers\n'
        '\n'
        '    def get_eagle3_aux_hidden_state_layers(self) -> tuple[int, ...]:\n'
        '        """Return default auxiliary hidden state layers for MiniMax-M2."""\n'
        '        num_layers = len(self.model.layers)\n'
        '        return (2, num_layers // 2, num_layers - 3)'
    )
    if old_causal_forward not in content:
        print("[PATCH] WARNING: Could not find MiniMaxM2ForCausalLM.forward target")
        return False
    content = content.replace(old_causal_forward, new_causal_forward)

    model_file.write_text(content)
    print("[PATCH] Added Eagle3 interface (supports_eagle3 + methods) to MiniMax-M2 model")

    # Verify the patch
    verify_content = model_file.read_text()
    checks = [
        ("supports_eagle3 = True", "supports_eagle3 class variable"),
        ("set_aux_hidden_state_layers", "set_aux_hidden_state_layers method"),
        ("get_eagle3_aux_hidden_state_layers", "get_eagle3_aux_hidden_state_layers method"),
        ("aux_hidden_state_layers", "aux_hidden_state_layers in Model.__init__"),
    ]
    for pattern, desc in checks:
        if pattern not in verify_content:
            print(f"[PATCH] VERIFY FAILED: {desc} not found after patching")
            return False
    print("[PATCH] Verification passed: all Eagle3 interface elements present")
    return True


def revert_model_runner_debug(vllm_dir: Path) -> bool:
    """Revert any debug patches from gpu_model_runner.py."""
    runner_file = vllm_dir / "v1" / "worker" / "gpu_model_runner.py"
    if not runner_file.exists():
        return True

    content = runner_file.read_text()
    if "DEBUG eagle3 check" not in content:
        return True

    # Restore from backup if available
    backup = runner_file.with_suffix(".py.orig")
    if backup.exists():
        shutil.copy2(backup, runner_file)
        print("[PATCH] Reverted gpu_model_runner.py debug patches")
    return True


if __name__ == "__main__":
    import vllm
    vllm_dir = Path(vllm.__file__).parent

    ok1 = patch_speculative_config(vllm_dir)
    ok2 = patch_minimax_model(vllm_dir)
    revert_model_runner_debug(vllm_dir)

    if ok1 and ok2:
        print("[PATCH] All patches applied successfully")
    else:
        print("[PATCH] Some patches failed")
        sys.exit(1)
