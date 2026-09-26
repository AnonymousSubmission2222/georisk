import copy
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from transformers import Qwen3VLConfig

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from georisk.llm.constants import DEFAULT_TRAJ_TOKEN, DEFAULT_STOP_TOKEN
from georisk.llm.qwen_uav_model import GeoRiskForNavigation
from georisk.llm.checkpoint import (METADATA, validate_checkpoint_config, load_non_lora_weights,
                                   save_special_embeddings, load_special_embeddings)
from georisk.llm.train_uav_qwen import (ModelArguments, configure_trainable_params,
    restore_non_lora_trainables_after_lora, find_lora_target_modules,
    find_visual_encoder_block_lora_targets, collect_non_lora_state_dict)
from georisk.llm.prompts import observation_prompt, trajectory_prompt, TRAJECTORY_REPLY
from georisk.model_wrapper.utils.online_rgb import airsim_bgr_to_model_rgb


def tiny_config():
    config = Qwen3VLConfig(
        text_config=dict(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                         vocab_size=151940, rope_scaling=dict(rope_type="default", mrope_section=[1, 1, 2])),
        vision_config=dict(depth=24, hidden_size=32, intermediate_size=64,
                           num_heads=4, out_hidden_size=32, patch_size=16,
                           temporal_patch_size=2, spatial_merge_size=2,
                           deepstack_visual_indexes=[5, 11, 17]),
    )
    config._attn_implementation = "sdpa"
    return config


def tiny_batch(config, device="cpu"):
    image = [config.vision_start_token_id] + [config.image_token_id] * 64 + [config.vision_end_token_id]
    ids = torch.tensor([image + image + [101, 19, 102]])
    batch = dict(input_ids=ids, attention_mask=torch.ones_like(ids),
        pixel_values=torch.randn(512, 1536), image_grid_thw=torch.tensor([[1, 16, 16], [1, 16, 16]]),
        trajectory=torch.randn(1, 10, 3), stop_labels=torch.tensor([0.25]),
        stop_valid_mask=torch.ones(1), phase2_valid_mask=torch.ones(1), traj_valid_mask=torch.ones(1, 10),
        da3_joint_feat=torch.randn(1, 2, 64, 1024), da3_joint_valid_mask=torch.ones(1),
        da3_joint_token_mask=torch.ones(1, 2, 64),
        clearance_depth_gt=torch.full((1, 2, 256, 256), 3, dtype=torch.uint8),
        clearance_depth_valid_mask=torch.ones(1, 2), clearance_geom_valid_mask=torch.ones(1),
        clearance_view_to_current_poses=torch.eye(4).reshape(1, 1, 4, 4).repeat(1, 2, 1, 1),
        clearance_target_to_current_rot=torch.eye(3).unsqueeze(0))
    return {key: value.to(device) for key, value in batch.items()}


def tiny_model():
    model = GeoRiskForNavigation(tiny_config())
    model.get_special_token_id({DEFAULT_STOP_TOKEN: 101, DEFAULT_TRAJ_TOKEN: 102})
    return model


class NavigationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_rgb_boundary_does_not_modify_save_buffer(self):
        bgr = np.full((4, 4, 3), [11, 37, 203], dtype=np.uint8)
        rgb = airsim_bgr_to_model_rgb(bgr)
        np.testing.assert_array_equal(rgb[0, 0], [203, 37, 11])
        np.testing.assert_array_equal(bgr[0, 0], [11, 37, 203])
        self.assertFalse(np.shares_memory(rgb, bgr))
        source = (ROOT / "src/vlnce_src/dino_monitor_online.py").read_text()
        self.assertIn("airsim_bgr_to_model_rgb(np.asarray(img))", source)

    def test_shared_prompt_and_token(self):
        self.assertEqual(observation_prompt("land").count("<image>"), 2)
        self.assertIn("future trajectory points", trajectory_prompt("cruise", "1,0,0", "0,0,0"))
        self.assertEqual(TRAJECTORY_REPLY, "Future trajectory points: <traj>")
        for relative in ("src/georisk/llm/dataset_uav.py", "src/georisk/model_wrapper/utils/travel_qwen_util.py"):
            self.assertIn("TRAJECTORY_REPLY", (ROOT / relative).read_text())

    def test_checkpoint_identity_and_missing_trajectory_token(self):
        validate_checkpoint_config(METADATA)
        for patch in ({"project": "OtherProject"}, {"trajectory_token": "wrong"}, {"georisk_schema_version": 0}):
            with self.assertRaises(ValueError):
                validate_checkpoint_config({**METADATA, **patch})
        model = tiny_model()
        with self.assertRaises(ValueError):
            model._extract_trajectory_states(torch.randn(1, 3, 32), torch.tensor([[1, 2, 3]]), None)
        with self.assertRaises(ValueError):
            model._extract_trajectory_states(torch.randn(1, 3, 32), torch.tensor([[102, 2, 102]]), None)

    def test_shapes_loss_and_invalid_geometry_backward(self):
        model = tiny_model().train()
        batch = tiny_batch(model.config)
        out = model(**batch)
        self.assertEqual(out.predicted_trajectory.shape, (1, 10, 3))
        self.assertEqual(out.predicted_stop_probs.shape, (1,))
        torch.testing.assert_close(out.loss, out.trajectory_loss + .1 * out.stop_loss
                                   + 10 * out.clearance_loss + .5 * out.da3_joint_loss)
        out.loss.backward()
        for module in (model.trajectory_head, model.stop_head, model.da3_joint_projector):
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in module.parameters()), 0)
        model.zero_grad(set_to_none=True)
        batch["da3_joint_valid_mask"].zero_()
        batch["clearance_geom_valid_mask"].zero_()
        out = model(**batch)
        self.assertEqual(float(out.da3_joint_loss.detach()), 0)
        self.assertEqual(float(out.clearance_loss.detach()), 0)
        out.loss.backward()
        self.assertTrue(all(p.grad is not None for p in model.da3_joint_projector.parameters()))

    def test_checkpoint_roundtrip_and_inference_has_no_alignment_head(self):
        torch.manual_seed(42)
        model = tiny_model()
        args = ModelArguments()
        configure_trainable_params(model, args)
        targets = find_lora_target_modules(model) + find_visual_encoder_block_lora_targets(model, (5, 11, 17, 23))
        model = get_peft_model(model, LoraConfig(r=2, lora_alpha=4, target_modules=targets, task_type="CAUSAL_LM"))
        restore_non_lora_trainables_after_lora(model, args)
        core = model.base_model.model
        with tempfile.TemporaryDirectory() as tmp:
            state = collect_non_lora_state_dict(model.named_parameters())
            torch.save(state, Path(tmp) / "non_lora_trainables.bin")
            save_special_embeddings(core, tmp)
            original = core.trajectory_output.weight.detach().clone()
            with torch.no_grad():
                core.trajectory_output.weight.add_(1)
            load_non_lora_weights(model, tmp)
            load_special_embeddings(core, tmp)
            torch.testing.assert_close(original, core.trajectory_output.weight)
            config = tiny_config()
            inference = GeoRiskForNavigation(config, use_da3_joint_supervision=False, use_clearance_supervision=False)
            inference.get_special_token_id(core.special_token_dict)
            self.assertFalse(hasattr(inference, "da3_joint_projector"))
            inference = get_peft_model(inference, LoraConfig(r=2, lora_alpha=4, target_modules=targets, task_type="CAUSAL_LM"))
            load_non_lora_weights(inference, tmp, inference=True)
            inputs = {key: val for key, val in tiny_batch(config).items()
                      if key in {"input_ids", "attention_mask", "pixel_values", "image_grid_thw"}}
            inference.eval()
            with torch.no_grad():
                trajectory = inference(**inputs, return_trajectory=True)
            self.assertEqual(trajectory.shape, (1, 10, 3))
            state.pop(next(iter(state)))
            torch.save(state, Path(tmp) / "non_lora_trainables.bin")
            with self.assertRaises(ValueError):
                load_non_lora_weights(model, tmp)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_single_gpu_two_optimizer_steps_with_visual_lora(self):
        model = tiny_model()
        args = ModelArguments()
        configure_trainable_params(model, args)
        targets = find_lora_target_modules(model) + find_visual_encoder_block_lora_targets(model, (5, 11, 17, 23))
        model = get_peft_model(model, LoraConfig(r=2, lora_alpha=4, target_modules=targets, task_type="CAUSAL_LM"))
        restore_non_lora_trainables_after_lora(model, args)
        model = model.cuda().train()
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
        batch = tiny_batch(model.config, "cuda")
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            loss = model(**batch).loss
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            for block in (5, 11, 17, 23):
                grads = [p.grad for name, p in model.named_parameters()
                         if f"visual.blocks.{block}." in name and "lora_B" in name]
                self.assertTrue(grads and all(g is not None and torch.isfinite(g).all() for g in grads))
                self.assertGreater(sum(float(g.abs().sum()) for g in grads), 0)
            optimizer.step()


if __name__ == "__main__":
    unittest.main()
