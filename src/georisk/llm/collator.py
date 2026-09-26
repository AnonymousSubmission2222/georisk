from dataclasses import dataclass
from typing import Dict, Sequence

import torch
import transformers

from .constants import IGNORE_INDEX


def pad_and_cat(tensor_list):
    max_length = max(tensor.shape[2] for tensor in tensor_list)
    padded_tensors = []
    for tensor in tensor_list:
        pad_length = max_length - tensor.shape[2]
        padded_tensor = torch.nn.functional.pad(tensor, (0, pad_length), "constant", 1)
        padded_tensors.append(padded_tensor)
    return torch.cat(padded_tensors, dim=1)


@dataclass
class GeoRiskCollator:
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = [instance["input_ids"].squeeze(0) for instance in instances]
        labels = [instance["labels"].squeeze(0) for instance in instances]
        position_ids = [instance["position_ids"] for instance in instances]

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        position_ids = pad_and_cat(position_ids)

        input_ids = input_ids[:, : self.tokenizer.model_max_length]
        labels = labels[:, : self.tokenizer.model_max_length]
        position_ids = position_ids[:, :, : self.tokenizer.model_max_length]

        batch = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": input_ids.ne(self.tokenizer.pad_token_id),
            "position_ids": position_ids,
            "pixel_values": torch.cat([instance["pixel_values"] for instance in instances], dim=0),
            "image_grid_thw": torch.cat([instance["image_grid_thw"] for instance in instances], dim=0),
            "trajectory": torch.stack([instance["trajectory"] for instance in instances], dim=0),
            "traj_valid_mask": torch.stack([instance["traj_valid_mask"] for instance in instances], dim=0),
            "stop_labels": torch.stack([instance["stop_label"] for instance in instances], dim=0),
            "stop_valid_mask": torch.stack([instance["stop_valid_mask"] for instance in instances], dim=0),
            "stop_distance_to_goal": torch.stack(
                [instance["stop_distance_to_goal"] for instance in instances], dim=0
            ),
            "phase2_valid_mask": torch.stack([instance["phase2_valid_mask"] for instance in instances], dim=0),
        }
        optional_stack_keys = [
            "clearance_depth_gt",
            "clearance_depth_valid_mask",
            "clearance_geom_valid_mask",
            "clearance_view_to_current_poses",
            "clearance_target_to_current_rot",
            "da3_joint_valid_mask",
        ]
        for key in optional_stack_keys:
            if all(key in instance for instance in instances):
                batch[key] = torch.stack([instance[key] for instance in instances], dim=0)
        if all("da3_joint_feat" in instance for instance in instances):
            feats = [instance["da3_joint_feat"] for instance in instances]
            max_views = max(int(feat.shape[0]) for feat in feats)
            max_tokens = max(int(feat.shape[1]) for feat in feats)
            feat_dim = int(feats[0].shape[-1])
            padded_feats = []
            padded_masks = []
            for idx, feat in enumerate(feats):
                if int(feat.shape[-1]) != feat_dim:
                    raise ValueError(
                        f"DA3 feature dim mismatch in batch: sample0={feat_dim}, sample{idx}={int(feat.shape[-1])}"
                    )
                out = feat.new_zeros((max_views, max_tokens, feat_dim))
                out[: feat.shape[0], : feat.shape[1], :] = feat
                padded_feats.append(out)
                if "da3_joint_token_mask" in instances[idx]:
                    mask = instances[idx]["da3_joint_token_mask"]
                else:
                    mask = torch.ones((feat.shape[0], feat.shape[1]), dtype=torch.float32)
                mask_out = mask.new_zeros((max_views, max_tokens))
                mask_out[: mask.shape[0], : mask.shape[1]] = mask
                padded_masks.append(mask_out.float())
            batch["da3_joint_feat"] = torch.stack(padded_feats, dim=0)
            batch["da3_joint_token_mask"] = torch.stack(padded_masks, dim=0)
        return batch
