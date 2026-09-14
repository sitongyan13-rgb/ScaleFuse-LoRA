"""Minimal, auditable MMDetection extensions for project baselines."""

import math
from typing import Dict, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from mmcv.transforms import BaseTransform
from mmengine.hooks import Hook
from torch import Tensor, nn

from mmdet.models.detectors.grounding_dino import GroundingDINO
from mmdet.models.dense_heads.grounding_dino_head import GroundingDINOHead
from mmdet.registry import HOOKS, MODELS, TRANSFORMS
from mmdet.structures.bbox import (
    autocast_box_type,
    bbox_cxcywh_to_xyxy,
    bbox_overlaps,
)


def object_aware_crop_window(
    bbox: Tensor,
    image_shape: Tuple[int, int],
    crop_scale: float,
    offset_fractions: Tuple[float, float],
) -> Tuple[int, int, int, int]:
    """Return a target-containing crop as ``(left, top, right, bottom)``.

    The crop uses the same relative scale along both image axes. Its size is
    enlarged when necessary so the selected target remains completely inside.
    Fractions choose a reproducible location among all valid offsets.
    """
    if bbox.shape != (4,):
        raise ValueError("bbox must have shape [4]")
    height, width = image_shape
    if height <= 0 or width <= 0:
        raise ValueError("image_shape must be positive")
    if not 0 < crop_scale <= 1:
        raise ValueError("crop_scale must be in (0, 1]")
    if len(offset_fractions) != 2 or any(
        value < 0 or value > 1 for value in offset_fractions
    ):
        raise ValueError("offset fractions must be in [0, 1]")
    x1, y1, x2, y2 = [float(value) for value in bbox]
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError("bbox must be non-empty and inside the image")

    crop_width = min(
        width, max(int(round(width * crop_scale)), int(math.ceil(x2 - x1))))
    crop_height = min(
        height, max(int(round(height * crop_scale)), int(math.ceil(y2 - y1))))
    minimum_left = max(0, int(math.ceil(x2)) - crop_width)
    maximum_left = min(int(math.floor(x1)), width - crop_width)
    minimum_top = max(0, int(math.ceil(y2)) - crop_height)
    maximum_top = min(int(math.floor(y1)), height - crop_height)
    if minimum_left > maximum_left or minimum_top > maximum_top:
        raise RuntimeError("No valid crop window can retain the target")
    fraction_x, fraction_y = offset_fractions
    left = int(round(minimum_left + fraction_x * (maximum_left - minimum_left)))
    top = int(round(minimum_top + fraction_y * (maximum_top - minimum_top)))
    return left, top, left + crop_width, top + crop_height


@TRANSFORMS.register_module()
class ObjectAwareZoomCrop(BaseTransform):
    """Training-only crop that preferentially magnifies small objects.

    A target is sampled with inverse-area weighting. The crop always retains
    that target completely. Other boxes are clipped and kept only when at
    least ``min_retained_area`` of their original area remains. MMEngine's
    worker-specific NumPy seed makes the stochastic decisions reproducible.
    """

    def __init__(
        self,
        prob: float = 0.5,
        crop_scale_range: Tuple[float, float] = (0.5, 0.8),
        selection_power: float = 0.5,
        min_retained_area: float = 0.5,
    ) -> None:
        if not 0 <= prob <= 1:
            raise ValueError("prob must be in [0, 1]")
        if (len(crop_scale_range) != 2
                or not 0 < crop_scale_range[0] <= crop_scale_range[1] <= 1):
            raise ValueError("crop_scale_range must satisfy 0 < low <= high <= 1")
        if selection_power < 0:
            raise ValueError("selection_power must be non-negative")
        if not 0 <= min_retained_area <= 1:
            raise ValueError("min_retained_area must be in [0, 1]")
        self.prob = float(prob)
        self.crop_scale_range = tuple(float(value) for value in crop_scale_range)
        self.selection_power = float(selection_power)
        self.min_retained_area = float(min_retained_area)

    @autocast_box_type()
    def transform(self, results: dict) -> dict:
        if np.random.random() >= self.prob:
            results["zoom_crop_applied"] = False
            return results
        if "img" not in results:
            raise KeyError("ObjectAwareZoomCrop requires 'img'")
        bboxes = results.get("gt_bboxes")
        if bboxes is None or len(bboxes) == 0:
            results["zoom_crop_applied"] = False
            return results
        if results.get("gt_masks") is not None:
            raise ValueError("ObjectAwareZoomCrop currently supports box-only data")

        image = results["img"]
        height, width = image.shape[:2]
        original_areas = bboxes.areas.clamp(min=1e-6)
        weights = original_areas.pow(-self.selection_power)
        probabilities = (weights / weights.sum()).cpu().numpy()
        target_index = int(np.random.choice(len(bboxes), p=probabilities))
        crop_scale = float(np.random.uniform(*self.crop_scale_range))
        offsets = (float(np.random.random()), float(np.random.random()))
        left, top, right, bottom = object_aware_crop_window(
            bboxes.tensor[target_index],
            (height, width),
            crop_scale,
            offsets,
        )

        homography = np.array(
            [[1, 0, -left], [0, 1, -top], [0, 0, 1]], dtype=np.float32)
        if results.get("homography_matrix") is None:
            results["homography_matrix"] = homography
        else:
            results["homography_matrix"] = (
                homography @ results["homography_matrix"])
        results["img"] = image[top:bottom, left:right, ...]
        results["img_shape"] = results["img"].shape[:2]

        bboxes.translate_([-left, -top])
        bboxes.clip_(results["img_shape"])
        retained = bboxes.areas / original_areas
        original_target_fraction = float(
            original_areas[target_index] / float(height * width))
        cropped_target_fraction = float(
            bboxes.areas[target_index]
            / float((bottom - top) * (right - left)))
        valid = ((bboxes.widths > 0) & (bboxes.heights > 0)
                 & (retained >= self.min_retained_area))
        if not bool(valid[target_index]):
            raise RuntimeError("Selected crop target was not retained")
        valid_indices = valid.cpu().numpy()
        results["gt_bboxes"] = bboxes[valid_indices]
        for key in ("gt_bboxes_labels", "gt_ignore_flags", "gt_instances_ids"):
            if results.get(key) is not None:
                results[key] = results[key][valid_indices]
        if results.get("gt_seg_map") is not None:
            results["gt_seg_map"] = results["gt_seg_map"][top:bottom, left:right]

        results["zoom_crop_applied"] = True
        results["zoom_crop_scale"] = crop_scale
        results["zoom_crop_window"] = (left, top, right, bottom)
        results["zoom_crop_target_original_index"] = target_index
        results["zoom_crop_retained_boxes"] = int(valid.sum())
        results["zoom_crop_target_original_area_fraction"] = original_target_fraction
        results["zoom_crop_target_cropped_area_fraction"] = cropped_target_fraction
        results["zoom_crop_target_original_area"] = float(
            original_areas[target_index])
        return results

    def __repr__(self) -> str:
        return (
            "ObjectAwareZoomCrop(prob={!r}, crop_scale_range={!r}, "
            "selection_power={!r}, min_retained_area={!r})"
        ).format(
            self.prob,
            self.crop_scale_range,
            self.selection_power,
            self.min_retained_area,
        )


def localization_quality_token_targets(
    token_targets: Tensor,
    bbox_predictions: Tensor,
    bbox_targets: Tensor,
    bbox_weights: Tensor,
    power: float = 1.0,
) -> Tensor:
    """Scale positive token targets by detached aligned box IoU.

    Grounding DINO normally assigns the same binary token target to every
    Hungarian-matched query regardless of localization quality.  This helper
    retains all negatives and token mappings, but gives well-localized matches
    a larger classification target.  Detaching the IoU prevents classification
    loss from changing box coordinates through this target path.
    """
    if power <= 0:
        raise ValueError("Quality-target power must be positive")
    if token_targets.ndim != 2:
        raise ValueError("Token targets must have shape [queries, tokens]")
    if bbox_predictions.shape != bbox_targets.shape or bbox_targets.ndim != 2:
        raise ValueError("Box predictions and targets must share shape [queries, 4]")
    if bbox_predictions.shape[-1] != 4 or bbox_weights.shape != bbox_targets.shape:
        raise ValueError("Box tensors and weights must have shape [queries, 4]")
    if token_targets.shape[0] != bbox_predictions.shape[0]:
        raise ValueError("Token and box targets must contain the same queries")

    positive = bbox_weights[:, 0] > 0
    quality_targets = token_targets.clone()
    if bool(positive.any()):
        predicted_xyxy = bbox_cxcywh_to_xyxy(bbox_predictions[positive].detach())
        target_xyxy = bbox_cxcywh_to_xyxy(bbox_targets[positive].detach())
        quality = bbox_overlaps(
            predicted_xyxy, target_xyxy, is_aligned=True
        ).clamp_(min=0.0, max=1.0).pow(power)
        quality_targets[positive] = quality_targets[positive] * quality[:, None]
    return quality_targets


def scale_adaptive_regression_weights(
    bbox_targets: Tensor,
    bbox_weights: Tensor,
    reference_area: float = 0.01,
    max_weight: float = 2.0,
) -> Tensor:
    """Continuously upweight regression for small normalized target boxes.

    ``bbox_targets`` are normalized ``cxcywh`` targets. Only positive rows,
    identified by nonzero upstream box weights, are changed. At zero area the
    multiplier approaches ``max_weight`` and it decreases linearly to one at
    ``reference_area``. Larger positives keep their original weight, while
    negatives remain exactly zero.
    """
    if bbox_targets.ndim != 2 or bbox_targets.shape[-1] != 4:
        raise ValueError("bbox_targets must have shape [queries, 4]")
    if bbox_weights.shape != bbox_targets.shape:
        raise ValueError("bbox_weights must match bbox_targets")
    if reference_area <= 0:
        raise ValueError("reference_area must be positive")
    if max_weight < 1:
        raise ValueError("max_weight must be at least one")
    if bool((bbox_targets[:, 2:] < 0).any()):
        raise ValueError("target widths and heights must be non-negative")

    normalized_area = bbox_targets[:, 2] * bbox_targets[:, 3]
    multiplier = 1.0 + (max_weight - 1.0) * torch.clamp(
        1.0 - normalized_area / reference_area, min=0.0, max=1.0)
    return bbox_weights * multiplier[:, None]


@MODELS.register_module()
class ScaleAdaptiveRegressionGroundingDINOHead(GroundingDINOHead):
    """Grounding DINO head with scale-adaptive matched regression weights.

    Hungarian assignment, binary token targets, and denoising supervision are
    unchanged. The override affects only L1/GIoU weights returned for matched
    positives, and therefore has no inference-time path or parameters.
    """

    def __init__(
        self,
        regression_reference_area: float = 0.01,
        regression_max_weight: float = 2.0,
        **kwargs,
    ) -> None:
        if regression_reference_area <= 0:
            raise ValueError("regression_reference_area must be positive")
        if regression_max_weight < 1:
            raise ValueError("regression_max_weight must be at least one")
        self.regression_reference_area = float(regression_reference_area)
        self.regression_max_weight = float(regression_max_weight)
        super().__init__(**kwargs)

    def _get_targets_single(
        self, cls_score: Tensor, bbox_pred: Tensor, gt_instances, img_meta: dict
    ) -> tuple:
        targets = super()._get_targets_single(
            cls_score, bbox_pred, gt_instances, img_meta
        )
        (labels, label_weights, bbox_targets, bbox_weights,
         pos_inds, neg_inds) = targets
        bbox_weights = scale_adaptive_regression_weights(
            bbox_targets,
            bbox_weights,
            reference_area=self.regression_reference_area,
            max_weight=self.regression_max_weight,
        )
        return (labels, label_weights, bbox_targets, bbox_weights,
                pos_inds, neg_inds)


@MODELS.register_module()
class QualityAlignedGroundingDINOHead(GroundingDINOHead):
    """Grounding DINO head with localization-aware positive token targets.

    The inference path and parameterization are identical to the upstream
    head. Only Hungarian-matched classification targets are changed; the
    denoising targets remain binary to preserve stable positive supervision.
    """

    def __init__(self, quality_target_power: float = 1.0, **kwargs) -> None:
        if quality_target_power <= 0:
            raise ValueError("quality_target_power must be positive")
        self.quality_target_power = float(quality_target_power)
        super().__init__(**kwargs)

    def _get_targets_single(
        self, cls_score: Tensor, bbox_pred: Tensor, gt_instances, img_meta: dict
    ) -> tuple:
        targets = super()._get_targets_single(
            cls_score, bbox_pred, gt_instances, img_meta
        )
        (labels, label_weights, bbox_targets, bbox_weights,
         pos_inds, neg_inds) = targets
        labels = localization_quality_token_targets(
            labels,
            bbox_pred,
            bbox_targets,
            bbox_weights,
            power=self.quality_target_power,
        )
        return (labels, label_weights, bbox_targets, bbox_weights,
                pos_inds, neg_inds)


def localization_aware_ranking_loss(
    cls_logits: Tensor,
    token_targets: Tensor,
    bbox_predictions: Tensor,
    bbox_targets: Tensor,
    bbox_weights: Tensor,
    text_token_mask: Tensor,
    min_quality_gap: float = 0.05,
    temperature: float = 0.5,
    hard_negative_k: int = 20,
    hard_negative_weight: float = 0.25,
) -> Tensor:
    """Rank well-localized matches above weaker matches and hard negatives.

    Binary Grounding DINO classification targets remain unchanged. The
    detached aligned IoU is used only to order Hungarian-matched queries, so
    this auxiliary term cannot update box coordinates through its target.
    """
    if cls_logits.ndim != 3 or token_targets.shape != cls_logits.shape:
        raise ValueError("Class logits and token targets must share [B, Q, T]")
    if bbox_predictions.shape != bbox_targets.shape or bbox_targets.ndim != 3:
        raise ValueError("Box predictions and targets must share [B, Q, 4]")
    if bbox_predictions.shape[-1] != 4 or bbox_weights.shape != bbox_targets.shape:
        raise ValueError("Box tensors and weights must have shape [B, Q, 4]")
    if cls_logits.shape[:2] != bbox_predictions.shape[:2]:
        raise ValueError("Class and box tensors must share batch/query axes")
    if text_token_mask.shape != (cls_logits.shape[0], cls_logits.shape[2]):
        raise ValueError("Text mask must have shape [B, T]")
    if min_quality_gap < 0 or temperature <= 0:
        raise ValueError("Quality gap must be non-negative and temperature positive")
    if hard_negative_k < 0 or hard_negative_weight < 0:
        raise ValueError("Hard-negative settings must be non-negative")

    zero = cls_logits[..., :1].sum() * 0.0
    image_losses = []
    for image_index in range(cls_logits.shape[0]):
        positive = bbox_weights[image_index, :, 0] > 0
        if not bool(positive.any()):
            continue
        positive_tokens = token_targets[image_index, positive].clamp(min=0)
        positive_logits = cls_logits[image_index, positive]
        positive_scores = torch.where(
            positive_tokens > 0, positive_logits, torch.zeros_like(positive_logits)
        ).mul(positive_tokens).sum(-1) / positive_tokens.sum(-1).clamp(min=1e-6)

        predicted_xyxy = bbox_cxcywh_to_xyxy(
            bbox_predictions[image_index, positive].detach())
        target_xyxy = bbox_cxcywh_to_xyxy(
            bbox_targets[image_index, positive].detach())
        quality = bbox_overlaps(
            predicted_xyxy, target_xyxy, is_aligned=True
        ).clamp(min=0.0, max=1.0)

        terms = []
        if positive_scores.numel() > 1:
            quality_delta = quality[:, None] - quality[None, :]
            ordered = quality_delta > min_quality_gap
            if bool(ordered.any()):
                score_delta = positive_scores[:, None] - positive_scores[None, :]
                terms.append((
                    F.softplus(-score_delta / temperature)
                    * quality_delta.detach()
                )[ordered].mean())

        negative = ~positive
        if hard_negative_k and bool(negative.any()):
            valid_tokens = text_token_mask[image_index].bool()
            negative_logits = cls_logits[image_index, negative].masked_fill(
                ~valid_tokens[None, :], float("-inf"))
            negative_scores = negative_logits.max(-1).values
            count = min(hard_negative_k, int(negative_scores.numel()))
            hard_scores = negative_scores.topk(count).values
            hard_loss = F.softplus(
                (hard_scores[None, :] - positive_scores[:, None]) / temperature
            )
            terms.append(hard_negative_weight * (
                hard_loss * quality.detach()[:, None]
            ).mean())

        if terms:
            image_losses.append(torch.stack(terms).sum())
    return torch.stack(image_losses).mean() if image_losses else zero


@MODELS.register_module()
class LocalizationRankingGroundingDINOHead(GroundingDINOHead):
    """Grounding DINO head with a final-layer localization ranking loss."""

    def __init__(
        self,
        ranking_loss_weight: float = 0.1,
        ranking_min_quality_gap: float = 0.05,
        ranking_temperature: float = 0.5,
        ranking_hard_negative_k: int = 20,
        ranking_hard_negative_weight: float = 0.25,
        **kwargs,
    ) -> None:
        if ranking_loss_weight <= 0:
            raise ValueError("ranking_loss_weight must be positive")
        self.ranking_loss_weight = float(ranking_loss_weight)
        self.ranking_min_quality_gap = float(ranking_min_quality_gap)
        self.ranking_temperature = float(ranking_temperature)
        self.ranking_hard_negative_k = int(ranking_hard_negative_k)
        self.ranking_hard_negative_weight = float(ranking_hard_negative_weight)
        super().__init__(**kwargs)

    def loss_by_feat(
        self,
        all_layers_cls_scores: Tensor,
        all_layers_bbox_preds: Tensor,
        enc_cls_scores: Tensor,
        enc_bbox_preds: Tensor,
        batch_gt_instances,
        batch_img_metas,
        dn_meta,
        batch_gt_instances_ignore=None,
    ) -> Dict[str, Tensor]:
        losses = super().loss_by_feat(
            all_layers_cls_scores,
            all_layers_bbox_preds,
            enc_cls_scores,
            enc_bbox_preds,
            batch_gt_instances,
            batch_img_metas,
            dn_meta,
            batch_gt_instances_ignore,
        )
        matching_cls, matching_boxes, _, _ = self.split_outputs(
            all_layers_cls_scores, all_layers_bbox_preds, dn_meta)
        final_cls = matching_cls[-1]
        final_boxes = matching_boxes[-1]
        with torch.no_grad():
            targets = self.get_targets(
                [row for row in final_cls],
                [row for row in final_boxes],
                batch_gt_instances,
                batch_img_metas,
            )
        labels_list, _, bbox_targets_list, bbox_weights_list, _, _ = targets
        token_targets = torch.stack(labels_list)
        bbox_targets = torch.stack(bbox_targets_list)
        bbox_weights = torch.stack(bbox_weights_list)
        text_mask = final_cls.new_zeros(
            (final_cls.shape[0], self.max_text_len), dtype=torch.bool)
        text_mask[:, :self.text_masks.shape[1]] = self.text_masks.bool()
        losses["loss_localization_ranking"] = self.ranking_loss_weight * (
            localization_aware_ranking_loss(
                final_cls,
                token_targets,
                final_boxes,
                bbox_targets,
                bbox_weights,
                text_mask,
                min_quality_gap=self.ranking_min_quality_gap,
                temperature=self.ranking_temperature,
                hard_negative_k=self.ranking_hard_negative_k,
                hard_negative_weight=self.ranking_hard_negative_weight,
            ))
        return losses


class ResidualAdapter(nn.Module):
    """A zero-initialized bottleneck residual adapter."""

    def __init__(self, dim: int = 256, bottleneck: int = 16) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, bottleneck)
        self.act = nn.GELU()
        self.up = nn.Linear(bottleneck, dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs + self.up(self.act(self.down(self.norm(inputs))))


class LoRALinear(nn.Linear):
    """Checkpoint-compatible linear layer with a low-rank residual update.

    The inherited ``weight`` and ``bias`` names remain unchanged so an
    official Grounding DINO checkpoint can initialize the frozen base layer.
    Only ``lora_down`` and ``lora_up`` are new checkpoint keys.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.05,
        bias: bool = True,
    ) -> None:
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        super().__init__(in_features, out_features, bias=bias)
        self.rank = rank
        self.scaling = float(alpha) / rank
        self.lora_dropout = nn.Dropout(dropout)
        self.lora_down = nn.Linear(in_features, rank, bias=False)
        self.lora_up = nn.Linear(rank, out_features, bias=False)
        self.reset_lora_parameters()

    def reset_lora_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

    @classmethod
    def from_linear(
        cls,
        module: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float,
    ) -> "LoRALinear":
        replacement = cls(
            module.in_features,
            module.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            bias=module.bias is not None,
        ).to(device=module.weight.device, dtype=module.weight.dtype)
        with torch.no_grad():
            replacement.weight.copy_(module.weight)
            if module.bias is not None:
                replacement.bias.copy_(module.bias)
        return replacement

    def forward(self, inputs: Tensor) -> Tensor:
        base = nn.functional.linear(inputs, self.weight, self.bias)
        update = self.lora_up(self.lora_down(self.lora_dropout(inputs)))
        return base + update * self.scaling


class VisualConditionedPromptFusion(nn.Module):
    """Fuse a continuous domain-prompt bank using image context.

    Multi-scale visual features are globally pooled and used to predict a
    simplex over a small learned prompt bank.  A zero-initialized projection
    injects the fused prompt into valid text tokens, preserving the original
    class strings, token-positive map, and exact pretrained output at
    initialization.
    """

    def __init__(
        self,
        dim: int = 256,
        num_prompts: int = 8,
        gate_hidden_dim: int = 128,
        temperature: float = 1.0,
        gate_mode: str = "learned",
        uniform_prior_mix: float = 0.0,
    ) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError("Prompt dimension must be positive")
        if num_prompts <= 1:
            raise ValueError("Prompt bank must contain at least two prompts")
        if gate_hidden_dim <= 0:
            raise ValueError("Prompt gate hidden dimension must be positive")
        if temperature <= 0:
            raise ValueError("Prompt temperature must be positive")
        if gate_mode not in ("learned", "uniform"):
            raise ValueError("Prompt gate mode must be 'learned' or 'uniform'")
        if not 0.0 <= uniform_prior_mix <= 1.0:
            raise ValueError("Uniform-prior mix must lie in [0, 1]")
        self.dim = dim
        self.num_prompts = num_prompts
        self.temperature = float(temperature)
        self.gate_mode = gate_mode
        self.uniform_prior_mix = float(uniform_prior_mix)
        self.prompt_bank = nn.Parameter(torch.empty(num_prompts, dim))
        self.gate = (
            nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, gate_hidden_dim),
                nn.GELU(),
                nn.Linear(gate_hidden_dim, num_prompts),
            )
            if gate_mode == "learned"
            else None
        )
        self.projection = nn.Linear(dim, dim, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.prompt_bank, mean=0.0, std=0.02)
        if self.gate is not None:
            for module in self.gate.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.zeros_(module.bias)
        nn.init.zeros_(self.projection.weight)

    def forward(
        self,
        image_features: Sequence[Tensor],
        text_embeddings: Tensor,
        text_token_mask: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Return adapted embeddings and per-image prompt weights."""
        if not image_features:
            raise ValueError("At least one image feature level is required")
        if text_embeddings.ndim != 3:
            raise ValueError("Text embeddings must have shape [B, L, C]")
        if text_token_mask.shape != text_embeddings.shape[:2]:
            raise ValueError("Text-token mask must have shape [B, L]")
        contexts = []
        for feature in image_features:
            if feature.ndim != 4:
                raise ValueError("Image features must have shape [B, C, H, W]")
            if feature.shape[0] != text_embeddings.shape[0]:
                raise ValueError("Image/text batch sizes must match")
            if feature.shape[1] != self.dim:
                raise ValueError(
                    "Expected image feature dimension {}, got {}".format(
                        self.dim, feature.shape[1]
                    )
                )
            contexts.append(feature.mean(dim=(-2, -1)))
        if self.gate is None:
            prompt_weights = text_embeddings.new_full(
                (text_embeddings.shape[0], self.num_prompts),
                1.0 / self.num_prompts,
            )
        else:
            visual_context = torch.stack(contexts, dim=1).mean(dim=1)
            prompt_weights = torch.softmax(
                self.gate(visual_context) / self.temperature, dim=-1
            )
            if self.uniform_prior_mix:
                prompt_weights = (
                    (1.0 - self.uniform_prior_mix) * prompt_weights
                    + self.uniform_prior_mix / self.num_prompts
                )
        fused_prompt = prompt_weights @ self.prompt_bank
        prompt_residual = self.projection(fused_prompt).unsqueeze(1)
        valid_tokens = text_token_mask.to(text_embeddings.dtype).unsqueeze(-1)
        adapted = text_embeddings + prompt_residual * valid_tokens
        return adapted, prompt_weights


class ScaleAwareCrossLevelFusion(nn.Module):
    """Inject coarse semantics into finer feature maps with gated residuals.

    Grounding DINO's channel mapper emits a multi-level feature tuple but does
    not perform an FPN-style top-down fusion.  This module targets localization
    of small objects by propagating each coarse level into its adjacent finer
    level.  Every residual projection is zero initialized, so the complete
    detector is exactly checkpoint-compatible at initialization.
    """

    def __init__(
        self,
        dim: int = 256,
        num_levels: int = 4,
        gate_hidden_dim: int = 64,
        interpolation_mode: str = "bilinear",
        gate_mode: str = "learned",
    ) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError("Feature dimension must be positive")
        if num_levels < 2:
            raise ValueError("Cross-level fusion requires at least two levels")
        if gate_hidden_dim <= 0:
            raise ValueError("Scale gate hidden dimension must be positive")
        if interpolation_mode not in ("nearest", "bilinear"):
            raise ValueError("Interpolation mode must be 'nearest' or 'bilinear'")
        if gate_mode not in ("learned", "constant_one"):
            raise ValueError("Scale gate mode must be 'learned' or 'constant_one'")
        self.dim = dim
        self.num_levels = num_levels
        self.interpolation_mode = interpolation_mode
        self.gate_mode = gate_mode
        self.coarse_norms = nn.ModuleList(
            nn.GroupNorm(32, dim, affine=False) for _ in range(num_levels - 1)
        )
        self.residual_projections = nn.ModuleList(
            nn.Conv2d(dim, dim, kernel_size=1, bias=False)
            for _ in range(num_levels - 1)
        )
        self.gates = nn.ModuleList(
            nn.Sequential(
                nn.LayerNorm(2 * dim),
                nn.Linear(2 * dim, gate_hidden_dim),
                nn.GELU(),
                nn.Linear(gate_hidden_dim, 1),
            )
            for _ in range(num_levels - 1)
        ) if gate_mode == "learned" else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for projection in self.residual_projections:
            nn.init.zeros_(projection.weight)
        if self.gates is not None:
            for gate in self.gates:
                for module in gate.modules():
                    if isinstance(module, nn.Linear):
                        nn.init.xavier_uniform_(module.weight)
                        nn.init.zeros_(module.bias)

    def forward(
        self, image_features: Sequence[Tensor]
    ) -> Tuple[Tuple[Tensor, ...], Tensor]:
        """Return top-down calibrated features and gates shaped ``[B, L-1]``."""
        if len(image_features) != self.num_levels:
            raise ValueError(
                "Expected {} feature levels, got {}".format(
                    self.num_levels, len(image_features)
                )
            )
        batch_size = image_features[0].shape[0]
        for feature in image_features:
            if feature.ndim != 4:
                raise ValueError("Image features must have shape [B, C, H, W]")
            if feature.shape[:2] != (batch_size, self.dim):
                raise ValueError(
                    "Every feature must have batch/channel dimensions [{}, {}]".format(
                        batch_size, self.dim
                    )
                )
        outputs = list(image_features)
        gate_values = [None] * (self.num_levels - 1)
        # Coarse-to-fine ordering makes every fine level receive the already
        # enriched representation from its immediate coarser neighbor.
        for fine_index in range(self.num_levels - 2, -1, -1):
            fine = outputs[fine_index]
            coarse = outputs[fine_index + 1]
            if self.gates is None:
                gate = fine.new_ones((batch_size, 1))
            else:
                context = torch.cat(
                    [fine.mean(dim=(-2, -1)), coarse.mean(dim=(-2, -1))], dim=-1
                )
                gate = torch.sigmoid(self.gates[fine_index](context))
            upsample_args = {}
            if self.interpolation_mode == "bilinear":
                upsample_args["align_corners"] = False
            upsampled = F.interpolate(
                self.coarse_norms[fine_index](coarse),
                size=fine.shape[-2:],
                mode=self.interpolation_mode,
                **upsample_args
            )
            residual = self.residual_projections[fine_index](upsampled)
            outputs[fine_index] = fine + residual * gate[:, :, None, None]
            gate_values[fine_index] = gate.squeeze(-1)
        return tuple(outputs), torch.stack(gate_values, dim=-1)


@MODELS.register_module()
class AdapterGroundingDINO(GroundingDINO):
    """Grounding DINO with small residual adapters after the encoder."""

    def __init__(self, adapter_bottleneck: int = 16, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.vision_adapter = ResidualAdapter(self.embed_dims, adapter_bottleneck)
        self.text_adapter = ResidualAdapter(self.embed_dims, adapter_bottleneck)

    def forward_encoder(
        self,
        feat: Tensor,
        feat_mask: Tensor,
        feat_pos: Tensor,
        spatial_shapes: Tensor,
        level_start_index: Tensor,
        valid_ratios: Tensor,
        text_dict: Dict,
    ) -> Dict:
        outputs = super().forward_encoder(
            feat=feat,
            feat_mask=feat_mask,
            feat_pos=feat_pos,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            text_dict=text_dict,
        )
        outputs["memory"] = self.vision_adapter(outputs["memory"])
        outputs["memory_text"] = self.text_adapter(outputs["memory_text"])
        return outputs


@MODELS.register_module()
class FusionLoRAGroundingDINO(GroundingDINO):
    """Grounding DINO with distributed LoRA in fusion/decoder/head layers."""

    def __init__(
        self,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
        lora_target_roots: Sequence[str] = (
            "encoder",
            "decoder",
            "bbox_head",
            "memory_trans_fc",
            "text_feat_map",
        ),
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_target_roots = tuple(lora_target_roots)
        self.lora_target_names = self._inject_lora()

    def _is_lora_target(self, name: str) -> bool:
        in_target_root = any(
            name == root or name.startswith(root + ".")
            for root in self.lora_target_roots
        )
        # torch.nn.MultiheadAttention calls functional code with the
        # out-projection weight directly, so replacing this module would not
        # execute LoRALinear.forward.
        return in_target_root and ".attn.out_proj" not in name

    def _inject_lora(self):
        targets = [
            (name, module)
            for name, module in self.named_modules()
            if isinstance(module, nn.Linear) and self._is_lora_target(name)
        ]
        names = []
        for name, module in targets:
            parent_name, child_name = name.rsplit(".", 1) if "." in name else ("", name)
            parent = self.get_submodule(parent_name) if parent_name else self
            replacement = LoRALinear.from_linear(
                module,
                rank=self.lora_rank,
                alpha=self.lora_alpha,
                dropout=self.lora_dropout,
            )
            setattr(parent, child_name, replacement)
            names.append(name)
        if not names:
            raise RuntimeError("No linear modules matched the LoRA target roots")
        return tuple(names)

    def init_weights(self) -> None:
        super().init_weights()
        # DeformableDETR initializes all encoder/decoder matrices with Xavier,
        # which includes newly inserted LoRA matrices. Restore the required
        # zero initial residual after that generic initialization.
        for module in self.modules():
            if isinstance(module, LoRALinear):
                module.reset_lora_parameters()


@MODELS.register_module()
class DomainPromptFusionGroundingDINO(FusionLoRAGroundingDINO):
    """Fusion-LoRA Grounding DINO with visual-conditioned domain prompts."""

    def __init__(
        self,
        domain_prompt_count: int = 8,
        domain_prompt_gate_hidden_dim: int = 128,
        domain_prompt_temperature: float = 1.0,
        domain_prompt_gate_mode: str = "learned",
        domain_prompt_uniform_prior_mix: float = 0.0,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.domain_prompt_fusion = VisualConditionedPromptFusion(
            dim=self.embed_dims,
            num_prompts=domain_prompt_count,
            gate_hidden_dim=domain_prompt_gate_hidden_dim,
            temperature=domain_prompt_temperature,
            gate_mode=domain_prompt_gate_mode,
            uniform_prior_mix=domain_prompt_uniform_prior_mix,
        )

    def init_weights(self) -> None:
        super().init_weights()
        # Generic transformer initialization visits new linear modules. Reset
        # the projection last to retain exact pretrained behavior at step 0.
        self.domain_prompt_fusion.reset_parameters()

    def forward_transformer(
        self,
        img_feats: Sequence[Tensor],
        text_dict: Dict,
        batch_data_samples=None,
    ) -> Dict:
        adapted_text_dict = dict(text_dict)
        adapted_embeddings, _ = self.domain_prompt_fusion(
            image_features=img_feats,
            text_embeddings=text_dict["embedded"],
            text_token_mask=text_dict["text_token_mask"],
        )
        adapted_text_dict["embedded"] = adapted_embeddings
        return super().forward_transformer(
            img_feats=img_feats,
            text_dict=adapted_text_dict,
            batch_data_samples=batch_data_samples,
        )


@MODELS.register_module()
class ScaleAwareFusionLoRAGroundingDINO(FusionLoRAGroundingDINO):
    """Fusion-LoRA Grounding DINO with gated top-down scale calibration."""

    def __init__(
        self,
        scale_fusion_num_levels: int = 4,
        scale_fusion_gate_hidden_dim: int = 64,
        scale_fusion_interpolation: str = "bilinear",
        scale_fusion_gate_mode: str = "learned",
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.scale_aware_fusion = ScaleAwareCrossLevelFusion(
            dim=self.embed_dims,
            num_levels=scale_fusion_num_levels,
            gate_hidden_dim=scale_fusion_gate_hidden_dim,
            interpolation_mode=scale_fusion_interpolation,
            gate_mode=scale_fusion_gate_mode,
        )

    def init_weights(self) -> None:
        super().init_weights()
        # Restore exact identity after generic detector initialization visits
        # the newly introduced convolutional projections.
        self.scale_aware_fusion.reset_parameters()

    def forward_transformer(
        self,
        img_feats: Sequence[Tensor],
        text_dict: Dict,
        batch_data_samples=None,
    ) -> Dict:
        calibrated_features, _ = self.scale_aware_fusion(img_feats)
        return super().forward_transformer(
            img_feats=calibrated_features,
            text_dict=text_dict,
            batch_data_samples=batch_data_samples,
        )


@HOOKS.register_module()
class SetTrainableModulesHook(Hook):
    """Set requires_grad at training start using auditable name prefixes."""

    priority = "VERY_HIGH"

    def __init__(
        self,
        freeze_prefixes=None,
        train_only_prefixes=None,
        train_only_substrings=None,
    ) -> None:
        self.freeze_prefixes = tuple(freeze_prefixes or ())
        self.train_only_prefixes = tuple(train_only_prefixes or ())
        self.train_only_substrings = tuple(train_only_substrings or ())

    def before_train(self, runner) -> None:
        model = runner.model.module if hasattr(runner.model, "module") else runner.model
        for name, parameter in model.named_parameters():
            if self.train_only_prefixes or self.train_only_substrings:
                parameter.requires_grad = (
                    name.startswith(self.train_only_prefixes)
                    or any(item in name for item in self.train_only_substrings)
                )
            elif name.startswith(self.freeze_prefixes):
                parameter.requires_grad = False
        total = sum(parameter.numel() for parameter in model.parameters())
        trainable = sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
        runner.logger.info(
            "SetTrainableModulesHook: trainable=%d total=%d ratio=%.8f",
            trainable,
            total,
            trainable / total,
        )


@HOOKS.register_module()
class GradientAndMemoryAuditHook(Hook):
    """Log finite/nonzero gradients and CUDA peak memory without mutation."""

    priority = "VERY_LOW"

    def __init__(self, interval: int = 1) -> None:
        self.interval = interval
        self._handles = []
        self._active = False
        self._tensors = 0
        self._nonzero = 0
        self._nonfinite = 0
        self._squared_norm = 0.0

    def before_train(self, runner) -> None:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        model = runner.model.module if hasattr(runner.model, "module") else runner.model

        def capture_gradient(gradient):
            if not self._active:
                return gradient
            self._tensors += 1
            if not bool(torch.isfinite(gradient).all()):
                self._nonfinite += 1
            if bool(torch.count_nonzero(gradient)):
                self._nonzero += 1
            self._squared_norm += float(gradient.detach().float().norm()) ** 2
            return gradient

        for parameter in model.parameters():
            if parameter.requires_grad:
                self._handles.append(parameter.register_hook(capture_gradient))

    def before_train_iter(self, runner, batch_idx: int, data_batch=None) -> None:
        self._active = (runner.iter == 0) or ((runner.iter + 1) % self.interval == 0)
        self._tensors = 0
        self._nonzero = 0
        self._nonfinite = 0
        self._squared_norm = 0.0

    def after_train_iter(
        self, runner, batch_idx: int, data_batch=None, outputs=None
    ) -> None:
        if (runner.iter + 1) % self.interval != 0 and runner.iter != 0:
            return
        global_norm = math.sqrt(self._squared_norm)
        allocated = (
            torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
        )
        reserved = (
            torch.cuda.max_memory_reserved() if torch.cuda.is_available() else 0
        )
        runner.message_hub.update_scalar("train/gradient_global_norm", global_norm)
        runner.message_hub.update_scalar(
            "train/cuda_peak_allocated_bytes", allocated
        )
        runner.message_hub.update_scalar("train/cuda_peak_reserved_bytes", reserved)
        runner.logger.info(
            "gradient_audit iter=%d tensors=%d nonzero=%d nonfinite=%d "
            "global_norm=%.8g cuda_peak_allocated_bytes=%d "
            "cuda_peak_reserved_bytes=%d",
            runner.iter + 1,
            self._tensors,
            self._nonzero,
            self._nonfinite,
            global_norm,
            allocated,
            reserved,
        )
        self._active = False

    def after_train(self, runner) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []
