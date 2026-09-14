_base_ = '../proposed/odinw_pothole_scale_aware_lora_r16.py'

# Checkpoint intervention/causal ablation: the learned scale gates in the
# selected epoch-11 model saturate above 0.9996 on validation. Replace them by
# constant one while retaining the learned residual projections and LoRA
# weights. For a fresh run this is also a lower-parameter static candidate.
model = dict(scale_fusion_gate_mode='constant_one')

work_dir = 'experiments/odinw_pothole_static_scale_aware_lora_r16_seed42'
