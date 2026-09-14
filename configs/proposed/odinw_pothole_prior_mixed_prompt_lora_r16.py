_base_ = './odinw_pothole_domain_prompt_fusion_lora_r16.py'

# Routing-collapse repair: retain image-conditioned deviations while mixing
# every route with a 50% uniform prior. With eight prompts this enforces a
# per-prompt lower bound of 0.0625 and an upper bound of 0.5625.
model = dict(domain_prompt_uniform_prior_mix=0.5)

work_dir = 'experiments/odinw_pothole_prior_mixed_prompt_lora_r16_seed42'
