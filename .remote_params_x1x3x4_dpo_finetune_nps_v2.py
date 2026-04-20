from copy import deepcopy

from parameters.params_x1x3x4_diffusion_mosesaq_20240824 import params as base_params


params = deepcopy(base_params)

params["data"] = "NPs"

params["training"]["enable_dpo"] = True
params["training"]["beta_dpo"] = 0.3
params["training"]["dpo_max_weight"] = 0.3
params["training"]["dpo_ramp_up_epochs"] = 10
params["training"]["dpo_optimize_x4"] = False
params["training"]["real_data_ratio"] = 0.5
params["training"]["dpo_min_score_gap"] = 0.15
params["training"]["dpo_sampling_every_n_epochs"] = 3
params["training"]["iterative_dpo_enabled"] = False
params["training"]["iterative_dpo_score_threshold"] = 0.0
params["training"]["iterative_dpo_force_update_every_n_rounds"] = 0
params["training"]["buffer_gate_min_validity_rate"] = 0.0
params["training"]["buffer_gate_min_pairs"] = 1
params["training"]["buffer_gate_require_zero_score_failures"] = False
params["training"]["protect_stop_validity_rate"] = 0.0
params["training"]["protect_stop_patience_rounds"] = 0
params["training"]["protect_stop_min_pairs"] = 0
params["training"]["protect_stop_on_zero_winner_group"] = False
params["training"]["num_reference_molecules"] = 3
params["training"]["initial_sampling_attempts"] = 3
params["training"]["pretrained_checkpoint_path"] = "../data/shepherd_chkpts/x1x3x4_diffusion_mosesaq_20240824_submission.ckpt"
params["training"]["nps_data_path"] = "../data/conformers/np/molblock_charges_NPs.pkl"
params["training"]["lr"] = 2e-6
params["training"]["min_lr"] = 2e-6
params["training"]["lr_steps"] = 1
params["training"]["batch_size"] = 2
params["training"]["accumulate_grad_batches"] = 4
params["training"]["num_gpus"] = 1
params["training"]["sampling_gpu_ids"] = [0]
params["training"]["num_workers"] = 2
params["training"]["output_dir"] = "x1x3x4_dpo_finetune_nps_v2"
params["training"]["log_every_n_steps"] = 10
params["training"]["checkpoint_every_n_epochs"] = 1

params["sampling"] = {
    "num_samples_per_molecule": 32,
    "inference_sub_batch_size": 4,
    "fixed_n_atoms": 78,
}

params["dpo"] = {
    "min_score_gap": 0.15,
    "num_surf_points": 400,
    "probe_radius": 1.2,
}
