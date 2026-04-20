import argparse
import datetime
import importlib
import json
import multiprocessing
import pickle
import resource
import shutil
from pathlib import Path

import numpy as np
import rdkit.Chem as Chem
import torch
import torch.multiprocessing

import pytorch_lightning as pl
from lightning_fabric.utilities.seed import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from rdkit.Chem import AllChem

from shepherd.datasets import HeteroDataset, get_atomic_partial_charges
from shepherd.dpo_dataset import DPODataset
from shepherd.dpo_utils import (
    MultiGPUOnlineSampler,
    OnlineSampler,
    PreferencePairBuilder,
    ShepherdScorer,
)
from shepherd.lightning_module import LightningModule
from shepherd.mixed_dataloader import create_mixed_dataloader
from shepherd.shepherd_score_utils.conformer_generation import update_mol_coordinates
from shepherd.shepherd_score_utils.generate_point_cloud import get_atomic_vdw_radii


rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))

SHARING_STRATEGY = "file_system"
torch.multiprocessing.set_sharing_strategy(SHARING_STRATEGY)


def set_worker_sharing_strategy(worker_id):
    torch.multiprocessing.set_sharing_strategy(SHARING_STRATEGY)


def resolve_path(base_dir, raw_path):
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def load_reference_molecules(params, script_dir):
    nps_data_path = resolve_path(script_dir, params["training"]["nps_data_path"])
    with open(nps_data_path, "rb") as handle:
        molblocks_and_charges = pickle.load(handle)
    num_reference_molecules = int(params["training"].get("num_reference_molecules", 1))
    return molblocks_and_charges[:num_reference_molecules]


def create_dataset(params, molblocks_and_charges):
    return HeteroDataset(
        molblocks_and_charges=molblocks_and_charges,
        noise_schedule_dict=params["noise_schedules"],
        explicit_hydrogens=params["dataset"]["explicit_hydrogens"],
        use_MMFF94_charges=params["dataset"]["use_MMFF94_charges"],
        formal_charge_diffusion=params["x1_formal_charge_diffusion"],
        x1=params["dataset"]["compute_x1"],
        x2=params["dataset"]["compute_x2"],
        x3=params["dataset"]["compute_x3"],
        x4=params["dataset"]["compute_x4"],
        recenter_x1=params["dataset"]["x1"]["recenter"],
        add_virtual_node_x1=params["dataset"]["x1"]["add_virtual_node"],
        remove_noise_COM_x1=params["dataset"]["x1"]["remove_noise_COM"],
        atom_types_x1=params["dataset"]["x1"]["atom_types"],
        charge_types_x1=params["dataset"]["x1"]["charge_types"],
        bond_types_x1=params["dataset"]["x1"]["bond_types"],
        scale_atom_features_x1=params["dataset"]["x1"]["scale_atom_features"],
        scale_bond_features_x1=params["dataset"]["x1"]["scale_bond_features"],
        independent_timesteps_x2=params["dataset"]["x2"]["independent_timesteps"],
        recenter_x2=params["dataset"]["x2"]["recenter"],
        add_virtual_node_x2=params["dataset"]["x2"]["add_virtual_node"],
        remove_noise_COM_x2=params["dataset"]["x2"]["remove_noise_COM"],
        num_points_x2=params["dataset"]["x2"]["num_points"],
        independent_timesteps_x3=params["dataset"]["x3"]["independent_timesteps"],
        recenter_x3=params["dataset"]["x3"]["recenter"],
        add_virtual_node_x3=params["dataset"]["x3"]["add_virtual_node"],
        remove_noise_COM_x3=params["dataset"]["x3"]["remove_noise_COM"],
        num_points_x3=params["dataset"]["x3"]["num_points"],
        scale_node_features_x3=params["dataset"]["x3"]["scale_node_features"],
        independent_timesteps_x4=params["dataset"]["x4"]["independent_timesteps"],
        recenter_x4=params["dataset"]["x4"]["recenter"],
        add_virtual_node_x4=params["dataset"]["x4"]["add_virtual_node"],
        remove_noise_COM_x4=params["dataset"]["x4"]["remove_noise_COM"],
        max_node_types_x4=params["dataset"]["x4"]["max_node_types"],
        scale_node_features_x4=params["dataset"]["x4"]["scale_node_features"],
        scale_vector_features_x4=params["dataset"]["x4"]["scale_vector_features"],
        multivectors=params["dataset"]["x4"]["multivectors"],
        check_accessibility=params["dataset"]["x4"]["check_accessibility"],
        probe_radius=params["dataset"]["probe_radius"],
    )


def get_partial_charges_with_fallback(mol):
    try:
        return get_atomic_partial_charges(mol)
    except Exception:
        try:
            AllChem.ComputeGasteigerCharges(mol)
            charges = np.array(
                [atom.GetDoubleProp("_GasteigerCharge") for atom in mol.GetAtoms()],
                dtype=float,
            )
            return np.nan_to_num(charges, nan=0.0)
        except Exception:
            return np.zeros(mol.GetNumAtoms(), dtype=float)


def build_reference_cache(molblocks_and_charges):
    cache = {}
    for idx, (molblock, charges) in enumerate(molblocks_and_charges):
        mol = Chem.MolFromMolBlock(molblock, removeHs=False)
        if mol is None:
            raise ValueError(f"Failed to parse reference molecule at index {idx}")
        if mol.GetNumConformers() == 0:
            raise ValueError(f"Reference molecule at index {idx} is missing a conformer")

        coords = np.array(mol.GetConformer().GetPositions())
        coords = coords - coords.mean(axis=0)
        mol = update_mol_coordinates(mol, coords)
        radii = get_atomic_vdw_radii(mol)

        if charges is None:
            charges = get_partial_charges_with_fallback(mol)
        else:
            charges = np.asarray(charges, dtype=float)

        cache[idx] = {
            "mol": mol,
            "charges": charges,
            "radii": radii,
            "mol_coordinates": coords,
        }
    return cache


def load_model(params, output_dir, script_dir):
    ckpt_path = output_dir / "last.ckpt"
    if ckpt_path.exists():
        model_pl = LightningModule.load_from_checkpoint(str(ckpt_path), params=params, strict=False)
        return model_pl, str(ckpt_path)

    pretrained_path = params["training"].get("pretrained_checkpoint_path")
    if pretrained_path:
        resolved_pretrained = resolve_path(script_dir, pretrained_path)
        if resolved_pretrained.exists():
            model_pl = LightningModule.load_from_checkpoint(
                str(resolved_pretrained),
                params=params,
                strict=False,
            )
            return model_pl, None

    return LightningModule(params), None


def sample_preference_pairs(sampler, params, reference_molblocks_and_charges):
    scorer = ShepherdScorer(params)
    pair_builder = PreferencePairBuilder(params, scorer=scorer)
    generated_samples, reference_mols = sampler.sample(reference_molblocks_and_charges)
    return pair_builder.evaluate_and_build_pairs(generated_samples, reference_mols)


class DPOSamplingCallback(pl.Callback):
    def __init__(self, params, references, dpo_dataset, output_dir, sampler):
        super().__init__()
        self.params = params
        self.references = references
        self.dpo_dataset = dpo_dataset
        self.output_dir = output_dir
        self.sampler = sampler
        self.save_pairs_detail = params["training"].get("save_pairs_detail", False)
        self.metrics_path = output_dir / "dpo_round_metrics.json"
        self.metrics = []
        self.best_score = -float("inf")
        self.best_checkpoint_score = -float("inf")
        self.best_checkpoint_latest_path = output_dir / "best_dpo_score.ckpt"
        self.best_checkpoint_history_dir = output_dir / "best_dpo_score_history"
        self.best_checkpoint_index_path = output_dir / "best_dpo_score_history.jsonl"
        self.iterative_dpo_enabled = params["training"].get("iterative_dpo_enabled", False)
        self.score_threshold = params["training"].get("iterative_dpo_score_threshold", 0.0)
        self.force_update_every_n_rounds = params["training"].get(
            "iterative_dpo_force_update_every_n_rounds",
            0,
        )
        self.rounds_since_ref_update = 0
        self.buffer_gate_min_validity_rate = params["training"].get(
            "buffer_gate_min_validity_rate",
            0.0,
        )
        self.buffer_gate_min_pairs = params["training"].get("buffer_gate_min_pairs", 1)
        self.buffer_gate_require_zero_score_failures = params["training"].get(
            "buffer_gate_require_zero_score_failures",
            False,
        )
        self.protect_stop_validity_rate = params["training"].get(
            "protect_stop_validity_rate",
            0.0,
        )
        self.protect_stop_patience_rounds = params["training"].get(
            "protect_stop_patience_rounds",
            0,
        )
        self.protect_stop_min_pairs = params["training"].get("protect_stop_min_pairs", 0)
        self.protect_stop_on_zero_winner_group = params["training"].get(
            "protect_stop_on_zero_winner_group",
            False,
        )
        self.unhealthy_round_streak = 0
        self._load_existing_metrics()

    def _load_existing_metrics(self):
        if not self.metrics_path.exists():
            return
        try:
            with open(self.metrics_path, "r", encoding="utf-8") as handle:
                existing_metrics = json.load(handle)
        except Exception:
            return
        if not isinstance(existing_metrics, list):
            return

        self.metrics = existing_metrics
        historical_scores = [
            float(metric["avg_score"])
            for metric in existing_metrics
            if metric.get("avg_score") is not None and np.isfinite(metric.get("avg_score"))
        ]
        if historical_scores:
            historical_best = max(historical_scores)
            self.best_score = historical_best
            self.best_checkpoint_score = historical_best

    def _maybe_save_best_checkpoint(
        self,
        trainer,
        avg_score,
        epoch,
        num_pairs,
        validity_stats,
        sampling_error,
    ):
        if sampling_error is not None or num_pairs <= 0:
            return None
        if avg_score == -float("inf") or not np.isfinite(avg_score):
            return None
        if avg_score <= self.best_checkpoint_score:
            return None

        round_idx = len(self.metrics)
        score_tag = f"{avg_score:.4f}".replace("-", "neg").replace(".", "p")
        self.best_checkpoint_history_dir.mkdir(parents=True, exist_ok=True)
        history_path = self.best_checkpoint_history_dir / (
            f"best-round{round_idx:04d}-epoch{epoch:04d}-score{score_tag}.ckpt"
        )
        trainer.save_checkpoint(str(history_path))
        shutil.copyfile(history_path, self.best_checkpoint_latest_path)

        record = {
            "round": round_idx,
            "epoch": epoch,
            "avg_score": float(avg_score),
            "num_pairs": int(num_pairs),
            "validity_rate": validity_stats.get("validity_rate"),
            "checkpoint_path": str(history_path),
            "latest_checkpoint_path": str(self.best_checkpoint_latest_path),
        }
        with open(self.best_checkpoint_index_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

        self.best_checkpoint_score = float(avg_score)
        return record

    def _buffer_gate(self, pairs, validity_stats):
        reasons = []
        if len(pairs) < self.buffer_gate_min_pairs:
            reasons.append(f"num_pairs<{self.buffer_gate_min_pairs}")
        if validity_stats.get("validity_rate", 0.0) < self.buffer_gate_min_validity_rate:
            reasons.append(f"validity_rate<{self.buffer_gate_min_validity_rate:.1%}")
        if (
            self.buffer_gate_require_zero_score_failures
            and validity_stats.get("score_failed_count", 0) > 0
        ):
            reasons.append("score_failed_count>0")
        return len(reasons) == 0, "; ".join(reasons) if reasons else "passed"

    def _protective_stop(self, pairs, validity_stats):
        if validity_stats.get("validity_rate", 0.0) < self.protect_stop_validity_rate:
            self.unhealthy_round_streak += 1
        else:
            self.unhealthy_round_streak = 0

        if self.protect_stop_on_zero_winner_group and validity_stats.get("zero_winner_group_count", 0) > 0:
            return "zero_winner_group_count>0"
        if len(pairs) < self.protect_stop_min_pairs:
            return f"num_pairs<{self.protect_stop_min_pairs}"
        if (
            self.protect_stop_patience_rounds > 0
            and self.unhealthy_round_streak >= self.protect_stop_patience_rounds
        ):
            return (
                f"validity_rate<{self.protect_stop_validity_rate:.1%} "
                f"for {self.unhealthy_round_streak} rounds"
            )
        return None

    def _maybe_update_ref_model(self, pl_module, avg_score):
        ref_model_updated = False
        if not self.iterative_dpo_enabled:
            return ref_model_updated

        self.rounds_since_ref_update += 1
        should_force_update = (
            self.force_update_every_n_rounds > 0
            and self.rounds_since_ref_update >= self.force_update_every_n_rounds
        )
        if avg_score > self.best_score + self.score_threshold or should_force_update:
            pl_module.ref_model.load_state_dict(pl_module.model.state_dict())
            pl_module._freeze_ref_model()
            self.best_score = max(self.best_score, avg_score)
            self.rounds_since_ref_update = 0
            ref_model_updated = True
        return ref_model_updated

    def _write_metrics(
        self,
        epoch,
        num_pairs,
        avg_score,
        validity_stats,
        training_metrics,
        ref_model_updated,
        buffer_gate_passed,
        buffer_gate_reason,
        protective_stop_reason,
        best_checkpoint_record=None,
        sampling_error=None,
        pairs=None,
    ):
        score_keys = [
            "sims_surf_target",
            "sims_esp_target",
            "sims_pharm_target",
            "total_score",
            "sa_score",
            "sa_fallback_used",
            "logp",
        ]
        winner_agg = {key: [] for key in score_keys}
        loser_agg = {key: [] for key in score_keys}

        for pair in (pairs or []):
            winner_scores, loser_scores = pair[3], pair[4]
            for key in score_keys:
                winner_value = winner_scores.get(key)
                if winner_value is not None:
                    winner_value = float(winner_value)
                    if np.isfinite(winner_value):
                        winner_agg[key].append(winner_value)

                loser_value = loser_scores.get(key)
                if loser_value is not None:
                    loser_value = float(loser_value)
                    if np.isfinite(loser_value):
                        loser_agg[key].append(loser_value)

        def _safe_mean(values):
            return sum(values) / len(values) if values else None

        winner_avg = {key: _safe_mean(winner_agg[key]) for key in score_keys}
        loser_avg = {key: _safe_mean(loser_agg[key]) for key in score_keys}

        status = "ok"
        if sampling_error is not None:
            status = "error"
        elif num_pairs == 0:
            status = "empty"

        score_gap = None
        if winner_avg["total_score"] is not None and loser_avg["total_score"] is not None:
            score_gap = winner_avg["total_score"] - loser_avg["total_score"]

        record = {
            "round": len(self.metrics),
            "epoch": epoch,
            "status": status,
            "num_pairs": num_pairs,
            "avg_score": None if avg_score == -float("inf") else avg_score,
            "validity_stats": validity_stats,
            "training_metrics": training_metrics,
            "winner": winner_avg,
            "loser": loser_avg,
            "score_gap": score_gap,
            "ref_model_updated": ref_model_updated,
            "best_checkpoint_saved": best_checkpoint_record is not None,
            "best_checkpoint_record": best_checkpoint_record,
            "buffer_gate_passed": buffer_gate_passed,
            "buffer_gate_reason": buffer_gate_reason,
            "protective_stop_reason": protective_stop_reason,
            "sampling_error": sampling_error,
        }

        if self.save_pairs_detail:
            pairs_detail = []
            for pair in (pairs or []):
                reference_idx = int(pair[2])
                winner_scores, loser_scores = pair[3], pair[4]
                detail = {
                    "reference_idx": reference_idx,
                    "winner": {
                        key: float(winner_scores[key])
                        for key in score_keys
                        if winner_scores.get(key) is not None
                    },
                    "loser": {
                        key: float(loser_scores[key])
                        for key in score_keys
                        if loser_scores.get(key) is not None
                    },
                }
                detail["winner"]["charge_method"] = winner_scores.get("charge_method")
                detail["loser"]["charge_method"] = loser_scores.get("charge_method")

                winner_total = winner_scores.get("total_score")
                loser_total = loser_scores.get("total_score")
                if winner_total is not None and loser_total is not None:
                    detail["score_gap"] = float(winner_total) - float(loser_total)
                else:
                    detail["score_gap"] = None
                pairs_detail.append(detail)

            record["pairs_detail"] = pairs_detail

        self.metrics.append(record)
        with open(self.metrics_path, "w", encoding="utf-8") as handle:
            json.dump(self.metrics, handle, ensure_ascii=False, indent=2)

    def on_train_epoch_end(self, trainer, pl_module):
        every_n_epochs = self.params["training"].get("dpo_sampling_every_n_epochs", 3)
        if trainer.current_epoch == 0 or trainer.current_epoch % every_n_epochs != 0:
            return

        training_metrics = getattr(pl_module, "_last_dpo_metrics", {}).copy()

        sampling_error = None
        pairs = []
        avg_score = -float("inf")
        validity_stats = {
            "num_valid": 0,
            "num_total": 0,
            "validity_rate": 0.0,
            "score_failed_count": 0,
            "pairable_count": 0,
            "zero_winner_group_count": 0,
        }

        try:
            self.sampler.sync_weights(pl_module)
            pairs, avg_score, validity_stats = sample_preference_pairs(
                self.sampler,
                self.params,
                self.references,
            )
        except Exception as exc:
            sampling_error = f"{type(exc).__name__}: {exc}"

        buffer_gate_passed, buffer_gate_reason = self._buffer_gate(pairs, validity_stats)
        if buffer_gate_passed and pairs:
            self.dpo_dataset.update_preference_pairs(pairs)

        ref_model_updated = False
        if sampling_error is None and buffer_gate_passed and pairs:
            ref_model_updated = self._maybe_update_ref_model(pl_module, avg_score)

        protective_stop_reason = None
        if sampling_error is None:
            protective_stop_reason = self._protective_stop(pairs, validity_stats)
            if protective_stop_reason is not None:
                trainer.should_stop = True

        best_checkpoint_record = self._maybe_save_best_checkpoint(
            trainer=trainer,
            avg_score=avg_score,
            epoch=trainer.current_epoch,
            num_pairs=len(pairs),
            validity_stats=validity_stats,
            sampling_error=sampling_error,
        )

        self._write_metrics(
            epoch=trainer.current_epoch,
            num_pairs=len(pairs),
            avg_score=avg_score,
            validity_stats=validity_stats,
            training_metrics=training_metrics,
            ref_model_updated=ref_model_updated,
            buffer_gate_passed=buffer_gate_passed,
            buffer_gate_reason=buffer_gate_reason,
            protective_stop_reason=protective_stop_reason,
            best_checkpoint_record=best_checkpoint_record,
            sampling_error=sampling_error,
            pairs=pairs,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", type=str)
    parser.add_argument("seed", type=int)
    args = parser.parse_args()

    seed_everything(seed=args.seed, workers=True)

    script_dir = Path(__file__).resolve().parent
    params = importlib.import_module(f"parameters.{args.model_name}").params
    if params["training"].get("dpo_optimize_x4", True):
        raise ValueError("dpo_optimize_x4 must be False when using shared reference condition")

    reference_molblocks_and_charges = load_reference_molecules(params, script_dir)
    dataset = create_dataset(params, reference_molblocks_and_charges)
    reference_cache = build_reference_cache(reference_molblocks_and_charges)

    dpo_dataset = DPODataset(
        preference_pairs=[],
        base_dataset=dataset,
        noise_schedule_dict=params["noise_schedules"],
        params=params,
        reference_cache=reference_cache,
    )

    output_dir = script_dir / "jobs" / params["training"]["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    model_pl, resume_ckpt_path = load_model(params, output_dir, script_dir)

    train_device = (
        torch.device("cuda", 0)
        if torch.cuda.is_available() and params["training"]["num_gpus"] >= 1
        else torch.device("cpu")
    )
    model_pl.to(train_device)
    model_pl.model.device = train_device

    cuda_available = torch.cuda.is_available()
    if cuda_available:
        visible_gpu_count = torch.cuda.device_count()
        raw_sampling_gpu_ids = params["training"].get(
            "sampling_gpu_ids",
            list(range(visible_gpu_count)),
        )
        sampling_gpu_ids = []
        for gpu_id in raw_sampling_gpu_ids:
            gpu_id = int(gpu_id)
            if 0 <= gpu_id < visible_gpu_count:
                sampling_gpu_ids.append(gpu_id)
    else:
        sampling_gpu_ids = []

    if len(sampling_gpu_ids) > 1:
        sampler = MultiGPUOnlineSampler(
            model_pl,
            params,
            [torch.device("cuda", gpu_id) for gpu_id in sampling_gpu_ids],
        )
    elif len(sampling_gpu_ids) == 1:
        sampler = OnlineSampler(
            model_pl,
            params,
            device=torch.device("cuda", sampling_gpu_ids[0]),
        )
    else:
        sampler = OnlineSampler(model_pl, params, device=train_device)

    initial_attempts = int(params["training"].get("initial_sampling_attempts", 3))
    initial_pairs = []
    initial_avg_score = -float("inf")
    initial_validity_stats = None
    for _ in range(initial_attempts):
        sampler.sync_weights(model_pl)
        initial_pairs, initial_avg_score, initial_validity_stats = sample_preference_pairs(
            sampler,
            params,
            reference_molblocks_and_charges,
        )
        if initial_pairs:
            break

    if not initial_pairs:
        raise RuntimeError(
            f"Initial DPO sampling produced no preference pairs. last_validity_stats={initial_validity_stats}"
        )

    dpo_dataset.update_preference_pairs(initial_pairs)

    train_loader = create_mixed_dataloader(
        standard_dataset=dataset,
        dpo_dataset=dpo_dataset,
        batch_size=params["training"]["batch_size"],
        num_workers=params["training"]["num_workers"],
        real_data_ratio=params["training"].get("real_data_ratio", 0.5),
        shuffle=True,
        multiprocessing_spawn=params["training"].get("multiprocessing_spawn", False),
        worker_init_fn=set_worker_sharing_strategy,
    )

    checkpoint_callback = ModelCheckpoint(
        save_top_k=0,
        save_last=True,
        monitor="train_loss",
        mode="min",
        dirpath=str(output_dir),
        filename="epoch-{epoch:03d}",
        every_n_epochs=params["training"].get("checkpoint_every_n_epochs", 1),
    )
    sampling_callback = DPOSamplingCallback(
        params=params,
        references=reference_molblocks_and_charges,
        dpo_dataset=dpo_dataset,
        output_dir=output_dir,
        sampler=sampler,
    )
    sampling_callback._write_metrics(
        epoch=0,
        num_pairs=len(initial_pairs),
        avg_score=initial_avg_score,
        validity_stats=initial_validity_stats,
        training_metrics={},
        ref_model_updated=False,
        buffer_gate_passed=True,
        buffer_gate_reason="initial_sampling",
        protective_stop_reason=None,
        pairs=initial_pairs,
    )

    csv_logger = CSVLogger(save_dir=str(output_dir), name="csv_logger")

    trainer = pl.Trainer(
        callbacks=[checkpoint_callback, sampling_callback],
        logger=[csv_logger],
        default_root_dir=str(output_dir),
        accelerator="gpu" if cuda_available and params["training"]["num_gpus"] >= 1 else "cpu",
        devices=1 if cuda_available and params["training"]["num_gpus"] >= 1 else "auto",
        max_epochs=10000,
        gradient_clip_val=params["training"]["gradient_clip_val"],
        accumulate_grad_batches=params["training"]["accumulate_grad_batches"],
        log_every_n_steps=params["training"]["log_every_n_steps"],
        reload_dataloaders_every_n_epochs=1,
        precision=32,
        detect_anomaly=True,
    )

    if resume_ckpt_path is not None and trainer.global_rank == 0:
        timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
        backup_path = output_dir / f"last_{timestamp}.ckpt"
        if not backup_path.exists():
            shutil.copyfile(output_dir / "last.ckpt", backup_path)

    trainer.fit(model_pl, train_loader, ckpt_path=resume_ckpt_path)


if __name__ == "__main__":
    main()
