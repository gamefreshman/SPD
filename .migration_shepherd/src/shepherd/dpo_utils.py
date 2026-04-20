from concurrent.futures import ThreadPoolExecutor
import sys
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import torch

from shepherd.extract import create_rdkit_molecule
from shepherd.inference.sampler import generate
from shepherd.shepherd_score_utils.conformer_generation import update_mol_coordinates
from shepherd.shepherd_score_utils.generate_point_cloud import (
    get_atomic_vdw_radii,
    get_electrostatics_given_point_charges,
    get_molecular_surface,
)
from shepherd.shepherd_score_utils.pharm_utils.pharmacophore import get_pharmacophores


def _ensure_shepherd_score_importable():
    try:
        import shepherd_score  # noqa: F401
        return
    except ModuleNotFoundError:
        repo_root = Path(__file__).resolve().parents[3]
        candidates = [
            repo_root.parent / "shepherd-score",
            repo_root.parent / "SPD" / "src" / "score",
        ]
        for candidate in candidates:
            if (candidate / "shepherd_score").exists():
                sys.path.insert(0, str(candidate))
                try:
                    import shepherd_score  # noqa: F401
                    return
                except ModuleNotFoundError:
                    continue
    raise ImportError(
        "DPO scoring requires the `shepherd_score` package. Install it or expose a "
        "`shepherd_score` source tree on PYTHONPATH before running dpo_train.py."
    )


def _resolve_log_value(value, default):
    if value is None:
        return default
    return float(value)


def build_sample_signature(sample_item, decimals=3):
    def signature(values):
        if values is None:
            return ()
        array = np.asarray(values)
        if array.size == 0:
            return ()
        if np.issubdtype(array.dtype, np.floating):
            array = np.round(array.astype(np.float64), decimals=decimals)
        return tuple(array.reshape(-1).tolist())

    return (
        signature(sample_item["sample"]["x1"].get("atoms")),
        signature(sample_item["sample"]["x1"].get("positions")),
        signature(sample_item["sample"]["x1"].get("bonds")),
    )


class OnlineSampler:
    def __init__(self, model_pl, params, device=None):
        self.model_pl = model_pl
        self.params = params
        self.device = device
        self.num_samples_per_molecule = params.get("sampling", {}).get(
            "num_samples_per_molecule",
            8,
        )
        self.sub_batch_size = params.get("sampling", {}).get("inference_sub_batch_size", 4)
        self.fixed_n_atoms = params.get("sampling", {}).get("fixed_n_atoms")

    def _prepare_condition(self, mol_block, charges, mol_index):
        mol = Chem.MolFromMolBlock(mol_block, removeHs=False)
        if mol is None:
            raise ValueError(f"Failed to parse reference molecule at index {mol_index}")

        charges = np.asarray(charges, dtype=float)
        mol_coordinates = np.array(mol.GetConformer().GetPositions())
        mol_coordinates = mol_coordinates - np.mean(mol_coordinates, axis=0)
        mol = update_mol_coordinates(mol, mol_coordinates)

        centers = mol.GetConformer().GetPositions()
        radii = get_atomic_vdw_radii(mol)
        surface = get_molecular_surface(
            centers,
            radii,
            self.params["dataset"]["x3"]["num_points"],
            probe_radius=self.params["dataset"]["probe_radius"],
            num_samples_per_atom=20,
        )
        pharm_types, pharm_pos, pharm_direction = get_pharmacophores(
            mol,
            multi_vector=self.params["dataset"]["x4"]["multivectors"],
            check_access=self.params["dataset"]["x4"]["check_accessibility"],
        )
        electrostatics = get_electrostatics_given_point_charges(charges, centers, surface)

        return {
            "mol_index": mol_index,
            "mol": mol,
            "surface": surface,
            "electrostatics": electrostatics,
            "pharm_types": pharm_types,
            "pharm_pos": pharm_pos,
            "pharm_direction": pharm_direction,
            "num_pharmacophores": int(len(pharm_types)),
        }

    def sync_weights(self, model_pl):
        self.model_pl = model_pl

    def _generate_samples(self, model_pl, condition, batch_size, n_atoms):
        with torch.no_grad():
            return generate(
                model_pl=model_pl,
                batch_size=batch_size,
                N_x1=int(n_atoms),
                N_x4=max(1, condition["num_pharmacophores"]),
                unconditional=False,
                prior_noise_scale=1.0,
                denoising_noise_scale=1.0,
                inject_noise_at_ts=[],
                inject_noise_scales=[],
                harmonize=False,
                harmonize_ts=[],
                harmonize_jumps=[],
                inpaint_x2_pos=False,
                inpaint_x3_pos=True,
                inpaint_x3_x=True,
                inpaint_x4_pos=True,
                inpaint_x4_direction=True,
                inpaint_x4_type=True,
                stop_inpainting_at_time_x2=0.0,
                add_noise_to_inpainted_x2_pos=0.0,
                stop_inpainting_at_time_x3=0.0,
                add_noise_to_inpainted_x3_pos=0.0,
                add_noise_to_inpainted_x3_x=0.0,
                stop_inpainting_at_time_x4=0.0,
                add_noise_to_inpainted_x4_pos=0.0,
                add_noise_to_inpainted_x4_direction=0.0,
                add_noise_to_inpainted_x4_type=0.0,
                center_of_mass=np.zeros(3),
                surface=condition["surface"],
                electrostatics=condition["electrostatics"],
                pharm_types=condition["pharm_types"],
                pharm_pos=condition["pharm_pos"],
                pharm_direction=condition["pharm_direction"],
                verbose=False,
            )

    def sample(self, reference_molblocks_and_charges):
        was_training = self.model_pl.training
        self.model_pl.eval()
        if self.device is not None:
            self.model_pl.to(self.device)
            self.model_pl.model.device = self.device

        generated_samples = []
        reference_mols = {}
        global_group_id = 0

        try:
            for mol_index, (mol_block, charges) in enumerate(reference_molblocks_and_charges):
                condition = self._prepare_condition(mol_block, charges, mol_index)
                reference_mols[mol_index] = condition["mol"]
                n_atoms = self.fixed_n_atoms or condition["mol"].GetNumAtoms()
                num_samples = self.num_samples_per_molecule

                for start in range(0, num_samples, self.sub_batch_size):
                    batch_size = min(self.sub_batch_size, num_samples - start)
                    samples = self._generate_samples(self.model_pl, condition, batch_size, n_atoms)
                    for sample in samples:
                        sample["source_mol_index"] = mol_index
                        sample["group_id"] = global_group_id
                    generated_samples.extend(samples)
                    global_group_id += 1
        finally:
            if was_training:
                self.model_pl.train()

        return generated_samples, reference_mols


class _InferenceWrapper:
    def __init__(self, model, params, device):
        self.model = model
        self.params = params
        self.device = device

    def eval(self):
        self.model.eval()
        return self

    @property
    def training(self):
        return self.model.training


class MultiGPUOnlineSampler(OnlineSampler):
    def __init__(self, model_pl, params, devices):
        super().__init__(model_pl=model_pl, params=params, device=None)
        self.devices = list(devices)
        if not self.devices:
            raise ValueError("MultiGPUOnlineSampler requires at least one device")
        self.replicas = [self._make_inference_replica(model_pl, device) for device in self.devices]

    @staticmethod
    def _make_inference_replica(model_pl, device):
        model_copy = deepcopy(model_pl.model)
        model_copy.to(device)
        model_copy.device = device
        model_copy.eval()
        for parameter in model_copy.parameters():
            parameter.requires_grad_(False)
        return _InferenceWrapper(model=model_copy, params=model_pl.params, device=device)

    def sync_weights(self, model_pl):
        self.model_pl = model_pl
        state_dict = model_pl.model.state_dict()
        for replica in self.replicas:
            replica.model.load_state_dict(state_dict)
            replica.model.eval()

    def sample(self, reference_molblocks_and_charges):
        generated_samples = []
        reference_mols = {}
        global_group_id = 0

        for mol_index, (mol_block, charges) in enumerate(reference_molblocks_and_charges):
            condition = self._prepare_condition(mol_block, charges, mol_index)
            reference_mols[mol_index] = condition["mol"]
            n_atoms = self.fixed_n_atoms or condition["mol"].GetNumAtoms()

            tasks = []
            remaining = self.num_samples_per_molecule
            while remaining > 0:
                batch_size = min(self.sub_batch_size, remaining)
                tasks.append(batch_size)
                remaining -= batch_size

            gpu_tasks = [[] for _ in self.devices]
            for task_idx, batch_size in enumerate(tasks):
                gpu_tasks[task_idx % len(self.devices)].append((task_idx, batch_size))

            def worker(replica, assigned_tasks):
                local_results = {}
                for task_idx, batch_size in assigned_tasks:
                    local_results[task_idx] = self._generate_samples(
                        replica,
                        condition,
                        batch_size,
                        n_atoms,
                    )
                return local_results

            results_by_task = {}
            with ThreadPoolExecutor(max_workers=len(self.devices)) as executor:
                futures = []
                for gpu_idx, replica in enumerate(self.replicas):
                    assigned_tasks = gpu_tasks[gpu_idx]
                    if assigned_tasks:
                        futures.append(executor.submit(worker, replica, assigned_tasks))
                for future in futures:
                    results_by_task.update(future.result())

            for task_idx in range(len(tasks)):
                samples = results_by_task[task_idx]
                for sample in samples:
                    sample["source_mol_index"] = mol_index
                    sample["group_id"] = global_group_id
                generated_samples.extend(samples)
                global_group_id += 1

        return generated_samples, reference_mols


class ShepherdScorer:
    def __init__(self, params):
        _ensure_shepherd_score_importable()

        from shepherd_score.container import Molecule
        from shepherd_score.evaluations.evaluate import ConfEval
        from shepherd_score.score.constants import ALPHA, LAM_SCALING
        from shepherd_score.score.electrostatic_scoring_np import get_overlap_esp_np
        from shepherd_score.score.gaussian_overlap_np import get_overlap_np
        from shepherd_score.score.pharmacophore_scoring_np import get_overlap_pharm_np

        self.Molecule = Molecule
        self.ConfEval = ConfEval
        self.ALPHA = ALPHA
        self.LAM_SCALING = LAM_SCALING
        self.get_overlap_np = get_overlap_np
        self.get_overlap_esp_np = get_overlap_esp_np
        self.get_overlap_pharm_np = get_overlap_pharm_np

        self.params = params
        self.num_surf_points = params.get("dpo", {}).get("num_surf_points", 400)
        self.probe_radius = params.get("dpo", {}).get("probe_radius", 1.2)

    def sanitize_scoring_molecule(self, mol):
        if mol is None:
            return None, "rdkit_mol_missing"
        scoring_mol = Chem.Mol(mol)
        if scoring_mol is None:
            return None, "rdkit_copy_failed"
        try:
            Chem.SanitizeMol(scoring_mol)
        except Exception as exc:
            return None, f"sanitize_failed:{type(exc).__name__}"
        try:
            if (
                scoring_mol.GetNumConformers() > 0
                and not any(atom.GetAtomicNum() == 1 for atom in scoring_mol.GetAtoms())
            ):
                scoring_mol = Chem.AddHs(scoring_mol, addCoords=True)
                Chem.SanitizeMol(scoring_mol)
        except Exception:
            pass
        return scoring_mol, None

    def compute_partial_charges_with_fallback(self, mol):
        try:
            if AllChem.MMFFHasAllMoleculeParams(mol):
                props = AllChem.MMFFGetMoleculeProperties(mol)
                if props is not None:
                    charges = np.array(
                        [props.GetMMFFPartialCharge(i) for i in range(mol.GetNumAtoms())],
                        dtype=float,
                    )
                    return charges, "mmff"
        except Exception:
            pass

        try:
            AllChem.ComputeGasteigerCharges(mol)
            charges = np.array(
                [atom.GetDoubleProp("_GasteigerCharge") for atom in mol.GetAtoms()],
                dtype=float,
            )
            return np.nan_to_num(charges, nan=0.0), "gasteiger"
        except Exception:
            return None, None

    def build_reference(self, mol):
        scoring_mol, sanitize_error = self.sanitize_scoring_molecule(mol)
        if scoring_mol is None:
            raise ValueError(sanitize_error or "reference_sanitize_failed")
        partial_charges, charge_method = self.compute_partial_charges_with_fallback(scoring_mol)
        if partial_charges is None:
            raise ValueError("reference_partial_charge_unavailable")
        reference = self.Molecule(
            scoring_mol,
            num_surf_points=self.num_surf_points,
            probe_radius=self.probe_radius,
            partial_charges=partial_charges,
            pharm_multi_vector=False,
        )
        return {
            "molecule": reference,
            "charge_method": charge_method,
            "alpha": self.ALPHA(self.num_surf_points),
            "lam_scaled": 0.3 * self.LAM_SCALING,
        }

    def score_sample(self, sample, reference_bundle):
        atoms = np.asarray(sample["x1"]["atoms"])
        positions = np.asarray(sample["x1"]["positions"])
        bonds = sample["x1"].get("bonds", None)

        if atoms.size == 0:
            return None, {"is_valid": False, "reason": "no_atoms"}

        conf_eval = self.ConfEval(atoms, positions, solvent="water", bonds=bonds)
        if not conf_eval.is_valid:
            return None, {"is_valid": False, "reason": "conf_eval_invalid"}

        scoring_mol, sanitize_error = self.sanitize_scoring_molecule(conf_eval.mol)
        if scoring_mol is None:
            return None, {"is_valid": False, "reason": sanitize_error}

        partial_charges, charge_method = self.compute_partial_charges_with_fallback(scoring_mol)
        if partial_charges is None:
            return None, {"is_valid": False, "reason": "partial_charge_unavailable"}

        generated = self.Molecule(
            scoring_mol,
            num_surf_points=self.num_surf_points,
            probe_radius=self.probe_radius,
            partial_charges=partial_charges,
            pharm_multi_vector=False,
        )
        reference = reference_bundle["molecule"]

        sims_surf_target = 0.0
        if generated.surf_pos is not None and reference.surf_pos is not None:
            sims_surf_target = float(
                self.get_overlap_np(
                    generated.surf_pos,
                    reference.surf_pos,
                    alpha=reference_bundle["alpha"],
                )
            )

        sims_esp_target = 0.0
        if (
            generated.surf_pos is not None
            and generated.surf_esp is not None
            and reference.surf_pos is not None
            and reference.surf_esp is not None
        ):
            sims_esp_target = float(
                self.get_overlap_esp_np(
                    generated.surf_pos,
                    reference.surf_pos,
                    generated.surf_esp,
                    reference.surf_esp,
                    alpha=reference_bundle["alpha"],
                    lam=reference_bundle["lam_scaled"],
                )
            )

        sims_pharm_target = 0.0
        if (
            generated.pharm_ancs is not None
            and reference.pharm_ancs is not None
            and len(generated.pharm_ancs) > 0
            and len(reference.pharm_ancs) > 0
        ):
            sims_pharm_target = float(
                self.get_overlap_pharm_np(
                    generated.pharm_types,
                    reference.pharm_types,
                    generated.pharm_ancs,
                    reference.pharm_ancs,
                    generated.pharm_vecs,
                    reference.pharm_vecs,
                    similarity="tanimoto",
                    extended_points=False,
                    only_extended=False,
                )
            )

        sa_score = _resolve_log_value(
            getattr(conf_eval, "SA_score", getattr(conf_eval, "sa_score", None)),
            5.0,
        )
        logp = _resolve_log_value(
            getattr(conf_eval, "logP", getattr(conf_eval, "logp", None)),
            2.5,
        )
        sa_normalized = (sa_score - 1.0) / 9.0
        total_score = (
            sims_surf_target * 1.0
            + sims_esp_target * 3.0
            + sims_pharm_target * 3.0
            - sa_normalized * 1.5
            + 2.0
        )

        scores = {
            "is_valid": True,
            "sa_score": sa_score,
            "logp": logp,
            "sims_surf_target": float(np.nan_to_num(sims_surf_target)),
            "sims_esp_target": float(np.nan_to_num(sims_esp_target)),
            "sims_pharm_target": float(np.nan_to_num(sims_pharm_target)),
            "total_score": float(np.nan_to_num(total_score)),
            "charge_method": charge_method,
        }
        return scoring_mol, scores


class PreferencePairBuilder:
    def __init__(self, params, scorer=None):
        self.params = params
        self.training_cfg = params.get("training", {})
        self.scorer = scorer or ShepherdScorer(params)
        self.min_score_gap = params.get("dpo", {}).get(
            "min_score_gap",
            self.training_cfg.get("dpo_min_score_gap", 0.15),
        )
        self.top_k_winners = max(1, int(self.training_cfg.get("dpo_top_k_winners", 2)))
        self.max_losers_per_winner = max(
            1,
            int(self.training_cfg.get("dpo_max_losers_per_winner", 2)),
        )
        self.max_pair_repeats_per_molecule = max(
            1,
            int(self.training_cfg.get("dpo_max_pair_repeats_per_molecule", 2)),
        )

    def evaluate_and_build_pairs(self, generated_samples, reference_mols):
        grouped_samples = defaultdict(list)
        for sample in generated_samples:
            grouped_samples[sample.get("group_id", sample.get("source_mol_index", 0))].append(sample)

        all_pairs = []
        all_scores = []
        pairable_count = 0
        zero_winner_group_count = 0
        score_failed_count = 0
        valid_count = 0
        total_count = len(generated_samples)
        seen_pairs = set()
        repeat_counts = defaultdict(int)

        reference_cache = {}
        for idx, mol in reference_mols.items():
            try:
                reference_cache[idx] = self.scorer.build_reference(mol)
            except Exception:
                continue

        for group_id, samples in grouped_samples.items():
            scored_items = []
            reference_idx = samples[0].get("source_mol_index", 0)
            reference_bundle = reference_cache.get(reference_idx)
            if reference_bundle is None:
                zero_winner_group_count += 1
                score_failed_count += len(samples)
                continue

            for sample in samples:
                try:
                    _, scores = self.scorer.score_sample(sample, reference_bundle)
                except Exception:
                    scores = {"is_valid": False, "reason": "scoring_exception"}
                if not scores.get("is_valid", False):
                    score_failed_count += 1
                    continue

                valid_count += 1
                item = {"sample": sample, "scores": scores}
                item["sample_signature"] = build_sample_signature(item)
                scored_items.append(item)
                all_scores.append(scores["total_score"])

            unique_items = []
            seen_signatures = set()
            for item in sorted(scored_items, key=lambda entry: entry["scores"]["total_score"], reverse=True):
                signature = item["sample_signature"]
                if signature in seen_signatures:
                    continue
                seen_signatures.add(signature)
                unique_items.append(item)

            pairable_count += len(unique_items)
            if len(unique_items) < 2:
                zero_winner_group_count += 1
                continue

            winners = unique_items[: min(self.top_k_winners, len(unique_items) - 1)]
            losers = list(reversed(unique_items[len(winners):] or unique_items[1:]))
            group_pair_count = 0

            for winner in winners:
                loser_count = 0
                for loser in losers:
                    if loser is winner:
                        continue
                    winner_sig = winner["sample_signature"]
                    loser_sig = loser["sample_signature"]
                    if winner_sig == loser_sig:
                        continue
                    if repeat_counts[winner_sig] >= self.max_pair_repeats_per_molecule:
                        continue
                    if repeat_counts[loser_sig] >= self.max_pair_repeats_per_molecule:
                        continue

                    score_gap = winner["scores"]["total_score"] - loser["scores"]["total_score"]
                    if score_gap < self.min_score_gap:
                        continue

                    pair_key = (reference_idx, winner_sig, loser_sig)
                    if pair_key in seen_pairs:
                        continue

                    winner_mol = create_rdkit_molecule(winner["sample"])
                    loser_mol = create_rdkit_molecule(loser["sample"])
                    if winner_mol is None or loser_mol is None:
                        continue

                    all_pairs.append(
                        (
                            winner_mol,
                            loser_mol,
                            int(reference_idx),
                            deepcopy(winner["scores"]),
                            deepcopy(loser["scores"]),
                        )
                    )
                    seen_pairs.add(pair_key)
                    repeat_counts[winner_sig] += 1
                    repeat_counts[loser_sig] += 1
                    group_pair_count += 1
                    loser_count += 1
                    if loser_count >= self.max_losers_per_winner:
                        break

            if group_pair_count == 0:
                zero_winner_group_count += 1

        avg_score = float(np.mean(all_scores)) if all_scores else -float("inf")
        validity_rate = valid_count / total_count if total_count > 0 else 0.0
        validity_stats = {
            "num_valid": valid_count,
            "num_total": total_count,
            "validity_rate": validity_rate,
            "chemical_valid_count": valid_count,
            "chemical_invalid_count": total_count - valid_count,
            "score_failed_count": score_failed_count,
            "pairable_count": pairable_count,
            "zero_winner_group_count": zero_winner_group_count,
            "num_scoreable": valid_count,
            "scorable_rate": validity_rate,
        }
        return all_pairs, avg_score, validity_stats
