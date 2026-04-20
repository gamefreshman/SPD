from contextlib import contextmanager
from copy import deepcopy
from typing import List, Tuple

import numpy as np
import rdkit.Chem as Chem
import torch
import torch_geometric
from rdkit.Chem import AllChem
from torch_geometric.data import Batch, Dataset, HeteroData

from shepherd.datasets import get_atomic_partial_charges
from shepherd.shepherd_score_utils.conformer_generation import update_mol_coordinates
from shepherd.shepherd_score_utils.generate_point_cloud import get_atomic_vdw_radii


@contextmanager
def _temporary_numpy_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


class DPODataset(Dataset):
    def __init__(
        self,
        preference_pairs: List[Tuple],
        base_dataset,
        noise_schedule_dict,
        params,
        reference_cache=None,
    ):
        super().__init__()
        self.preference_pairs = preference_pairs
        self.base_dataset = base_dataset
        self.noise_schedule_dict = noise_schedule_dict
        self.params = params
        self.reference_cache = dict(reference_cache or {})

    def len(self):
        return len(self.preference_pairs)

    def update_preference_pairs(self, new_pairs):
        self.preference_pairs = list(new_pairs)
        self._indices = None

    def update_pairs(self, new_pairs):
        self.update_preference_pairs(new_pairs)

    def get(self, idx):
        pair = self.preference_pairs[idx]
        if self._has_reference_condition(pair):
            return self._get_shared_reference_batch(idx, pair)
        return self._get_legacy_batch(idx, pair)

    def _has_reference_condition(self, pair):
        return len(pair) >= 5 and isinstance(pair[2], (int, np.integer))

    def _sample_timestep(self):
        schedule_key = "x1" if "x1" in self.noise_schedule_dict else next(iter(self.noise_schedule_dict))
        ts = self.noise_schedule_dict[schedule_key]["ts"]
        t_idx = int(np.random.randint(0, len(ts)))
        shared_timestep = torch.tensor([float(ts[t_idx])], dtype=torch.float32)
        return t_idx, shared_timestep

    def _get_legacy_batch(self, idx, pair):
        winner_mol, loser_mol, *score_payload = pair
        t_idx, shared_timestep = self._sample_timestep()
        shared_seed = int(np.random.randint(0, 2**31 - 1))

        winner_data = self._mol_to_hetero_data(winner_mol, t_idx, shared_seed, molecule_id=idx * 2)
        loser_data = self._mol_to_hetero_data(loser_mol, t_idx, shared_seed, molecule_id=idx * 2 + 1)

        batch = {
            "batch_type": "dpo",
            "winner": winner_data,
            "loser": loser_data,
            "shared_timestep": shared_timestep,
        }
        if len(score_payload) >= 2:
            batch["winner_score"] = score_payload[0]
            batch["loser_score"] = score_payload[1]
        return batch

    def _get_shared_reference_batch(self, idx, pair):
        winner_mol, loser_mol, reference_idx, *score_payload = pair
        reference_idx = int(reference_idx)
        if reference_idx not in self.reference_cache:
            raise KeyError(f"Missing reference cache entry for index {reference_idx}")

        ref = self.reference_cache[reference_idx]
        t_idx, shared_timestep = self._sample_timestep()

        x1_schedule = self._schedule_params("x1", t_idx)
        ts_x1 = self.noise_schedule_dict["x1"]["ts"]
        t_x1 = x1_schedule["t"]

        x2_schedule = None
        if self.base_dataset.x2:
            x2_schedule = self._resolve_timestep("x2", t_idx, ts_x1, t_x1)
        x3_schedule = None
        if self.base_dataset.x3:
            x3_schedule = self._resolve_timestep("x3", t_idx, ts_x1, t_x1)
        x4_schedule = None
        if self.base_dataset.x4:
            x4_schedule = self._resolve_timestep("x4", t_idx, ts_x1, t_x1)

        condition_seed = int(np.random.randint(0, 2**31 - 1))
        winner_seed = int(np.random.randint(0, 2**31 - 1))
        loser_seed = int(np.random.randint(0, 2**31 - 1))

        shared_x2 = None
        shared_x3 = None
        shared_x4 = None
        with _temporary_numpy_seed(condition_seed):
            if self.base_dataset.x2:
                shared_x2 = self._build_x2_from_ref(ref["mol_coordinates"], ref["radii"], x2_schedule)
            if self.base_dataset.x3:
                shared_x3 = self._build_x3_from_ref(
                    ref["mol_coordinates"],
                    ref["radii"],
                    ref["charges"],
                    x3_schedule,
                )
            if self.base_dataset.x4:
                shared_x4 = self._build_x4_from_ref(ref["mol"], ref["mol_coordinates"], x4_schedule)

        winner_bundle = self._prepare_molecule_bundle(winner_mol, winner_seed)
        loser_bundle = self._prepare_molecule_bundle(loser_mol, loser_seed)

        with _temporary_numpy_seed(winner_seed):
            winner_x1, _, _ = self._build_x1(winner_bundle["mol"], t_idx)
        with _temporary_numpy_seed(loser_seed):
            loser_x1, _, _ = self._build_x1(loser_bundle["mol"], t_idx)

        winner_dict = {
            "molecule_id": torch.tensor([idx * 2], dtype=torch.long),
            "x1": winner_x1,
        }
        loser_dict = {
            "molecule_id": torch.tensor([idx * 2 + 1], dtype=torch.long),
            "x1": loser_x1,
        }
        if shared_x2 is not None:
            winner_dict["x2"] = deepcopy(shared_x2)
            loser_dict["x2"] = deepcopy(shared_x2)
        if shared_x3 is not None:
            winner_dict["x3"] = deepcopy(shared_x3)
            loser_dict["x3"] = deepcopy(shared_x3)
        if shared_x4 is not None:
            winner_dict["x4"] = deepcopy(shared_x4)
            loser_dict["x4"] = deepcopy(shared_x4)

        winner_data = self._to_heterodata(winner_dict)
        loser_data = self._to_heterodata(loser_dict)

        batch = {
            "batch_type": "dpo",
            "winner": winner_data,
            "loser": loser_data,
            "shared_timestep": torch.tensor([float(t_x1)], dtype=torch.float32),
        }
        if len(score_payload) >= 2:
            batch["winner_score"] = score_payload[0]
            batch["loser_score"] = score_payload[1]
        return batch

    def _prepare_molecule_bundle(self, mol, seed=None):
        if mol is None:
            return None

        mol = Chem.Mol(mol)
        if mol.GetNumConformers() == 0:
            random_seed = int(seed) if seed is not None else -1
            AllChem.EmbedMolecule(mol, randomSeed=random_seed)

        mol_coordinates = np.array(mol.GetConformer().GetPositions())
        mol_coordinates = mol_coordinates - np.mean(mol_coordinates, axis=0)
        mol = update_mol_coordinates(mol, mol_coordinates)

        return {
            "mol": mol,
            "charges": self._get_partial_charges(mol),
            "mol_coordinates": mol_coordinates,
            "radii": get_atomic_vdw_radii(mol),
        }

    def _mol_to_hetero_data(self, mol, t_idx, seed, molecule_id=0):
        if mol is None:
            return HeteroData()

        mol_bundle = self._prepare_molecule_bundle(mol, seed=seed)
        mol = mol_bundle["mol"]
        charges = mol_bundle["charges"]
        mol_coordinates = mol_bundle["mol_coordinates"]
        radii = mol_bundle["radii"]

        with _temporary_numpy_seed(seed):
            data_dict = {
                "molecule_id": torch.tensor([molecule_id], dtype=torch.long),
            }

            x1_pos = None
            x1_virtual_node_mask = None
            ts_x1 = None
            t_x1 = None

            if self.base_dataset.x1:
                x1_data, x1_pos, x1_virtual_node_mask = self._build_x1(mol, t_idx)
                data_dict["x1"] = x1_data
                ts_x1 = self.noise_schedule_dict["x1"]["ts"]
                t_x1 = ts_x1[t_idx]

            if self.base_dataset.x2:
                data_dict["x2"] = self._build_x2(
                    mol_coordinates,
                    radii,
                    x1_pos,
                    x1_virtual_node_mask,
                    t_idx,
                    ts_x1,
                    t_x1,
                )

            if self.base_dataset.x3:
                data_dict["x3"] = self._build_x3(
                    mol_coordinates,
                    radii,
                    charges,
                    x1_pos,
                    x1_virtual_node_mask,
                    t_idx,
                    ts_x1,
                    t_x1,
                )

            if self.base_dataset.x4:
                data_dict["x4"] = self._build_x4(
                    mol,
                    mol_coordinates,
                    x1_pos,
                    x1_virtual_node_mask,
                    t_idx,
                    ts_x1,
                    t_x1,
                )

        return self._to_heterodata(data_dict)

    def _get_partial_charges(self, mol):
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

    def _schedule_params(self, key, t_idx):
        schedule = self.noise_schedule_dict[key]
        return {
            "t": float(schedule["ts"][t_idx]),
            "alpha_t": float(schedule["alpha_ts"][t_idx]),
            "sigma_t": float(schedule["sigma_ts"][t_idx]),
            "alpha_dash_t": float(schedule["alpha_dash_ts"][t_idx]),
            "sigma_dash_t": float(schedule["sigma_dash_ts"][t_idx]),
        }

    def _attach_schedule_tensors(self, data, schedule):
        data["timestep"] = torch.tensor([schedule["t"]], dtype=torch.float32)
        data["alpha_t"] = torch.tensor([schedule["alpha_t"]], dtype=torch.float32)
        data["sigma_t"] = torch.tensor([schedule["sigma_t"]], dtype=torch.float32)
        data["alpha_dash_t"] = torch.tensor([schedule["alpha_dash_t"]], dtype=torch.float32)
        data["sigma_dash_t"] = torch.tensor([schedule["sigma_dash_t"]], dtype=torch.float32)
        return data

    def _build_x1(self, mol, t_idx):
        schedule = self._schedule_params("x1", t_idx)
        x1_data, x1_pos, x1_virtual_node_mask = self.base_dataset.get_x1_data(
            mol,
            schedule["t"],
            schedule["alpha_dash_t"],
            schedule["sigma_dash_t"],
        )
        return self._attach_schedule_tensors(x1_data, schedule), x1_pos, x1_virtual_node_mask

    def _resolve_timestep(self, key, t_idx, ts_x1, t_x1):
        if getattr(self.base_dataset, f"independent_timesteps_{key}", False):
            return self._schedule_params(key, t_idx)
        if ts_x1 is None or t_x1 is None:
            return self._schedule_params(key, t_idx)
        local_t_idx = int(np.where(self.noise_schedule_dict[key]["ts"] == t_x1)[0][0])
        return self._schedule_params(key, local_t_idx)

    def _build_x2(self, mol_coordinates, radii, x1_pos, x1_virtual_node_mask, t_idx, ts_x1, t_x1):
        schedule = self._resolve_timestep("x2", t_idx, ts_x1, t_x1)
        if x1_pos is not None:
            atom_centers = x1_pos[~x1_virtual_node_mask, :]
            virtual_node_pos = (
                atom_centers.mean(0)[None, ...]
                if self.base_dataset.add_virtual_node_x2 and not self.base_dataset.recenter_x2
                else None
            )
        else:
            atom_centers = mol_coordinates
            virtual_node_pos = None

        x2_data, _, _ = self.base_dataset.get_x2_data(
            radii,
            atom_centers,
            self.base_dataset.num_points_x2,
            self.base_dataset.recenter_x2,
            self.base_dataset.add_virtual_node_x2,
            self.base_dataset.remove_noise_COM_x2,
            schedule["t"],
            schedule["alpha_dash_t"],
            schedule["sigma_dash_t"],
            virtual_node_pos=virtual_node_pos,
        )
        return self._attach_schedule_tensors(x2_data, schedule)

    def _build_x2_from_ref(self, ref_mol_coordinates, ref_radii, x2_schedule):
        atom_centers = ref_mol_coordinates
        virtual_node_pos = (
            atom_centers.mean(0)[None, ...]
            if self.base_dataset.add_virtual_node_x2 and not self.base_dataset.recenter_x2
            else None
        )
        x2_data, _, _ = self.base_dataset.get_x2_data(
            ref_radii,
            atom_centers,
            self.base_dataset.num_points_x2,
            self.base_dataset.recenter_x2,
            self.base_dataset.add_virtual_node_x2,
            self.base_dataset.remove_noise_COM_x2,
            x2_schedule["t"],
            x2_schedule["alpha_dash_t"],
            x2_schedule["sigma_dash_t"],
            virtual_node_pos=virtual_node_pos,
        )
        return self._attach_schedule_tensors(x2_data, x2_schedule)

    def _build_x3(self, mol_coordinates, radii, charges, x1_pos, x1_virtual_node_mask, t_idx, ts_x1, t_x1):
        schedule = self._resolve_timestep("x3", t_idx, ts_x1, t_x1)
        if x1_pos is not None:
            atom_centers = x1_pos[~x1_virtual_node_mask, :]
            virtual_node_pos = (
                atom_centers.mean(0)[None, ...]
                if self.base_dataset.add_virtual_node_x3 and not self.base_dataset.recenter_x3
                else None
            )
        else:
            atom_centers = mol_coordinates
            virtual_node_pos = None

        x3_data, x3_pos, x3_virtual_node_mask = self.base_dataset.get_x2_data(
            radii,
            atom_centers,
            self.base_dataset.num_points_x3,
            self.base_dataset.recenter_x3,
            self.base_dataset.add_virtual_node_x3,
            self.base_dataset.remove_noise_COM_x3,
            schedule["t"],
            schedule["alpha_dash_t"],
            schedule["sigma_dash_t"],
            virtual_node_pos=virtual_node_pos,
        )
        x3_com_displacement = (
            x3_data["com"] - x3_data["com_before_centering"]
        ).detach().cpu().numpy()
        charge_centers = atom_centers + x3_com_displacement
        x3_data = self.base_dataset.get_x3_data_electrostatics_only(
            charges,
            charge_centers,
            x3_data,
            x3_pos,
            x3_virtual_node_mask,
            schedule["t"],
            schedule["alpha_dash_t"],
            schedule["sigma_dash_t"],
        )
        return self._attach_schedule_tensors(x3_data, schedule)

    def _build_x3_from_ref(self, ref_mol_coordinates, ref_radii, ref_charges, x3_schedule):
        atom_centers = ref_mol_coordinates
        virtual_node_pos = (
            atom_centers.mean(0)[None, ...]
            if self.base_dataset.add_virtual_node_x3 and not self.base_dataset.recenter_x3
            else None
        )
        x3_data, x3_pos, x3_virtual_node_mask = self.base_dataset.get_x2_data(
            ref_radii,
            atom_centers,
            self.base_dataset.num_points_x3,
            self.base_dataset.recenter_x3,
            self.base_dataset.add_virtual_node_x3,
            self.base_dataset.remove_noise_COM_x3,
            x3_schedule["t"],
            x3_schedule["alpha_dash_t"],
            x3_schedule["sigma_dash_t"],
            virtual_node_pos=virtual_node_pos,
        )
        x3_com_displacement = (
            x3_data["com"] - x3_data["com_before_centering"]
        ).detach().cpu().numpy()
        charge_centers = atom_centers + x3_com_displacement
        x3_data = self.base_dataset.get_x3_data_electrostatics_only(
            ref_charges,
            charge_centers,
            x3_data,
            x3_pos,
            x3_virtual_node_mask,
            x3_schedule["t"],
            x3_schedule["alpha_dash_t"],
            x3_schedule["sigma_dash_t"],
        )
        return self._attach_schedule_tensors(x3_data, x3_schedule)

    def _build_x4(self, mol, mol_coordinates, x1_pos, x1_virtual_node_mask, t_idx, ts_x1, t_x1):
        schedule = self._resolve_timestep("x4", t_idx, ts_x1, t_x1)
        if x1_pos is not None:
            atom_centers = x1_pos[~x1_virtual_node_mask, :]
            virtual_node_pos = (
                atom_centers.mean(0)[None, ...]
                if self.base_dataset.add_virtual_node_x4 and not self.base_dataset.recenter_x4
                else None
            )
        else:
            atom_centers = mol_coordinates
            virtual_node_pos = None

        x4_data = self.base_dataset.get_x4_data(
            mol,
            self.base_dataset.recenter_x4,
            self.base_dataset.add_virtual_node_x4,
            self.base_dataset.remove_noise_COM_x4,
            schedule["t"],
            schedule["alpha_dash_t"],
            schedule["sigma_dash_t"],
            virtual_node_pos,
        )
        return self._attach_schedule_tensors(x4_data, schedule)

    def _build_x4_from_ref(self, ref_mol, ref_mol_coordinates, x4_schedule):
        atom_centers = ref_mol_coordinates
        virtual_node_pos = (
            atom_centers.mean(0)[None, ...]
            if self.base_dataset.add_virtual_node_x4 and not self.base_dataset.recenter_x4
            else None
        )
        x4_data = self.base_dataset.get_x4_data(
            ref_mol,
            self.base_dataset.recenter_x4,
            self.base_dataset.add_virtual_node_x4,
            self.base_dataset.remove_noise_COM_x4,
            x4_schedule["t"],
            x4_schedule["alpha_dash_t"],
            x4_schedule["sigma_dash_t"],
            virtual_node_pos,
        )
        return self._attach_schedule_tensors(x4_data, x4_schedule)

    def _to_heterodata(self, data_dict):
        data = torch_geometric.data.HeteroData()
        data.molecule_id = data_dict["molecule_id"]

        if "x1" in data_dict:
            x1_data = data_dict["x1"]
            x1_node_dict = {k: v for k, v in x1_data.items() if "bond" not in k}
            x1_edge_dict = {
                "edge_index": x1_data["bond_edge_index"],
                "mask": x1_data["bond_edge_mask"],
                "x": x1_data["bond_edge_x"],
                "x_noise": x1_data["bond_edge_x_noise"],
                "x_forward_noised": x1_data["bond_edge_x_forward_noised"],
            }
            for key, value in x1_node_dict.items():
                data["x1"][key] = value
            for key, value in x1_edge_dict.items():
                data["x1", "bond", "x1"][key] = value

        for key in ("x2", "x3", "x4"):
            if key in data_dict:
                for field, value in data_dict[key].items():
                    data[key][field] = value

        return data


def collate_dpo_batch(batch_list):
    winners = [batch["winner"] for batch in batch_list]
    losers = [batch["loser"] for batch in batch_list]
    batched_winners = Batch.from_data_list(winners)
    batched_losers = Batch.from_data_list(losers)
    shared_timestep = torch.stack([batch["shared_timestep"] for batch in batch_list], dim=0)

    output = {
        "batch_type": "dpo",
        "winner": batched_winners,
        "loser": batched_losers,
        "shared_timestep": shared_timestep,
    }
    if all("winner_score" in batch for batch in batch_list):
        output["winner_score"] = [batch["winner_score"] for batch in batch_list]
        output["loser_score"] = [batch["loser_score"] for batch in batch_list]
    return output
