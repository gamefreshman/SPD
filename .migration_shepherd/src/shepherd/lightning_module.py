from copy import deepcopy

import torch

import pytorch_lightning as pl

from shepherd.model.model import Model


class LightningModule(pl.LightningModule):
    def __init__(self, params):
        super().__init__()

        self.save_hyperparameters()
        self.params = params

        self.model = Model(params)

        self.train_x1_denoising = params["training"]["train_x1_denoising"]
        self.train_x2_denoising = params["training"]["train_x2_denoising"]
        self.train_x3_denoising = params["training"]["train_x3_denoising"]
        self.train_x4_denoising = params["training"]["train_x4_denoising"]

        self.lr = params["training"]["lr"]
        self.min_lr = params["training"]["min_lr"]
        self.lr_steps = params["training"]["lr_steps"]

        self.enable_dpo = params["training"].get("enable_dpo", False)
        self.beta_dpo = params["training"].get("beta_dpo", 0.3)
        self.dpo_max_weight = params["training"].get("dpo_max_weight", 0.3)
        self.dpo_ramp_up_epochs = params["training"].get("dpo_ramp_up_epochs", 10)
        self.dpo_optimize_x4 = params["training"].get("dpo_optimize_x4", True)
        self._last_dpo_metrics = {}

        if self.enable_dpo:
            self.ref_model = deepcopy(self.model)
            self._freeze_ref_model()

    def _freeze_ref_model(self):
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False

    def load_state_dict(self, state_dict, strict=True):
        # DPO fine-tuning starts from non-DPO checkpoints that do not contain ref_model.
        result = super().load_state_dict(
            state_dict,
            strict=(strict and not self.enable_dpo),
        )
        if self.enable_dpo:
            self.ref_model.load_state_dict(self.model.state_dict())
            self._freeze_ref_model()
        return result

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        gamma = (self.min_lr / self.lr) ** (1.0 / self.lr_steps)
        func = lambda step: max(gamma ** step, self.min_lr / self.lr)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=func)

        lr_scheduler_config = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
            "strict": False,
            "name": None,
        }

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)

    def get_training_input_dict(self, data):
        input_dict = {}

        if self.params["dataset"]["compute_x1"]:
            x1_data = data["x1"]
            try:
                edge_store = data["x1", "bond", "x1"]
                bond_edge_mask = edge_store.mask
                bond_edge_index = edge_store.edge_index
                bond_edge_x = edge_store.x_forward_noised
                bond_edge_x_noise = edge_store.x_noise
            except (AttributeError, KeyError):
                bond_edge_mask = x1_data.bond_edge_mask
                bond_edge_x = x1_data.bond_edge_x_forward_noised
                bond_edge_x_noise = x1_data.bond_edge_x_noise
                try:
                    bond_edge_index = data["x1", "x1"].bond_edge_index
                except (AttributeError, KeyError):
                    bond_edge_index = x1_data.bond_edge_index

            input_dict["x1"] = {
                "decoder": {
                    "pos": x1_data.pos_forward_noised,
                    "x": x1_data.x_forward_noised,
                    "batch": x1_data.batch,
                    "bond_edge_mask": bond_edge_mask,
                    "bond_edge_index": bond_edge_index,
                    "bond_edge_x": bond_edge_x,
                    "timestep": x1_data.timestep,
                    "alpha_t": x1_data.alpha_t,
                    "sigma_t": x1_data.sigma_t,
                    "alpha_dash_t": x1_data.alpha_dash_t,
                    "sigma_dash_t": x1_data.sigma_dash_t,
                    "virtual_node_mask": x1_data.virtual_node_mask,
                    "pos_noise": x1_data.pos_noise,
                    "x_noise": x1_data.x_noise,
                    "bond_edge_x_noise": bond_edge_x_noise,
                },
            }

        if self.params["dataset"]["compute_x2"]:
            input_dict["x2"] = {
                "decoder": {
                    "pos": data["x2"].pos_forward_noised,
                    "x": data["x2"].x_forward_noised,
                    "batch": data["x2"].batch,
                    "timestep": data["x2"].timestep,
                    "alpha_t": data["x2"].alpha_t,
                    "sigma_t": data["x2"].sigma_t,
                    "alpha_dash_t": data["x2"].alpha_dash_t,
                    "sigma_dash_t": data["x2"].sigma_dash_t,
                    "virtual_node_mask": data["x2"].virtual_node_mask,
                    "pos_noise": data["x2"].pos_noise,
                },
            }

        if self.params["dataset"]["compute_x3"]:
            input_dict["x3"] = {
                "decoder": {
                    "pos": data["x3"].pos_forward_noised,
                    "x": data["x3"].x_forward_noised,
                    "batch": data["x3"].batch,
                    "timestep": data["x3"].timestep,
                    "alpha_t": data["x3"].alpha_t,
                    "sigma_t": data["x3"].sigma_t,
                    "alpha_dash_t": data["x3"].alpha_dash_t,
                    "sigma_dash_t": data["x3"].sigma_dash_t,
                    "virtual_node_mask": data["x3"].virtual_node_mask,
                    "pos_noise": data["x3"].pos_noise,
                    "x_noise": data["x3"].x_noise,
                },
            }

        if self.params["dataset"]["compute_x4"]:
            input_dict["x4"] = {
                "decoder": {
                    "x": data["x4"].x_forward_noised,
                    "pos": data["x4"].pos_forward_noised,
                    "direction": data["x4"].direction_forward_noised,
                    "batch": data["x4"].batch,
                    "timestep": data["x4"].timestep,
                    "alpha_t": data["x4"].alpha_t,
                    "sigma_t": data["x4"].sigma_t,
                    "alpha_dash_t": data["x4"].alpha_dash_t,
                    "sigma_dash_t": data["x4"].sigma_dash_t,
                    "virtual_node_mask": data["x4"].virtual_node_mask,
                    "direction_noise": data["x4"].direction_noise,
                    "pos_noise": data["x4"].pos_noise,
                    "x_noise": data["x4"].x_noise,
                },
            }

        input_dict["device"] = self.device
        input_dict["dtype"] = torch.float32
        return input_dict

    def forward_training(self, input_dict):
        _, output_dict = self.model.forward(input_dict)
        return output_dict

    def get_dpo_weight(self):
        if not self.enable_dpo:
            return 0.0
        if self.dpo_ramp_up_epochs <= 0:
            return self.dpo_max_weight
        ramp = min(1.0, float(self.current_epoch) / float(self.dpo_ramp_up_epochs))
        return ramp * self.dpo_max_weight

    def training_step(self, train_batch, batch_idx):
        if isinstance(train_batch, dict) and "winner" in train_batch and "loser" in train_batch:
            winner_batch = train_batch["winner"]
            loser_batch = train_batch["loser"]

            winner_input = self.get_training_input_dict(winner_batch)
            winner_output = self.forward_training(winner_input)
            loss_std, std_metrics = self._compute_standard_losses(winner_input, winner_output)

            loss_dpo, implicit_acc, model_diff, ref_diff = self.compute_dpo_loss(
                winner_batch,
                loser_batch,
                input_winner_precomputed=winner_input,
                output_model_winner_precomputed=winner_output,
            )

            dpo_weight = self.get_dpo_weight()
            loss = (1.0 - dpo_weight) * loss_std + dpo_weight * loss_dpo

            batch_size = int(winner_batch.molecule_id.shape[0])
            self._log_standard_metrics(winner_input, std_metrics, batch_size)
            self.log("loss_std_on_winner", loss_std, batch_size=batch_size)
            self.log("loss_dpo", loss_dpo, batch_size=batch_size)
            self.log("dpo_weight", dpo_weight, batch_size=batch_size)
            self.log("implicit_acc", implicit_acc, batch_size=batch_size)
            self.log("model_loss_diff", model_diff, batch_size=batch_size)
            self.log("ref_loss_diff", ref_diff, batch_size=batch_size)
            self.log("train_loss", loss, batch_size=batch_size)

            self._last_dpo_metrics = {
                "train_loss": float(loss.item()),
                "loss_std_on_winner": float(loss_std.item()),
                "loss_dpo": float(loss_dpo.item()),
                "dpo_weight": float(dpo_weight),
                "implicit_acc": float(implicit_acc.item()),
                "model_loss_diff": float(model_diff.item()),
                "ref_loss_diff": float(ref_diff.item()),
            }
            return loss

        data = train_batch
        batch_size = int(data.molecule_id.shape[0])

        input_dict = self.get_training_input_dict(data)
        output_dict = self.forward_training(input_dict)

        loss, metrics = self._compute_standard_losses(input_dict, output_dict)
        self._log_standard_metrics(input_dict, metrics, batch_size)
        self.log("train_loss", loss, batch_size=batch_size)
        return loss

    def _compute_standard_losses(self, input_dict, output_dict):
        loss = torch.tensor(0.0, device=self.device)
        metrics = {}

        if self.train_x1_denoising and "x1" in input_dict:
            loss_x1, feature_loss_x1, pos_loss_x1, bond_loss_x1 = self.x1_denoising_loss(
                input_dict,
                output_dict,
            )
            loss = loss + loss_x1
            metrics["x1"] = {
                "loss": loss_x1,
                "feature_loss": feature_loss_x1,
                "pos_loss": pos_loss_x1,
                "bond_loss": bond_loss_x1,
            }

        if self.train_x2_denoising and "x2" in input_dict:
            loss_x2 = self.x2_denoising_loss(input_dict, output_dict)
            loss = loss + loss_x2
            metrics["x2"] = {"loss": loss_x2}

        if self.train_x3_denoising and "x3" in input_dict:
            loss_x3, feature_loss_x3, pos_loss_x3 = self.x3_denoising_loss(
                input_dict,
                output_dict,
            )
            loss = loss + loss_x3
            metrics["x3"] = {
                "loss": loss_x3,
                "feature_loss": feature_loss_x3,
                "pos_loss": pos_loss_x3,
            }

        if self.train_x4_denoising and "x4" in input_dict:
            loss_x4, feature_loss_x4, pos_loss_x4, direction_loss_x4 = self.x4_denoising_loss(
                input_dict,
                output_dict,
            )
            loss = loss + loss_x4
            metrics["x4"] = {
                "loss": loss_x4,
                "feature_loss": feature_loss_x4,
                "pos_loss": pos_loss_x4,
                "direction_loss": direction_loss_x4,
            }

        metrics["total"] = loss
        return loss, metrics

    def _log_standard_metrics(self, input_dict, metrics, batch_size):
        if "x1" in metrics:
            batch_size_nodes = int((~input_dict["x1"]["decoder"]["virtual_node_mask"]).sum().item())
            batch_size_edges = int(input_dict["x1"]["decoder"]["bond_edge_x_noise"].shape[0])
            self.log("train_loss_x1", metrics["x1"]["loss"], batch_size=max(batch_size_nodes, 1))
            self.log(
                "train_pos_loss_x1",
                metrics["x1"]["pos_loss"],
                batch_size=max(batch_size_nodes, 1),
            )
            self.log(
                "train_feature_loss_x1",
                metrics["x1"]["feature_loss"],
                batch_size=max(batch_size_nodes, 1),
            )
            self.log(
                "train_bond_loss_x1",
                metrics["x1"]["bond_loss"],
                batch_size=max(batch_size_edges, 1),
            )

        if "x2" in metrics:
            batch_size_nodes = int((~input_dict["x2"]["decoder"]["virtual_node_mask"]).sum().item())
            self.log("train_loss_x2", metrics["x2"]["loss"], batch_size=max(batch_size_nodes, 1))

        if "x3" in metrics:
            batch_size_nodes = int((~input_dict["x3"]["decoder"]["virtual_node_mask"]).sum().item())
            self.log("train_loss_x3", metrics["x3"]["loss"], batch_size=max(batch_size_nodes, 1))
            self.log(
                "train_pos_loss_x3",
                metrics["x3"]["pos_loss"],
                batch_size=max(batch_size_nodes, 1),
            )
            self.log(
                "train_feature_loss_x3",
                metrics["x3"]["feature_loss"],
                batch_size=max(batch_size_nodes, 1),
            )

        if "x4" in metrics:
            batch_size_nodes = int((~input_dict["x4"]["decoder"]["virtual_node_mask"]).sum().item())
            self.log("train_loss_x4", metrics["x4"]["loss"], batch_size=max(batch_size_nodes, 1))
            self.log(
                "train_pos_loss_x4",
                metrics["x4"]["pos_loss"],
                batch_size=max(batch_size_nodes, 1),
            )
            self.log(
                "train_direction_loss_x4",
                metrics["x4"]["direction_loss"],
                batch_size=max(batch_size_nodes, 1),
            )
            self.log(
                "train_feature_loss_x4",
                metrics["x4"]["feature_loss"],
                batch_size=max(batch_size_nodes, 1),
            )

    def _masked_mse(self, pred, true, mask):
        if mask is None:
            return torch.mean((true - pred) ** 2.0)
        if int(mask.sum().item()) == 0:
            return pred.sum() * 0.0
        return torch.mean((true[mask] - pred[mask]) ** 2.0)

    def _balanced_bond_mse(self, pred, true, bond_mask):
        losses = []
        if bool(bond_mask.any()):
            losses.append(torch.mean((true[bond_mask] - pred[bond_mask]) ** 2.0))
        inverse_mask = ~bond_mask
        if bool(inverse_mask.any()):
            losses.append(torch.mean((true[inverse_mask] - pred[inverse_mask]) ** 2.0))
        if not losses:
            return pred.sum() * 0.0
        if len(losses) == 1:
            return losses[0]
        return torch.stack(losses).mean()

    def x1_denoising_loss(self, input_dict, output_dict):
        mask = ~input_dict["x1"]["decoder"]["virtual_node_mask"]
        pos_loss = self._masked_mse(
            output_dict["x1"]["decoder"]["denoiser"]["pos_out"],
            input_dict["x1"]["decoder"]["pos_noise"],
            mask,
        )
        feature_loss = self._masked_mse(
            output_dict["x1"]["decoder"]["denoiser"]["x_out"],
            input_dict["x1"]["decoder"]["x_noise"],
            mask,
        )

        bond_loss = feature_loss.sum() * 0.0
        if self.model.x1_bond_diffusion:
            bond_loss = self._balanced_bond_mse(
                output_dict["x1"]["decoder"]["denoiser"]["bond_edge_x_out"],
                input_dict["x1"]["decoder"]["bond_edge_x_noise"],
                input_dict["x1"]["decoder"]["bond_edge_mask"],
            )

        loss = pos_loss + feature_loss + bond_loss
        return loss, feature_loss, pos_loss, bond_loss

    def x2_denoising_loss(self, input_dict, output_dict):
        mask = ~input_dict["x2"]["decoder"]["virtual_node_mask"]
        return self._masked_mse(
            output_dict["x2"]["decoder"]["denoiser"]["pos_out"],
            input_dict["x2"]["decoder"]["pos_noise"],
            mask,
        )

    def x3_denoising_loss(self, input_dict, output_dict):
        mask = ~input_dict["x3"]["decoder"]["virtual_node_mask"]
        feature_loss = self._masked_mse(
            output_dict["x3"]["decoder"]["denoiser"]["x_out"].squeeze(),
            input_dict["x3"]["decoder"]["x_noise"],
            mask,
        )
        pos_loss = self._masked_mse(
            output_dict["x3"]["decoder"]["denoiser"]["pos_out"],
            input_dict["x3"]["decoder"]["pos_noise"],
            mask,
        )
        loss = feature_loss + pos_loss
        return loss, feature_loss, pos_loss

    def x4_denoising_loss(self, input_dict, output_dict):
        mask = ~input_dict["x4"]["decoder"]["virtual_node_mask"]
        zero = output_dict["x4"]["decoder"]["denoiser"]["x_out"].sum() * 0.0
        if int(mask.sum().item()) == 0:
            return zero, zero, zero, zero

        feature_loss = self._masked_mse(
            output_dict["x4"]["decoder"]["denoiser"]["x_out"].squeeze(),
            input_dict["x4"]["decoder"]["x_noise"],
            mask,
        )
        pos_loss = self._masked_mse(
            output_dict["x4"]["decoder"]["denoiser"]["pos_out"],
            input_dict["x4"]["decoder"]["pos_noise"],
            mask,
        )
        direction_loss = self._masked_mse(
            output_dict["x4"]["decoder"]["denoiser"]["direction_out"],
            input_dict["x4"]["decoder"]["direction_noise"],
            mask,
        )
        loss = feature_loss + pos_loss + direction_loss
        return loss, feature_loss, pos_loss, direction_loss

    def compute_dpo_loss(
        self,
        batch_winner,
        batch_loser,
        input_winner_precomputed=None,
        output_model_winner_precomputed=None,
    ):
        input_winner = (
            input_winner_precomputed
            if input_winner_precomputed is not None
            else self.get_training_input_dict(batch_winner)
        )
        input_loser = self.get_training_input_dict(batch_loser)

        output_model_winner = (
            output_model_winner_precomputed
            if output_model_winner_precomputed is not None
            else self.forward_training(input_winner)
        )
        output_model_loser = self.forward_training(input_loser)

        with torch.no_grad():
            output_ref_winner = self.ref_model.forward(input_winner)[1]
            output_ref_loser = self.ref_model.forward(input_loser)[1]

        loss_terms = []
        acc_terms = []
        model_diff_total = torch.tensor(0.0, device=self.device)
        ref_diff_total = torch.tensor(0.0, device=self.device)

        def add_channel(model_loss_w, model_loss_l, ref_loss_w, ref_loss_l):
            model_diff = model_loss_w - model_loss_l
            ref_diff = ref_loss_w - ref_loss_l
            inside_term = -self.beta_dpo * (model_diff - ref_diff)
            loss_terms.append(-torch.log(torch.sigmoid(inside_term) + 1e-8))
            acc_terms.append((model_diff < 0).float())
            return model_diff, ref_diff

        if self.train_x1_denoising and "x1" in input_winner and "x1" in input_loser:
            mask_w = ~input_winner["x1"]["decoder"]["virtual_node_mask"]
            mask_l = ~input_loser["x1"]["decoder"]["virtual_node_mask"]

            model_diff, ref_diff = add_channel(
                self._masked_mse(
                    output_model_winner["x1"]["decoder"]["denoiser"]["pos_out"],
                    input_winner["x1"]["decoder"]["pos_noise"],
                    mask_w,
                ),
                self._masked_mse(
                    output_model_loser["x1"]["decoder"]["denoiser"]["pos_out"],
                    input_loser["x1"]["decoder"]["pos_noise"],
                    mask_l,
                ),
                self._masked_mse(
                    output_ref_winner["x1"]["decoder"]["denoiser"]["pos_out"],
                    input_winner["x1"]["decoder"]["pos_noise"],
                    mask_w,
                ),
                self._masked_mse(
                    output_ref_loser["x1"]["decoder"]["denoiser"]["pos_out"],
                    input_loser["x1"]["decoder"]["pos_noise"],
                    mask_l,
                ),
            )
            model_diff_total = model_diff_total + model_diff
            ref_diff_total = ref_diff_total + ref_diff

            model_diff, ref_diff = add_channel(
                self._masked_mse(
                    output_model_winner["x1"]["decoder"]["denoiser"]["x_out"],
                    input_winner["x1"]["decoder"]["x_noise"],
                    mask_w,
                ),
                self._masked_mse(
                    output_model_loser["x1"]["decoder"]["denoiser"]["x_out"],
                    input_loser["x1"]["decoder"]["x_noise"],
                    mask_l,
                ),
                self._masked_mse(
                    output_ref_winner["x1"]["decoder"]["denoiser"]["x_out"],
                    input_winner["x1"]["decoder"]["x_noise"],
                    mask_w,
                ),
                self._masked_mse(
                    output_ref_loser["x1"]["decoder"]["denoiser"]["x_out"],
                    input_loser["x1"]["decoder"]["x_noise"],
                    mask_l,
                ),
            )
            model_diff_total = model_diff_total + model_diff
            ref_diff_total = ref_diff_total + ref_diff

            if self.model.x1_bond_diffusion:
                model_diff, ref_diff = add_channel(
                    self._balanced_bond_mse(
                        output_model_winner["x1"]["decoder"]["denoiser"]["bond_edge_x_out"],
                        input_winner["x1"]["decoder"]["bond_edge_x_noise"],
                        input_winner["x1"]["decoder"]["bond_edge_mask"],
                    ),
                    self._balanced_bond_mse(
                        output_model_loser["x1"]["decoder"]["denoiser"]["bond_edge_x_out"],
                        input_loser["x1"]["decoder"]["bond_edge_x_noise"],
                        input_loser["x1"]["decoder"]["bond_edge_mask"],
                    ),
                    self._balanced_bond_mse(
                        output_ref_winner["x1"]["decoder"]["denoiser"]["bond_edge_x_out"],
                        input_winner["x1"]["decoder"]["bond_edge_x_noise"],
                        input_winner["x1"]["decoder"]["bond_edge_mask"],
                    ),
                    self._balanced_bond_mse(
                        output_ref_loser["x1"]["decoder"]["denoiser"]["bond_edge_x_out"],
                        input_loser["x1"]["decoder"]["bond_edge_x_noise"],
                        input_loser["x1"]["decoder"]["bond_edge_mask"],
                    ),
                )
                model_diff_total = model_diff_total + model_diff
                ref_diff_total = ref_diff_total + ref_diff

        if (
            self.dpo_optimize_x4
            and self.train_x4_denoising
            and "x4" in input_winner
            and "x4" in input_loser
        ):
            mask_w = ~input_winner["x4"]["decoder"]["virtual_node_mask"]
            mask_l = ~input_loser["x4"]["decoder"]["virtual_node_mask"]
            if int(mask_w.sum().item()) > 0 and int(mask_l.sum().item()) > 0:
                model_diff, ref_diff = add_channel(
                    self._masked_mse(
                        output_model_winner["x4"]["decoder"]["denoiser"]["pos_out"],
                        input_winner["x4"]["decoder"]["pos_noise"],
                        mask_w,
                    ),
                    self._masked_mse(
                        output_model_loser["x4"]["decoder"]["denoiser"]["pos_out"],
                        input_loser["x4"]["decoder"]["pos_noise"],
                        mask_l,
                    ),
                    self._masked_mse(
                        output_ref_winner["x4"]["decoder"]["denoiser"]["pos_out"],
                        input_winner["x4"]["decoder"]["pos_noise"],
                        mask_w,
                    ),
                    self._masked_mse(
                        output_ref_loser["x4"]["decoder"]["denoiser"]["pos_out"],
                        input_loser["x4"]["decoder"]["pos_noise"],
                        mask_l,
                    ),
                )
                model_diff_total = model_diff_total + model_diff
                ref_diff_total = ref_diff_total + ref_diff

                model_diff, ref_diff = add_channel(
                    self._masked_mse(
                        output_model_winner["x4"]["decoder"]["denoiser"]["x_out"],
                        input_winner["x4"]["decoder"]["x_noise"],
                        mask_w,
                    ),
                    self._masked_mse(
                        output_model_loser["x4"]["decoder"]["denoiser"]["x_out"],
                        input_loser["x4"]["decoder"]["x_noise"],
                        mask_l,
                    ),
                    self._masked_mse(
                        output_ref_winner["x4"]["decoder"]["denoiser"]["x_out"],
                        input_winner["x4"]["decoder"]["x_noise"],
                        mask_w,
                    ),
                    self._masked_mse(
                        output_ref_loser["x4"]["decoder"]["denoiser"]["x_out"],
                        input_loser["x4"]["decoder"]["x_noise"],
                        mask_l,
                    ),
                )
                model_diff_total = model_diff_total + model_diff
                ref_diff_total = ref_diff_total + ref_diff

                model_diff, ref_diff = add_channel(
                    self._masked_mse(
                        output_model_winner["x4"]["decoder"]["denoiser"]["direction_out"],
                        input_winner["x4"]["decoder"]["direction_noise"],
                        mask_w,
                    ),
                    self._masked_mse(
                        output_model_loser["x4"]["decoder"]["denoiser"]["direction_out"],
                        input_loser["x4"]["decoder"]["direction_noise"],
                        mask_l,
                    ),
                    self._masked_mse(
                        output_ref_winner["x4"]["decoder"]["denoiser"]["direction_out"],
                        input_winner["x4"]["decoder"]["direction_noise"],
                        mask_w,
                    ),
                    self._masked_mse(
                        output_ref_loser["x4"]["decoder"]["denoiser"]["direction_out"],
                        input_loser["x4"]["decoder"]["direction_noise"],
                        mask_l,
                    ),
                )
                model_diff_total = model_diff_total + model_diff
                ref_diff_total = ref_diff_total + ref_diff

        if not loss_terms:
            zero = torch.tensor(0.0, device=self.device)
            return zero, zero, zero, zero

        loss_dpo = torch.stack(loss_terms).sum()
        implicit_acc = torch.stack(acc_terms).mean()
        return loss_dpo, implicit_acc, model_diff_total, ref_diff_total
