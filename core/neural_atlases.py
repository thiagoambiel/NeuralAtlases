from typing import Union, List, Tuple

import os
import json
import numpy as np
from pathlib import Path

import torch
from torch import optim

from models import IMLP, StackedIMLP

from core.losses import Losses

from core.utils import get_pixel_coordinates, stacked_run
from core.utils import save_atlas, save_video

from core.norms import squared_euclidean_norm, squared_taxicab_norm, foreground_range_norm

from core.config import Config
from dataclasses import asdict

from managers.data_manager import DataManager
from managers.atlas_manager import AtlasManager

import wandb
from tqdm import tqdm

import logging
import warnings

warnings.filterwarnings("ignore")

logging.basicConfig(
    format='%(asctime)s :: %(levelname)s :: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logging.getLogger().setLevel(logging.INFO)


class NeuralAtlases:
    def __init__(self,
                 project_path: Union[str, Path],
                 config: Union[Config, str] = None,
                 device: torch.device = torch.device("cpu")):

        self.device = device

        if self.device == torch.device("cuda") and not torch.cuda.is_available():
            logging.warning("CUDA unavailable, using CPU Only.")
            self.device = torch.device("cpu")

        self.project_path = Path(project_path)

        if not self.project_path.exists():
            raise FileNotFoundError(f"The project '{os.path.basename(self.project_path)}' cannot be found")

        # Define a config object for current project
        if (self.project_path / 'config.json').exists():
            if config is not None:
                logging.warning("Ignoring user config because current project already have a config file.")

            self.config = self.load_config(self.project_path / 'config.json')

        elif isinstance(config, Config):
            self.config = config

        elif isinstance(config, (str, Path)):
            self.config = self.load_config(config)

        else:
            logging.info("Using default config for current project.")
            self.config = Config()

        # Save Project Config to 'config.json' file
        with open(self.project_path / "config.json", 'w', encoding='utf-8') as f:
            json.dump(asdict(self.config), f, indent=4)

        self.larger_dim = np.maximum(self.config.size[0], self.config.size[1]).astype(np.int64)

        self.data_manager = DataManager(
            config=self.config,
            project_path=self.project_path,
            larger_dim=self.larger_dim,
            device=self.device
        )

        self.atlas_manager = AtlasManager(
            config=self.config,
            data_manager=self.data_manager,
            larger_dim=self.larger_dim,
            device=self.device
        )

        self.losses = Losses(
            config=self.config,
            larger_dim=self.larger_dim,
            device=self.device
        )

        self.foreground_model = None
        self.background_model = None
        self.atlas_model = None
        self.alpha_model = None

        self.optimizer = None

        self.checkpoint_loaded = False

        self.initial_step = 0

        if self.config.log_every > 0:
            wandb.init(project="neural_atlases")

        self.load_models()

    @staticmethod
    def load_config(config_path: Union[str, Path]) -> Config:
        with open(config_path) as f:
            logging.info(f"Loading Config File from: '{config_path}'")
            return Config(**json.load(f))

    def preprocess_masks(self):
        return self.data_manager.preprocess_masks()

    def preprocess_optical_flow(self):
        return self.data_manager.preprocess_optical_flow()

    def generate_atlases(self,
                         animated: bool = False,
                         debug: bool = False,
                         show_progress: bool = True):

        return self.atlas_manager.generate_atlases(
            foreground_model=self.foreground_model,
            background_model=self.background_model,
            alpha_model=self.alpha_model,
            atlas_model=self.atlas_model,

            animated=animated,
            debug=debug,
            show_progress=show_progress
        )

    def generate_atlases_from_frame(self,
                                    frame: np.array,
                                    frame_idx: int,
                                    animated: bool = False):

        return self.atlas_manager.generate_atlases_from_frame(
            foreground_model=self.foreground_model,
            background_model=self.background_model,
            alpha_model=self.alpha_model,

            frame=frame,
            frame_idx=frame_idx,
            animated=animated
        )

    def render_video_from_atlases(self,
                                  foreground_atlases: Union[None, List[None], List[torch.Tensor]] = None,
                                  background_atlas: torch.Tensor = None,

                                  remove_layer: str = None,
                                  show_progress: bool = True):

        return self.atlas_manager.render_video_from_atlases(
            foreground_atlases=foreground_atlases,
            background_atlas=background_atlas,

            foreground_model=self.foreground_model,
            background_model=self.background_model,
            alpha_model=self.alpha_model,

            remove_layer=remove_layer,
            show_progress=show_progress
        )

    def pretrain_mapping(self, steps, mapping_model, num_frames: int):
        optimizer = optim.Adam(mapping_model.parameters(), lr=1e-4)

        progress = tqdm(range(steps), desc="Pretraining Model")

        for _ in progress:
            for frame_idx in range(num_frames):
                Ys = torch.randint(self.config.size[1], (self.config.batch_size, 1))
                Xs = torch.randint(self.config.size[0], (self.config.batch_size, 1))

                y_norm = Ys / (self.larger_dim / 2) - 1
                x_norm = Xs / (self.larger_dim / 2) - 1
                t_norm = (frame_idx / (num_frames / 2.0) - 1) * torch.ones_like(y_norm)

                xyt = torch.cat(tensors=[x_norm, y_norm, t_norm], dim=1).to(self.device)

                uv_temp = mapping_model(xyt)

                optimizer.zero_grad()

                loss = (xyt[:, :2] * self.config.uv_mapping_scale - uv_temp).norm(dim=1).mean()
                loss.backward()

                optimizer.step()

                progress.set_postfix({"loss": loss.item()})

        return mapping_model.state_dict()

    def create_models(self):
        self.foreground_model = StackedIMLP(
            n_models=self.config.foreground_layers,
            input_dim=3,
            output_dim=2,
            hidden_dim=self.config.number_of_channels_foreground,
            num_layers=self.config.number_of_layers_foreground,
            use_positional=False,
        ).to(self.device)

        self.background_model = IMLP(
            input_dim=3,
            output_dim=2,
            hidden_dim=self.config.number_of_channels_background,
            num_layers=self.config.number_of_layers_background,
            use_positional=False
        ).to(self.device)

        self.atlas_model = IMLP(
            input_dim=2,
            output_dim=3,
            hidden_dim=self.config.number_of_channels_atlas,
            num_layers=self.config.number_of_layers_atlas,
            use_positional=True,
            positional_dim=self.config.positional_encoding_num_atlas,
            skip_layers=(4, 7),
            normalize=True
        ).to(self.device)

        self.alpha_model = IMLP(
            input_dim=3,
            output_dim=self.config.foreground_layers + (1 if self.config.foreground_layers > 1 else 0),
            hidden_dim=self.config.number_of_channels_alpha,
            num_layers=self.config.number_of_layers_alpha,
            use_positional=True,
            positional_dim=self.config.positional_encoding_num_alpha,
            softmax=self.config.foreground_layers > 1,
            normalize=self.config.foreground_layers == 1,
            alpha_range=self.config.foreground_layers == 1
        ).to(self.device)

        self.optimizer = optim.Adam(
            [{'params': list(self.foreground_model.parameters())},
             {'params': list(self.background_model.parameters())},
             {'params': list(self.alpha_model.parameters())},
             {'params': list(self.atlas_model.parameters())}], lr=1e-4)

    def train(self,
              steps: int = 400000,
              pretrain_steps: int = 100,
              bootstrapping_steps: int = 10000,
              global_rigidity_steps: int = 5000):

        if self.config.evaluate_every <= 10_000:
            logging.info("You can set 'evaluate_every' option to a higher "
                         "number (e.g. 50_000) to speed up model convergence.")

        frames_count = len(list((self.project_path / 'frames').glob('*')))

        masks_dir = self.project_path / 'results' / 'masks'
        masks_count = len(list(masks_dir.glob('*')))
        if not masks_dir.exists() or masks_count < frames_count:
            self.preprocess_masks()

        optical_flow_dir = self.project_path / 'results' / 'optical_flow'
        optical_flow_count = len(list(optical_flow_dir.glob('*')))
        if not optical_flow_dir.exists() or optical_flow_count < (frames_count - 1) * 2:
            self.preprocess_optical_flow()

        frames, masks, (scribbles, scribbles_colors), flows, flow_masks = self.data_manager.load_video()

        pixel_coords = get_pixel_coordinates(video_shape=frames.frames.shape)

        if pretrain_steps > 0 and not self.checkpoint_loaded:
            foreground_weights = self.pretrain_mapping(
                steps=pretrain_steps,
                mapping_model=self.foreground_model.models[0],
                num_frames=frames.num_frames,
            )

            for model in self.foreground_model.models:
                model.load_state_dict(foreground_weights)

            background_weights = self.pretrain_mapping(
                steps=pretrain_steps,
                mapping_model=self.background_model,
                num_frames=frames.num_frames,
            )

            self.background_model.load_state_dict(background_weights)

        progress = tqdm(range(self.initial_step, steps), initial=self.initial_step, total=steps, desc="Training")

        for step in progress:

            idxs_foreground = torch.randint(pixel_coords.shape[1], (self.config.batch_size, 1))
            pixel_coords_batch = pixel_coords[:, idxs_foreground]

            rgb_target = frames.frames[pixel_coords_batch[1, :], pixel_coords_batch[0, :], :, pixel_coords_batch[2, :]]
            rgb_target = rgb_target.squeeze(1).to(self.device)

            alpha_masks = masks[pixel_coords_batch[1, :], pixel_coords_batch[0, :], pixel_coords_batch[2, :]]
            alpha_masks = alpha_masks.to(self.device)

            if self.config.foreground_layers > 1:
                scribbles_batch = scribbles[pixel_coords_batch[1, :],
                                            pixel_coords_batch[0, :], :, pixel_coords_batch[2, :]]

                scribbles_batch = scribbles_batch.squeeze(1).to(self.device)
            else:
                scribbles_batch = None

            norm_coords_batch = torch.cat(
                (
                    pixel_coords_batch[0, :] / (self.larger_dim / 2) - 1,
                    pixel_coords_batch[1, :] / (self.larger_dim / 2) - 1,
                    pixel_coords_batch[2, :] / (frames.num_frames / 2.0) - 1
                ), dim=1).to(self.device)

            alpha = self.alpha_model(norm_coords_batch).unsqueeze(1)

            uv_background = self.background_model(norm_coords_batch)
            uv_foreground = self.foreground_model(norm_coords_batch)

            rgb_background = self.atlas_model(uv_background * 0.5 - 0.5)
            rgb_foreground = stacked_run(
                inputs=enumerate(uv_foreground),
                function=lambda x: self.atlas_model(foreground_range_norm(*x[::-1]))
            )

            if self.config.foreground_layers > 1:
                # Eq. 17 in Paper.
                background_term = alpha[:, :, 0] * rgb_background
                foreground_term = stacked_run(
                    inputs=range(len(rgb_foreground)),
                    function=lambda idx: alpha[:, :, idx + 1] * rgb_foreground[idx]
                )

                rgb_output = background_term + sum(foreground_term)
            else:
                # Eq. 4 in Paper.
                rgb_output = (1.0 - alpha[:, :, 0]) * rgb_background + alpha[:, :, 0] * rgb_foreground[0]

            # Eq. 6 in Paper.
            rgb_error = squared_euclidean_norm(rgb_target - rgb_output).mean()

            # Eq. 7 in Paper.
            gradient_error = self.losses.gradient_loss(
                frames_derivative_x=frames.derivative_x,
                frames_derivative_y=frames.derivative_y,
                pixel_coords=pixel_coords_batch,
                rgb_output=rgb_output,

                foreground_model=self.foreground_model,
                background_model=self.background_model,
                atlas_model=self.atlas_model,
                alpha_model=self.alpha_model,

                num_frames=frames.num_frames
            )

            # Eq. 5 in Paper.
            color_loss = self.config.rgb_coeff * rgb_error + self.config.gradient_coeff * gradient_error

            # Eq. 9 in Paper.
            foreground_rigidity_error = stacked_run(
                inputs=range(len(uv_foreground)),
                function=lambda idx: self.losses.rigidity_loss(
                    uv_map=uv_foreground[idx],
                    pixel_coords=pixel_coords_batch,
                    mapping_model=self.foreground_model.models[idx],
                    num_frames=frames.num_frames,
                    derivative_amount=np.int64(self.config.derivative_amount),
                    uv_mapping_scale=np.float64(self.config.uv_mapping_scale)
                )
            )

            # Eq. 9 in Paper.
            background_rigidity_error = self.losses.rigidity_loss(
                uv_map=uv_background,
                pixel_coords=pixel_coords_batch,
                mapping_model=self.background_model,
                num_frames=frames.num_frames,
                derivative_amount=np.int64(self.config.derivative_amount),
                uv_mapping_scale=np.float64(self.config.uv_mapping_scale)
            )

            # Eq. 10 in Paper.
            rigidity_loss = background_rigidity_error + sum(foreground_rigidity_error)
            rigidity_loss *= self.config.rigidity_coeff

            # Eq. 11 in Paper. (Term 1)
            foreground_flow_term = stacked_run(
                inputs=range(len(uv_foreground)),
                function=lambda idx: self.losses.optical_flow_loss(
                    mapping_model=self.foreground_model.models[idx],
                    pixel_coords=pixel_coords_batch,
                    uv_map=uv_foreground[idx],
                    optical_flows_forward=flows.forward,
                    optical_flows_forward_mask=flow_masks.forward,
                    optical_flows_backward=flows.backward,
                    optical_flows_backward_mask=flow_masks.backward,
                    alpha=alpha[:, :, idx + 1] if self.config.foreground_layers > 1 else alpha[:, :, 0],
                    num_frames=frames.num_frames,
                    uv_mapping_scale=np.float64(self.config.uv_mapping_scale)
                )
            )

            # Eq. 11 in Paper. (Term 2)
            background_flow_term = self.losses.optical_flow_loss(
                mapping_model=self.background_model,
                pixel_coords=pixel_coords_batch,
                uv_map=uv_background,
                optical_flows_forward=flows.forward,
                optical_flows_forward_mask=flow_masks.forward,
                optical_flows_backward=flows.backward,
                optical_flows_backward_mask=flow_masks.backward,
                alpha=alpha[:, :, 0] if self.config.foreground_layers > 1 else 1 - alpha[:, :, 0],
                num_frames=frames.num_frames,
                uv_mapping_scale=np.float64(self.config.uv_mapping_scale)
            )

            # Eq. 11 in Paper.
            flow_error = background_flow_term + sum(foreground_flow_term)

            # Eq. 12 in Paper.
            alpha_flow_error = self.losses.optical_flow_alpha_loss(
                alpha_model=self.alpha_model,
                pixel_coords=pixel_coords_batch,
                alpha=alpha,
                optical_flows_forward=flows.forward,
                optical_flows_forward_mask=flow_masks.forward,
                optical_flows_backward=flows.backward,
                optical_flows_backward_mask=flow_masks.backward,
                num_frames=frames.num_frames
            )

            # Eq. 13 in Paper.
            flow_loss = self.config.optical_flow_coeff * flow_error + self.config.alpha_flow_coeff * alpha_flow_error

            # Eq. 14 in Paper.
            if self.config.foreground_layers > 1:
                sparsity_loss = self.losses.sparsity_loss(
                    atlas_model=self.atlas_model,
                    batch_size=self.config.sparsity_batch_size
                )

            else:
                sparsity_loss = squared_taxicab_norm((1.0 - alpha[:, :, 0]) * rgb_foreground[0]).mean()

            sparsity_loss *= self.config.sparsity_coeff

            # Eq. 1 in Paper.
            loss = color_loss + rigidity_loss + flow_loss + sparsity_loss

            alpha_bootstrapping_loss = 0
            global_fg_rigidity_loss = 0
            global_bg_rigidity_loss = 0

            if step <= bootstrapping_steps:
                # BCE(m^p, α^p)
                alpha_bootstrapping_loss = self.losses.alpha_bootstrapping_loss(
                    alpha=alpha,
                    target_alpha=alpha_masks,
                    normalize=self.config.foreground_layers > 1
                )

                loss += self.config.alpha_bootstrapping_coeff * alpha_bootstrapping_loss

                if self.config.foreground_layers > 1:
                    scribbles_loss = stacked_run(
                        inputs=range(len(scribbles_colors)),
                        function=lambda idx: self.losses.scribble_loss(
                            scribbles=scribbles_batch,
                            alpha=alpha[:, :, idx + 1],
                            color=scribbles_colors[idx]
                        )
                    )

                    loss += self.config.scribbles_coeff * sum(scribbles_loss)

                if step <= global_rigidity_steps:
                    global_fg_rigidity_loss = stacked_run(
                        inputs=range(len(uv_foreground)),
                        function=lambda idx: self.losses.rigidity_loss(
                            uv_map=uv_foreground[idx],
                            pixel_coords=pixel_coords_batch,
                            mapping_model=self.foreground_model.models[idx],
                            num_frames=frames.num_frames,
                            derivative_amount=np.int64(self.config.global_derivative_amount),
                            uv_mapping_scale=np.float64(self.config.uv_mapping_scale)
                        )
                    )
                    global_fg_rigidity_loss = sum(global_fg_rigidity_loss)

                    global_bg_rigidity_loss = self.losses.rigidity_loss(
                        uv_map=uv_background,
                        pixel_coords=pixel_coords_batch,
                        mapping_model=self.background_model,
                        num_frames=frames.num_frames,
                        derivative_amount=np.int64(self.config.global_derivative_amount),
                        uv_mapping_scale=np.float64(self.config.uv_mapping_scale)
                    )

                    loss += self.config.global_rigidity_coeff_fg * global_fg_rigidity_loss
                    loss += self.config.global_rigidity_coeff_bg * global_bg_rigidity_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            progress.set_postfix({"loss": loss.item()})

            if self.config.log_every > 0 and step % self.config.log_every == self.config.log_every - 1:
                wandb.log({
                    "Rigidity Loss": rigidity_loss,
                    "RGB Loss": rgb_error,
                    "Optical Flow Loss": flow_error,
                    "Alpha Flow Loss": alpha_flow_error,
                    "Sparsity Loss": sparsity_loss,
                    "Gradient Loss": gradient_error,

                    "Alpha Bootstrapping Loss": alpha_bootstrapping_loss,
                    "Global Foreground Rigidity Loss": global_fg_rigidity_loss,
                    "Global Background Rigidity Loss": global_bg_rigidity_loss,

                    "Loss": loss
                })

            if step % self.config.save_every == self.config.save_every - 1:
                checkpoint_path = self.project_path / 'results' / "checkpoints.pth"
                self.save_checkpoints(checkpoint_path=checkpoint_path, current_step=step + 1)

            if step % self.config.evaluate_every == self.config.evaluate_every - 1:
                self.evaluate(dataset=(frames, masks, (scribbles, scribbles_colors), flows, flow_masks))

    def evaluate(self,
                 dataset: Tuple = None,
                 show_progress: bool = False):

        fg_atlases, bg_atlas, (textures, videos) = self.atlas_manager.generate_atlases(
            foreground_model=self.foreground_model,
            background_model=self.background_model,
            atlas_model=self.atlas_model,
            alpha_model=self.alpha_model,
            debug=True,
            show_progress=show_progress,
            dataset=dataset,
        )

        # Save Foreground and Background Atlases
        atlases_path = self.project_path / 'results' / 'atlases'
        atlases_path.mkdir(exist_ok=True)

        for layer_idx, fg_atlas in enumerate(fg_atlases):
            save_atlas(fg_atlas, str(atlases_path / f"foreground_atlas_{layer_idx}.png"))

        save_atlas(bg_atlas, str(atlases_path / "background_atlas.png"))

        # Save Debug Data Like Textures and Reconstructions
        debug_path = self.project_path / 'results' / 'debug'
        debug_path.mkdir(exist_ok=True)

        save_video(videos['video_reconstruction'], debug_path / "video_reconstruction.mp4")

        for layer_idx, alpha_recon in enumerate(videos['alpha_reconstruction']):
            save_video(alpha_recon[:, :, np.newaxis, :], debug_path / f"alpha_reconstruction_{layer_idx}.mp4")

        for layer_idx, fg_texture in enumerate(textures['foreground_textures']):
            fg_texture = (fg_texture * 255).astype(np.uint8)
            save_atlas(fg_texture, debug_path / f"foreground_texture_{layer_idx}.png")

        bg_texture = (textures['background_texture'] * 255).astype(np.uint8)
        save_atlas(bg_texture, debug_path / "background_texture.png")

    def load_models(self):
        checkpoint_path = self.project_path / 'results' / "checkpoints.pth"
        if checkpoint_path.exists():
            self.load_checkpoints(checkpoint_path=checkpoint_path)
        else:
            self.create_models()
            logging.info("No checkpoint file was found, Going to train from scratch")

    def load_checkpoints(self, checkpoint_path: Union[str, Path]):
        self.create_models()

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.foreground_model.load_state_dict(checkpoint["foreground_model"])
        self.background_model.load_state_dict(checkpoint["background_model"])
        self.atlas_model.load_state_dict(checkpoint["atlas_model"])
        self.alpha_model.load_state_dict(checkpoint["alpha_model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])

        self.initial_step = checkpoint["current_step"]

        self.checkpoint_loaded = True

        logging.info(f"Restoring Checkpoint from '{checkpoint_path}' at Step {self.initial_step}")

    def save_checkpoints(self, checkpoint_path: Union[str, Path], current_step: int = 0):
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

        torch.save({
            'foreground_model': self.foreground_model.state_dict(),
            'background_model': self.background_model.state_dict(),
            'atlas_model': self.atlas_model.state_dict(),
            'alpha_model': self.alpha_model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'current_step': current_step
        }, checkpoint_path)
