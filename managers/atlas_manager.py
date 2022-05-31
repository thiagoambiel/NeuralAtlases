from typing import List, Union
from dataclasses import dataclass

import torch
from torch import nn

import cv2
import numpy as np
from scipy.interpolate import griddata

from managers.data_manager import DataManager

from core.utils import stacked_run
from core.utils import bilinear_interpolate

from core.norms import foreground_range_norm
from core.config import Config

from tqdm import tqdm

import logging

logging.basicConfig(
    format='%(asctime)s :: %(levelname)s :: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


@dataclass
class Pixels:
    colors: np.array
    x_coords: np.array
    y_coords: np.array
    relevant_idxs: np.array


class AtlasManager:
    def __init__(self,
                 config: Config,
                 data_manager: DataManager,
                 larger_dim: np.int64,
                 device: torch.device = torch.device("cpu")):

        self.device = device

        self.config = config
        self.data_manager = data_manager
        self.larger_dim = larger_dim

    def compute_texture(self,
                        atlas_model: nn.Module,
                        xy_range: tuple,
                        resolution: int = 1000):

        """ Compute Texture Image from a Region in Atlas Space

        :param atlas_model: nn.Module: Atlas IMLP Model
        :param xy_range: Tuple: Region of Atlas Space to Render
        :param resolution: int: Desired Resolution to Render the Texture

        :return: texture: np.Array: Rendered Texture Image
        """

        x_range, y_range = xy_range

        x_coords = torch.linspace(*x_range, resolution).unsqueeze(1)
        y_coords = torch.linspace(*y_range, resolution)

        texture = torch.zeros((resolution, resolution, 3))

        with torch.no_grad():
            for idx, y_coord in enumerate(y_coords):
                pixels = atlas_model(torch.cat([x_coords, torch.ones_like(x_coords) * y_coord], dim=1).to(self.device))
                texture[idx, :, :] = pixels.detach().cpu()

        return texture.numpy()

    @staticmethod
    def compute_batches(coords: torch.Tensor,
                        batch_size: int = 100000):

        """ Split Coordinates in Batches of Coordinates

        :param coords: torch.Tensor: Input Coordinates to Split
        :param batch_size: int: Maximum Size of Each Split Batch

        :return: np.array: An Array with N Batches of Coords with Size [batch_size]
        """

        return np.array_split(coords.numpy(), np.ceil(coords.shape[0] / batch_size))

    def compute_background_range(self,
                                 background_model: nn.Module,
                                 alpha_model: nn.Module,

                                 masks: torch.Tensor,

                                 num_frames: np.int64,
                                 invert_alpha: bool = False,
                                 alpha_thresh: float = -0.5):

        """ Calculate the Atlas range where Background Pixels were Mapped

        :param background_model: nn.Module: Background IMLP Mapping Model
        :param alpha_model: nn.Module: Alpha IMLP Model

        :param masks: torch.Tensor: Mask Values from MaskRCNN/MODNet Model

        :param num_frames: np.int64: Number of Project Video Frames
        :param invert_alpha: bool: Invert Alpha Values Signal?
        :param alpha_thresh: float: Constant to Delimit Minimum Relevant Alpha Values

        :return: Tuple: Range where Background Pixels were Mapped
        """

        # consider only pixels that their masks are 1
        coords_y, coords_x, coords_t = torch.where(masks)

        # split all y, x, t coordinates to batches of size 100k
        y_batches = self.compute_batches(coords_y)
        x_batches = self.compute_batches(coords_x)
        t_batches = self.compute_batches(coords_t)

        min_x = 1
        min_y = 1
        max_x = -1
        max_y = -1

        with torch.no_grad():
            for i in range(len(y_batches)):
                norm_y = torch.from_numpy(y_batches[i]).unsqueeze(1) / (self.larger_dim / 2) - 1
                norm_x = torch.from_numpy(x_batches[i]).unsqueeze(1) / (self.larger_dim / 2) - 1
                norm_t = torch.from_numpy(t_batches[i]).unsqueeze(1) / (num_frames / 2) - 1

                xyt_coords = torch.cat((norm_x, norm_y, norm_t), dim=1).to(self.device)

                uv = background_model(xyt_coords).cpu()
                alpha = alpha_model(xyt_coords).cpu()
                alpha = (alpha[:, -self.config.foreground_layers:].sum(dim=1) / 0.5) - 1.0

                if invert_alpha:
                    alpha = -alpha

                if torch.any(alpha > alpha_thresh):
                    uv = uv * 0.5 - 0.5

                    current_min_x = torch.min(uv[alpha > alpha_thresh, 0])
                    current_min_y = torch.min(uv[alpha > alpha_thresh, 1])

                    current_max_x = torch.max(uv[alpha > alpha_thresh, 0])
                    current_max_y = torch.max(uv[alpha > alpha_thresh, 1])

                    min_x = torch.min(torch.tensor([current_min_x, min_x]))
                    min_y = torch.min(torch.tensor([current_min_y, min_y]))

                    max_x = torch.max(torch.tensor([current_max_x, max_x]))
                    max_y = torch.max(torch.tensor([current_max_y, max_y]))

        max_x = np.minimum(max_x, 1)
        max_y = np.minimum(max_y, 1)

        min_x = np.maximum(min_x, -1)
        min_y = np.maximum(min_y, -1)

        edge_size = torch.max(torch.tensor([max_x - min_x, max_y - min_y]))

        x_range = (min_x, min_x + edge_size)
        y_range = (min_y, min_x + edge_size)

        return (x_range, y_range), edge_size

    @staticmethod
    def compute_relevant_pixels(xy_range: tuple,
                                uv: np.array,
                                image: torch.Tensor,
                                resolution: int = 1000):

        """ Compute which Pixels are Relevant in the Input Image

        :param xy_range: Tuple: Region of Atlas Space to Render
        :param uv: np.array: Coordinates of Layer in Atlas Space
        :param image: torch.Tensor: Input Image to Extract Relevant Pixels
        :param resolution: int: The resolution at which the Texture was Rendered

        :return: Pixels: Object with Selected Pixels Colors, Coordinates and Indices
        """

        x_range, y_range = xy_range

        pixel_size = resolution / (x_range[1] - x_range[0])

        uv_x = uv[:, 0]
        uv_y = uv[:, 1]

        # Change uv to pixel coordinates of the discretized image
        x_coords = ((uv_x - x_range[0]) * pixel_size).numpy()
        y_coords = ((uv_y - y_range[0]) * pixel_size).numpy()

        # Bilinear interpolate pixel colors from the image
        pixels = bilinear_interpolate(image, x_coords, y_coords)

        # Relevant pixel locations should be positive:
        positive_logical_y = np.logical_and(np.ceil(y_coords) >= 0, np.floor(y_coords) >= 0)
        positive_logical_x = np.logical_and(np.ceil(x_coords) >= 0, np.floor(x_coords) >= 0)
        positive_logical = np.logical_and(positive_logical_y, positive_logical_x)

        # Relevant pixel locations should be inside the image borders:
        borders_logical_y = np.logical_and(np.ceil(y_coords) < resolution, np.floor(y_coords) < resolution)
        borders_logical_x = np.logical_and(np.ceil(x_coords) < resolution, np.floor(x_coords) < resolution)
        borders_logical = np.logical_and(borders_logical_y, borders_logical_x)

        # Relevant should satisfy both conditions
        relevant_pixels_idxs = np.logical_and(positive_logical, borders_logical)

        relevant_pixels = Pixels(
            colors=pixels[relevant_pixels_idxs],
            x_coords=x_coords[relevant_pixels_idxs],
            y_coords=y_coords[relevant_pixels_idxs],
            relevant_idxs=relevant_pixels_idxs
        )

        return relevant_pixels

    def compute_frame(self,
                      atlas_model: nn.Module,

                      foreground_uv: torch.Tensor,
                      background_uv: torch.Tensor,

                      alpha: torch.Tensor):

        """ Compute the RGB Values for a Given Foreground and Background UV Coordinates

        :param atlas_model: nn.Module: Atlas IMLP Model

        :param foreground_uv: torch.Tensor: Coordinates of Foreground Layer in Atlas Space
        :param background_uv: torch.Tensor: Coordinates of Background Layer in Atlas Space

        :param alpha: torch.Tensor: Alpha IMLP Model Output for Current Layer

        :return: frame_rgb: torch.Tensor: Rendered Output Frame for given UV Coordinates and Alpha
        """

        rgb_background = atlas_model(background_uv * 0.5 - 0.5)
        rgb_foreground = stacked_run(
            inputs=enumerate(foreground_uv),
            function=lambda x: atlas_model(foreground_range_norm(x[1], x[0]))
        )

        if self.config.foreground_layers > 1:
            background_term = alpha[:, :, 0] * rgb_background
            foreground_term = stacked_run(
                inputs=range(len(rgb_foreground)),
                function=lambda idx: alpha[:, :, idx + 1] * rgb_foreground[idx]
            )

            frame_rgb = background_term + sum(foreground_term)
        else:
            frame_rgb = (1.0 - alpha[:, :, 0]) * rgb_background + alpha[:, :, 0] * rgb_foreground[0]

        return frame_rgb

    def alpha_filter(self,
                     alpha: torch.Tensor,
                     remove_layer: str = None):

        """ Filter Alpha Values to Remove Foreground or Background Layers

        :param alpha: torch.Tensor: Alpha IMLP Model Output for Current Layer
        :param remove_layer: str: Layer to Remove ['foreground', 'background' or None]

        :return: alpha: torch.Tensor: Filtered Alpha with Desired Layers Removed
        """

        if remove_layer == 'foreground':
            if self.config.foreground_layers > 1:
                alpha[:, :, 0] = torch.ones_like(alpha[:, :, 0])
                alpha[:, :, 1] = torch.zeros_like(alpha[:, :, 1])
                alpha[:, :, 2] = torch.zeros_like(alpha[:, :, 2])
            else:
                alpha = torch.zeros_like(alpha)

        elif remove_layer == 'background':
            if self.config.foreground_layers > 1:
                alpha[:, :, 0] = torch.zeros_like(alpha[:, :, 0])
            else:
                alpha = torch.ones_like(alpha)

        return alpha

    def compute_alpha_and_pixels(self,
                                 xyt_coords: torch.Tensor,

                                 foreground_model: nn.Module,
                                 foreground_texture: Union[torch.Tensor, List[torch.Tensor]],

                                 background_model: nn.Module,
                                 background_texture: torch.Tensor,
                                 background_range: tuple,

                                 alpha_model: nn.Module,

                                 atlas_model: nn.Module = None,

                                 remove_layer: str = None,
                                 render_frame: bool = False):

        """ Compute Alpha and RGB Values for Given UV Coordinates

        :param xyt_coords: torch.Tensor: Coordinates X-Axis, Y-Axis and Time in Video Space

        :param foreground_model: nn.Module: Foreground IMLP Mapping Model
        :param foreground_texture: torch.Tensor: Rendered Foreground Texture Image

        :param background_model: nn.Module: Background IMLP Mapping Model
        :param background_texture: torch.Tensor: Rendered Background Texture Image
        :param background_range: Tuple: Range where Background Pixels were Mapped

        :param alpha_model: nn.Module: Alpha IMLP Model

        :param atlas_model: nn.Module: Atlas IMLP Model

        :param remove_layer: str: Layer to Remove ['foreground', 'background' or None]
        :param render_frame: bool: Render Current Coordinates RGB Values?

        :return: Alpha values, RGB values and Rendered Frame if render_frame = True
        """

        foreground_uv = foreground_model(xyt_coords)
        background_uv = background_model(xyt_coords)

        alpha = alpha_model(xyt_coords).unsqueeze(1)
        alpha = self.alpha_filter(alpha, remove_layer)

        frame_rgb = None

        if render_frame:
            assert atlas_model is not None, "You need to pass 'atlas_model' to render frames"

            frame_rgb = self.compute_frame(
                atlas_model=atlas_model,

                foreground_uv=foreground_uv,
                background_uv=background_uv,

                alpha=alpha
            )

        foreground_uv = foreground_uv.cpu()
        background_uv = background_uv.cpu()

        fg_pixels = stacked_run(
            inputs=range(self.config.foreground_layers),
            function=lambda idx: self.compute_relevant_pixels(
                resolution=1000,
                xy_range=((0, 1), np.array((0, 1)) - idx),
                uv=foreground_range_norm(foreground_uv[idx], idx),
                image=foreground_texture[idx]
            )
        )

        bg_pixels = self.compute_relevant_pixels(
            resolution=1000,
            xy_range=background_range,
            uv=background_uv * 0.5 - 0.5,
            image=background_texture
        )

        return alpha, fg_pixels, bg_pixels, frame_rgb

    def compute_masks(self,
                      foreground_model: nn.Module,
                      foreground_texture: torch.Tensor,

                      background_model: nn.Module,
                      background_texture: torch.Tensor,
                      background_range: tuple,

                      alpha_model: nn.Module,
                      atlas_model: nn.Module,

                      num_frames: np.int64,

                      resolution: int = 1000,
                      render_video: bool = True,
                      show_progress: bool = True):

        """ Compute Foreground and Background Alpha Values for Atlas

        :param foreground_model: nn.Module: Foreground IMLP Mapping Model
        :param foreground_texture: torch.Tensor: Rendered Foreground Texture Image

        :param background_model: nn.Module: Background IMLP Mapping Model
        :param background_texture: torch.Tensor: Rendered Background Texture Image
        :param background_range: Tuple: Range where Background Pixels were Mapped

        :param alpha_model: nn.Module: Alpha IMLP Model
        :param atlas_model: nn.Module: Atlas IMLP Model

        :param num_frames: np.int64: Number of Project Video Frames

        :param resolution: int: The resolution at which the Texture was Rendered
        :param render_video: bool: Render Alpha and RGB Video Reconstructions?
        :param show_progress: bool: Show Progress Bar?

        :return: Foreground and Background Alpha Values for Atlas and Rendered RGB and Alpha Videos
        """

        video_recon = np.zeros((self.config.size[1], self.config.size[0], 3, num_frames))

        alpha_recon = np.zeros((self.config.foreground_layers,
                                self.config.size[1], self.config.size[0], num_frames))

        foreground_alpha = np.zeros((self.config.foreground_layers, resolution, resolution))
        background_alpha = np.zeros((resolution, resolution))

        frames = tqdm(range(num_frames), desc="Computing Global Masks") if show_progress else range(num_frames)

        with torch.no_grad():
            for f in frames:
                coords_y, coords_x = torch.where(torch.ones(self.config.size[1], self.config.size[0]) > 0)

                batches_y = self.compute_batches(coords_y)
                batches_x = self.compute_batches(coords_x)

                for i in range(len(batches_y)):
                    norm_y = torch.from_numpy(batches_y[i]).unsqueeze(1) / (self.larger_dim / 2) - 1
                    norm_x = torch.from_numpy(batches_x[i]).unsqueeze(1) / (self.larger_dim / 2) - 1
                    norm_t = (f / (num_frames / 2.0) - 1) * torch.ones_like(norm_y)

                    xyt_coords = torch.cat((norm_x, norm_y, norm_t), dim=1).to(self.device)

                    alpha, fg_pixels, bg_pixels, frame_rgb = self.compute_alpha_and_pixels(
                        xyt_coords=xyt_coords,

                        foreground_model=foreground_model,
                        foreground_texture=foreground_texture,

                        background_model=background_model,
                        background_texture=background_texture,
                        background_range=background_range,

                        alpha_model=alpha_model,

                        atlas_model=atlas_model,
                        render_frame=render_video
                    )

                    for layer_idx, fg_i_pixels in enumerate(fg_pixels):
                        for y_function, x_function in [(np.ceil, np.ceil),
                                                       (np.floor, np.floor),
                                                       (np.floor, np.ceil),
                                                       (np.ceil, np.floor)]:

                            fg_y_coords = y_function(fg_i_pixels.y_coords).astype(np.int64)
                            fg_x_coords = x_function(fg_i_pixels.x_coords).astype(np.int64)

                            bg_y_coords = y_function(bg_pixels.y_coords).astype(np.int64)
                            bg_x_coords = x_function(bg_pixels.x_coords).astype(np.int64)

                            if self.config.foreground_layers > 1:
                                layer_alpha = alpha[:, :, layer_idx + 1].squeeze()
                            else:
                                layer_alpha = alpha.squeeze()

                            foreground_alpha[layer_idx, fg_y_coords, fg_x_coords] = np.maximum(
                                foreground_alpha[layer_idx, fg_y_coords, fg_x_coords],
                                layer_alpha[fg_i_pixels.relevant_idxs].cpu().numpy()
                            )

                            if layer_idx == 0:
                                background_alpha[bg_y_coords, bg_x_coords] = 1

                    if render_video:
                        video_recon[batches_y[i], batches_x[i], :, f] = frame_rgb.cpu().numpy()

                        for layer_idx in range(self.config.foreground_layers):

                            if self.config.foreground_layers > 1:
                                layer_alpha = alpha[:, 0, layer_idx + 1]
                            else:
                                layer_alpha = alpha[:, 0, 0]

                            alpha_recon[layer_idx, batches_y[i], batches_x[i], f] = layer_alpha.cpu().numpy()

        return foreground_alpha, background_alpha, video_recon, alpha_recon

    def render_video(self,
                     video_frames: torch.Tensor,

                     foreground_atlases: List[torch.Tensor],
                     background_atlas: torch.Tensor,

                     foreground_model: nn.Module,

                     background_model: nn.Module,
                     background_range: tuple,

                     alpha_model: nn.Module,

                     remove_layer: str = None,
                     show_progress: bool = True):

        """ Render the Edits in Atlas and Propagate them to the Original Video Frames

        :param video_frames: torch.Tensor: Original Video Frames to Concat with Rendered Edits

        :param foreground_atlases: torch.Tensor: Foreground Atlas Images with User Edits for Each Layer
        :param background_atlas: torch.Tensor: Background Atlas Image with User Edits

        :param foreground_model: nn.Module: Foreground IMLP Mapping Model

        :param background_model: nn.Module: Background IMLP Mapping Model
        :param background_range: Tuple: Range where Background Pixels were Mapped

        :param alpha_model: nn.Module: Alpha IMLP Model

        :param remove_layer: str: Layer to Remove ['foreground', 'background' or None]
        :param show_progress: bool: Show Progress Bar?

        :return: video_reconstruction: torch.Tensor: Reconstructed Video Frames with User Edits
        """

        num_frames = video_frames.shape[3]

        video_reconstruction = np.zeros((self.config.size[1], self.config.size[0], 3, num_frames))

        if self.config.foreground_layers > 1:
            alpha_reconstruction = np.zeros((self.config.size[1], self.config.size[0],
                                             self.config.foreground_layers + 1, num_frames))
        else:
            alpha_reconstruction = np.zeros((self.config.size[1], self.config.size[0], num_frames))

        background_changes = np.zeros((self.config.size[1], self.config.size[0], 4, num_frames))
        foreground_changes = np.zeros((self.config.foreground_layers,
                                       self.config.size[1], self.config.size[0], 4, num_frames))

        coords_y, coords_x = torch.where(torch.ones(self.config.size[1], self.config.size[0]) > 0)
        batches_y = self.compute_batches(coords_y)
        batches_x = self.compute_batches(coords_x)

        frames = tqdm(range(num_frames), desc="Rendering Video") if show_progress else range(num_frames)

        with torch.no_grad():
            for f in frames:

                for i in range(len(batches_y)):
                    norm_y = torch.from_numpy(batches_y[i]).unsqueeze(1) / (self.larger_dim / 2) - 1
                    norm_x = torch.from_numpy(batches_x[i]).unsqueeze(1) / (self.larger_dim / 2) - 1
                    norm_t = (f / (num_frames / 2.0) - 1) * torch.ones_like(norm_y)

                    xyt_coords = torch.cat((norm_x, norm_y, norm_t), dim=1).to(self.device)

                    # Support to animated atlases
                    fg_frame_atlas = []

                    for foreground_atlas in foreground_atlases:
                        if len(foreground_atlas.shape) == 4:
                            fg_frame_atlas.append(foreground_atlas[f])
                        else:
                            fg_frame_atlas.append(foreground_atlas)

                    bg_frame_atlas = background_atlas[f] if len(background_atlas.shape) == 4 else background_atlas

                    alpha, fg_pixels, bg_pixels, _ = self.compute_alpha_and_pixels(
                        xyt_coords=xyt_coords,

                        foreground_model=foreground_model,
                        foreground_texture=fg_frame_atlas,

                        background_model=background_model,
                        background_texture=bg_frame_atlas,
                        background_range=background_range,

                        alpha_model=alpha_model,

                        remove_layer=remove_layer
                    )

                    if self.config.foreground_layers > 1:
                        alpha_reconstruction[batches_y[i], batches_x[i], :, f] = alpha.squeeze().cpu().numpy()
                    else:
                        alpha_reconstruction[batches_y[i], batches_x[i], f] = alpha[:, 0, 0].cpu().numpy()

                    for layer_idx, fg_i_pixels in enumerate(fg_pixels):
                        fg_y_coords = batches_y[i][fg_i_pixels.relevant_idxs]
                        fg_x_coords = batches_x[i][fg_i_pixels.relevant_idxs]
                        foreground_changes[layer_idx, fg_y_coords, fg_x_coords, :, f] = fg_i_pixels.colors

                    bg_y_coords = batches_y[i][bg_pixels.relevant_idxs]
                    bg_x_coords = batches_x[i][bg_pixels.relevant_idxs]
                    background_changes[bg_y_coords, bg_x_coords, :, f] = bg_pixels.colors

                background_texture = background_changes[:, :, :3, f]
                foreground_texture = stacked_run(
                    inputs=range(self.config.foreground_layers),
                    function=lambda idx: foreground_changes[idx, :, :, :3, f]
                )

                background_alpha = background_changes[:, :, 3, f][:, :, np.newaxis]
                foreground_alpha = stacked_run(
                    inputs=range(self.config.foreground_layers),
                    function=lambda idx: foreground_changes[idx, :, :, 3, f][:, :, np.newaxis]
                )

                original_frame = video_frames[:, :, :, f].cpu().clone().numpy()

                edited_background = background_texture * background_alpha + original_frame * (1 - background_alpha)
                edited_foreground = stacked_run(
                    inputs=range(self.config.foreground_layers),
                    function=lambda idx: (foreground_texture[idx] * foreground_alpha[idx]) +
                                         (original_frame * (1 - foreground_alpha[idx]))
                )

                if self.config.foreground_layers > 1:
                    frame_alpha = alpha_reconstruction[:, :, :, f][:, :, np.newaxis]

                    edited_background_term = edited_background * frame_alpha[:, :, :, 0]
                    edited_foreground_term = stacked_run(
                        inputs=range(self.config.foreground_layers),
                        function=lambda idx: edited_foreground[idx] * frame_alpha[:, :, :, idx + 1]
                    )

                    edited_frame = edited_background_term + sum(edited_foreground_term)

                else:
                    frame_alpha = alpha_reconstruction[:, :, f][:, :, np.newaxis]
                    edited_frame = edited_foreground * frame_alpha + edited_background * (1 - frame_alpha)

                video_reconstruction[:, :, :, f] = edited_frame

        return video_reconstruction

    def render_video_from_atlases(self,
                                  foreground_atlases: List[torch.Tensor],
                                  background_atlas: torch.Tensor,

                                  foreground_model: nn.Module,
                                  background_model: nn.Module,
                                  alpha_model: nn.Module,

                                  remove_layer: str = None,
                                  show_progress: bool = True):

        """ Render the Edits in Atlas and Propagate them to the Original Video Frames

        :param foreground_atlases: torch.Tensor: Foreground Atlas Images with User Edits for Each Layer
        :param background_atlas: torch.Tensor: Background Atlas Image with User Edits

        :param foreground_model: nn.Module: Foreground IMLP Mapping Model
        :param background_model: nn.Module: Background IMLP Mapping Model
        :param alpha_model: nn.Module: Alpha IMLP Model

        :param remove_layer: str: Layer to Remove ['foreground', 'background' or None]
        :param show_progress: bool: Show Progress Bar?

        :return: np.array: Reconstructed Video Frames with User Edits Normalized to [0, 255]
        """

        if len(foreground_atlases) != self.config.foreground_layers:
            raise ValueError("The 'foreground_atlases' length need to be equal to the "
                             "number of foreground layers were the model was trained. "
                             f"Trained Foreground Layers: {self.config.foreground_layers}. "
                             "If you want to pass a blank foreground atlas for a multi layer"
                             "model, try using: foreground_atlases=[None, None, ...] ")

        for layer_idx in range(len(foreground_atlases)):
            if foreground_atlases[layer_idx] is None:
                logging.info(f"Disabling foreground editing for layer {layer_idx}...")
                foreground_atlases[layer_idx] = torch.zeros(1000, 1000, 4)

        if remove_layer == "background" and self.config.foreground_layers == 1:
            foreground_atlases[0][:, :, 3].fill(1)

        if background_atlas is None:
            logging.info("Disabling background editing...")
            background_atlas = torch.zeros(1000, 1000, 4)

        frames, masks, _, flows, flow_masks = self.data_manager.load_video()

        background_range, background_edge_size = self.compute_background_range(
            background_model=background_model,
            alpha_model=alpha_model,
            masks=masks > -1,
            num_frames=frames.num_frames,
            invert_alpha=True
        )

        video_reconstruction = self.render_video(
            video_frames=frames.frames,

            foreground_atlases=foreground_atlases,
            background_atlas=background_atlas,

            foreground_model=foreground_model,

            background_model=background_model,
            background_range=background_range,

            alpha_model=alpha_model,

            remove_layer=remove_layer,
            show_progress=show_progress
        )

        return (video_reconstruction * 255).astype(np.uint8)

    def generate_atlases_from_frame(self,
                                    foreground_model: nn.Module,
                                    background_model: nn.Module,
                                    alpha_model: nn.Module,

                                    frame: np.array,
                                    frame_idx: int,

                                    animated: bool = False):

        """ Render an Atlas Image for each Foreground and Background Layers from a Frame Perspective

        :param foreground_model: nn.Module: Foreground IMLP Mapping Model
        :param background_model: nn.Module: Background IMLP Mapping Model
        :param alpha_model: nn.Module: Alpha IMLP Model

        :param frame: np.array: Input Frame to Use as Perspective Reference
        :param frame_idx: int: Index of Input Frame in Video Frames

        :param animated: bool: Generate Animated Foreground/Background Atlases (One per Frame)?

        :return: np.Array: Foreground and Background Atlases from Given Frame Perspective
        """

        if frame.shape[2] == 3:
            # Add Alpha Channel to RGB Frame
            frame_w, frame_h, n_channels = frame.shape
            alpha_channel = np.ones((frame_w, frame_h, 1))
            frame = np.concatenate([frame, alpha_channel], axis=2)

        frame = cv2.resize(frame, (self.config.size[0], self.config.size[1]))

        frames, masks, _, flows, flow_masks = self.data_manager.load_video()

        (x_range, y_range), background_edge_size = self.compute_background_range(
            background_model=background_model,
            alpha_model=alpha_model,
            masks=masks > -1,
            num_frames=frames.num_frames,
            invert_alpha=True
        )

        coords_y, coords_x = torch.where(torch.ones(self.config.size[1], self.config.size[0]) > 0)
        batches_y = self.compute_batches(coords_y)
        batches_x = self.compute_batches(coords_x)

        global_foreground_coords = []
        global_foreground_colors = []

        global_background_coords = []
        global_background_colors = []

        with torch.no_grad():
            for i in range(len(batches_y)):
                norm_y = torch.from_numpy(batches_y[i]).unsqueeze(1) / (self.larger_dim / 2) - 1
                norm_x = torch.from_numpy(batches_x[i]).unsqueeze(1) / (self.larger_dim / 2) - 1
                norm_t = (frame_idx / (frames.num_frames / 2.0) - 1) * torch.ones_like(norm_y)

                xyt_coords = torch.cat((norm_x, norm_y, norm_t), dim=1).to(self.device)

                foreground_uv = foreground_model(xyt_coords)
                background_uv = background_model(xyt_coords)

                fg_coords = stacked_run(
                    inputs=foreground_uv,
                    function=lambda uv: ((uv * 0.5 + 0.5) * 1000).cpu().numpy()
                )

                bg_coords = (background_uv * 0.5 - 0.5) - torch.tensor([[x_range[0], y_range[0]]]).to(self.device)
                bg_coords = (bg_coords / background_edge_size) * 1000
                bg_coords = bg_coords.cpu().numpy()

                global_foreground_coords.append(fg_coords)
                global_background_coords.append(bg_coords)

                alpha = alpha_model(xyt_coords).unsqueeze(1).cpu().numpy()

                if self.config.foreground_layers > 1:
                    global_background_colors.append(frame[batches_y[i], batches_x[i], :] * alpha[:, :, 0])

                    global_foreground_colors.append(
                        stacked_run(
                            inputs=range(self.config.foreground_layers),
                            function=lambda idx: frame[batches_y[i], batches_x[i], :] * alpha[:, :, idx + 1]
                        )
                    )
                else:
                    global_background_colors.append(frame[batches_y[i], batches_x[i], :] * (1 - alpha[:, :, 0]))
                    global_foreground_colors.append(frame[batches_y[i], batches_x[i], :] * alpha[:, :, 0])

        global_background_coords = np.concatenate(global_background_coords)
        global_background_colors = np.concatenate(global_background_colors)

        global_foreground_coords = stacked_run(
            inputs=range(self.config.foreground_layers),
            function=lambda idx: np.concatenate([coords[idx] for coords in global_foreground_coords])
        )

        if self.config.foreground_layers > 1:
            global_foreground_colors = stacked_run(
                inputs=range(self.config.foreground_layers),
                function=lambda idx: np.concatenate([colors[idx] for colors in global_foreground_colors])
            )
        else:
            global_foreground_colors = [np.concatenate(global_foreground_colors)]

        x_values, y_values = np.meshgrid(np.linspace(0, 999, 1000), np.linspace(0, 999, 1000))

        background_atlas = griddata(
            points=global_background_coords,
            values=global_background_colors,
            xi=(x_values, y_values),
            method='linear'
        )
        background_atlas = np.nan_to_num(background_atlas)
        background_atlas = background_atlas.__mul__(255).astype(np.uint8)

        foreground_atlases = stacked_run(
            inputs=range(self.config.foreground_layers),
            function=lambda idx: griddata(
                points=global_foreground_coords[idx],
                values=global_foreground_colors[idx],
                xi=(x_values, y_values),
                method='linear'
            )
        )

        foreground_atlases = stacked_run(
            inputs=foreground_atlases,
            function=lambda foreground_atlas: np.nan_to_num(foreground_atlas).__mul__(255).astype(np.uint8)
        )

        if animated:
            background_atlas = np.array([background_atlas] * frames.num_frames)
            foreground_atlases = stacked_run(
                inputs=foreground_atlases,
                function=lambda foreground_atlas: np.array([foreground_atlas] * frames.num_frames)
            )

        return foreground_atlases, background_atlas

    def generate_atlases(self,
                         foreground_model: nn.Module,
                         background_model: nn.Module,
                         alpha_model: nn.Module,
                         atlas_model: nn.Module,

                         animated: bool = False,
                         debug: bool = False,
                         show_progress: bool = True,
                         dataset: tuple = None):

        """ Render an Atlas Image for each Foreground and Background Layers

        :param foreground_model: nn.Module: Foreground IMLP Mapping Model
        :param background_model: nn.Module: Background IMLP Mapping Model
        :param alpha_model: nn.Module: Alpha IMLP Model
        :param atlas_model: nn.Module: Atlas IMLP Model

        :param animated: bool: Generate Animated Foreground/Background Atlases (One per Frame)?
        :param debug: bool: Render Debug Videos (Alpha/RGB Reconstruction) and Returns with Textures
        :param show_progress: bool: Show Progress Bar?
        :param dataset: Tuple: Pre-loaded Dataset to Speed up the Process

        :return: np.Array: Global Foreground and Background Atlases
        """

        if dataset is None:
            frames, masks, _, flows, flow_masks = self.data_manager.load_video()
        else:
            frames, masks, _, flows, flow_masks = dataset

        foreground_textures = stacked_run(
            inputs=range(self.config.foreground_layers),
            function=lambda idx: self.compute_texture(
                atlas_model=atlas_model,
                xy_range=((0, 1), np.array((0, 1)) - idx),
                resolution=1000
            )
        )

        background_range, background_edge_size = self.compute_background_range(
            background_model=background_model,
            alpha_model=alpha_model,
            masks=masks > -1,
            num_frames=frames.num_frames,
            invert_alpha=True
        )

        background_texture = self.compute_texture(
            atlas_model=atlas_model,
            xy_range=background_range,
            resolution=1000
        )

        foreground_alpha, background_alpha, video_recon, alpha_recon = self.compute_masks(
            foreground_model=foreground_model,
            foreground_texture=foreground_textures,

            background_model=background_model,
            background_texture=background_texture,
            background_range=background_range,

            alpha_model=alpha_model,
            atlas_model=atlas_model,

            num_frames=frames.num_frames,

            resolution=1000,
            render_video=debug,
            show_progress=show_progress
        )

        foreground_atlases = []

        for layer_idx in range(self.config.foreground_layers):
            foreground_layer_alpha = foreground_alpha[layer_idx, :, :, np.newaxis]
            foreground_atlas = foreground_textures[layer_idx] * foreground_layer_alpha
            foreground_atlas = np.concatenate((foreground_atlas, foreground_layer_alpha), axis=2)

            foreground_atlases.append((foreground_atlas * 255).astype(np.uint8))

        background_atlas = np.concatenate((background_texture, background_alpha[:, :, np.newaxis]), axis=2)
        background_atlas = (background_atlas * 255).astype(np.uint8)

        if animated:
            background_atlas = np.array([background_atlas] * frames.num_frames)
            foreground_atlases = stacked_run(
                inputs=foreground_atlases,
                function=lambda atlas: np.array([atlas] * frames.num_frames)
            )

        if debug:
            video_recon = (video_recon * 255).astype(np.uint8)
            alpha_recon = (alpha_recon * 255).astype(np.uint8)

            textures = {
                "foreground_textures": foreground_textures,
                "background_texture": background_texture
            }

            videos = {
                "video_reconstruction": video_recon,
                "alpha_reconstruction": alpha_recon
            }

            return foreground_atlases, background_atlas, (textures, videos)

        else:
            return foreground_atlases, background_atlas
