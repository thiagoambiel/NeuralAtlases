import torch
from torch import nn

import numpy as np

from core.config import Config
from core.utils import stacked_run

from core.norms import squared_euclidean_norm, taxicab_norm
from core.norms import foreground_range_norm, alpha_norm


class Losses:
    def __init__(self,
                 config: Config,
                 larger_dim: np.int64,
                 device: torch.device = torch.device("cpu")):

        self.config = config
        self.larger_dim = larger_dim
        self.device = device

    def gradient_loss(self,
                      frames_derivative_x: torch.Tensor,
                      frames_derivative_y: torch.Tensor,
                      pixel_coords: torch.Tensor,
                      rgb_output: torch.Tensor,

                      foreground_model: nn.Module,
                      background_model: nn.Module,
                      atlas_model: nn.Module,
                      alpha_model: nn.Module,

                      num_frames: np.int64):

        """ Compute Gradient Loss (Eq. 7 in Paper)

        :param frames_derivative_x: torch.Tensor: X-Axis Spatial Derivatives
        :param frames_derivative_y: torch.Tensor: Y-Axis Spatial Derivatives
        :param pixel_coords: torch.Tensor: Batch of Pixel Coordinates
        :param rgb_output: torch.Tensor: RGB Pixel Reconstruction

        :param foreground_model: nn.Module: Foreground IMLP Mapping Model
        :param background_model: nn.Module: Background IMLP Mapping Model
        :param atlas_model: nn.Module: Atlas IMLP Model
        :param alpha_model: nn.Module: Alpha IMLP Model

        :param num_frames: np.int64: Number of Project Video Frames

        :return: gradient_loss: torch.Tensor: Computed Differentiable Gradient Loss
        """

        pixel_coords_x = torch.cat(
            [
                (pixel_coords[0, :] + 1) / (self.larger_dim / 2) - 1,
                pixel_coords[1, :] / (self.larger_dim / 2) - 1,
                pixel_coords[2, :] / (num_frames / 2.0) - 1
            ],
            dim=1).to(self.device)

        pixel_coords_y = torch.cat(
            [
                pixel_coords[0, :] / (self.larger_dim / 2) - 1,
                (pixel_coords[1, :] + 1) / (self.larger_dim / 2) - 1,
                pixel_coords[2, :] / (num_frames / 2.0) - 1
            ],
            dim=1).to(self.device)

        # precomputed discrete derivative with respect to x,y direction
        rgb_derivative_x_gt = frames_derivative_x[pixel_coords[1, :], pixel_coords[0, :], :, pixel_coords[2, :]]
        rgb_derivative_x_gt = rgb_derivative_x_gt[:, 0, :].to(self.device)

        rgb_derivative_y_gt = frames_derivative_y[pixel_coords[1, :], pixel_coords[0, :], :, pixel_coords[2, :]]
        rgb_derivative_y_gt = rgb_derivative_y_gt[:, 0, :].to(self.device)

        # compute alpha values for x,y coords
        alpha_x = alpha_model(pixel_coords_x).unsqueeze(1)
        alpha_y = alpha_model(pixel_coords_y).unsqueeze(1)

        # uv coordinates for locations with offsets of 1 pixel
        uv_foreground_x = foreground_model(pixel_coords_x)
        uv_foreground_y = foreground_model(pixel_coords_y)

        uv_background_x = background_model(pixel_coords_x)
        uv_background_y = background_model(pixel_coords_y)

        # The RGB values (from the 2 layers) for locations with offsets of 1 pixel
        rgb_background_x = atlas_model(uv_background_x * 0.5 - 0.5)
        rgb_foreground_x = stacked_run(
            inputs=enumerate(uv_foreground_x),
            function=lambda x: atlas_model(foreground_range_norm(x[1], x[0]))
        )

        rgb_background_y = atlas_model(uv_background_y * 0.5 - 0.5)
        rgb_foreground_y = stacked_run(
            inputs=enumerate(uv_foreground_y),
            function=lambda x: atlas_model(foreground_range_norm(x[1], x[0]))
        )

        # Reconstructed RGB values:
        if self.config.foreground_layers > 1:
            background_term_x = alpha_x[:, :, 0] * rgb_background_x
            foreground_term_x = stacked_run(
                inputs=range(len(rgb_foreground_x)),
                function=lambda idx: alpha_x[:, :, idx + 1] * rgb_foreground_x[idx]
            )

            background_term_y = alpha_y[:, :, 0] * rgb_background_y
            foreground_term_y = stacked_run(
                inputs=range(len(rgb_foreground_y)),
                function=lambda idx: alpha_y[:, :, idx + 1] * rgb_foreground_y[idx]
            )

            rgb_output_x = background_term_x + sum(foreground_term_x)
            rgb_output_y = background_term_y + sum(foreground_term_y)

        else:
            rgb_output_x = (1.0 - alpha_x[:, :, 0]) * rgb_background_x + alpha_x[:, :, 0] * rgb_foreground_x[0]
            rgb_output_y = (1.0 - alpha_y[:, :, 0]) * rgb_background_y + alpha_y[:, :, 0] * rgb_foreground_y[0]

        # Use reconstructed RGB values for computing derivatives:
        rgb_derivative_x_pred = rgb_output_x - rgb_output
        rgb_derivative_y_pred = rgb_output_y - rgb_output

        # Eq. 7 in the Paper.
        rgb_derivative_x_error = squared_euclidean_norm(rgb_derivative_x_gt - rgb_derivative_x_pred)
        rgb_derivative_y_error = squared_euclidean_norm(rgb_derivative_y_gt - rgb_derivative_y_pred)

        gradient_loss = torch.mean(rgb_derivative_x_error + rgb_derivative_y_error)

        return gradient_loss

    def rigidity_loss(self,
                      uv_map: torch.Tensor,
                      pixel_coords: torch.Tensor,

                      mapping_model: nn.Module,

                      num_frames: np.int64,
                      derivative_amount: np.int64,

                      uv_mapping_scale: np.float64 = 1.0):

        """ Compute Rigidity Loss (Eq. 9 in Paper)

        :param uv_map: torch.Tensor: Foreground/Background UV Map
        :param pixel_coords: torch.Tensor: Batch of Pixel Coordinates

        :param mapping_model: nn.Module: Foreground/Background IMLP Mapping Model

        :param num_frames: np.int64: Number of Project Video Frames
        :param derivative_amount: np.int64: Rigidity Derivative Amount

        :param uv_mapping_scale: np.float64: Foreground/Background UV Mapping Scale

        :return: rigidity_loss: Computed Differentiable Rigidity Loss
        """

        # concatenating (x,y-derivative_amount,t) and (x-derivative_amount,y,t) to get xyt_patch:
        y_patch = torch.cat((pixel_coords[1, :] - derivative_amount, pixel_coords[1, :])) / (self.larger_dim / 2) - 1
        x_patch = torch.cat((pixel_coords[0, :], pixel_coords[0, :] - derivative_amount)) / (self.larger_dim / 2) - 1
        t_patch = torch.cat((pixel_coords[2, :], pixel_coords[2, :])) / (num_frames / 2.0) - 1

        xyt_patch = torch.cat((x_patch, y_patch, t_patch), dim=1).to(self.device)

        uv_patch = mapping_model(xyt_patch)

        # u_patch[0,:]= u(x,y-derivative_amount,t).  u_patch[1,:]= u(x-derivative_amount,y,t)
        u_patch = uv_patch[:, 0].view(2, -1)

        # v_patch[0,:]= u(x,y-derivative_amount,t).  v_patch[1,:]= v(x-derivative_amount,y,t)
        v_patch = uv_patch[:, 1].view(2, -1)

        # u_p_d_[0,:]=u(x,y,t)-u(x,y-derivative_amount,t)   u_p_d_[1,:]= u(x,y,t)-u(x-derivative_amount,y,t).
        u_p_d_ = uv_map[:, 0].unsqueeze(0) - u_patch

        # v_p_d_[0,:]=u(x,y,t)-v(x,y-derivative_amount,t).  v_p_d_[1,:]= u(x,y,t)-v(x-derivative_amount,y,t).
        v_p_d_ = uv_map[:, 1].unsqueeze(0) - v_patch

        # to match units: 1 in uv coordinates is resx/2 in image space.
        du_dx = u_p_d_[1, :] * self.larger_dim / 2
        du_dy = u_p_d_[0, :] * self.larger_dim / 2
        dv_dy = v_p_d_[0, :] * self.larger_dim / 2
        dv_dx = v_p_d_[1, :] * self.larger_dim / 2

        jacobians = torch.cat((
            torch.cat((du_dx.unsqueeze(-1).unsqueeze(-1), du_dy.unsqueeze(-1).unsqueeze(-1)), dim=2),
            torch.cat((dv_dx.unsqueeze(-1).unsqueeze(-1), dv_dy.unsqueeze(-1).unsqueeze(-1)), dim=2)
        ), dim=1)

        jacobians = jacobians / uv_mapping_scale
        jacobians = jacobians / derivative_amount

        # Apply a loss to constrain the Jacobian to be a rotation matrix as much as possible
        JtJ = torch.matmul(jacobians.transpose(1, 2), jacobians)

        a = JtJ[:, 0, 0] + 0.001
        b = JtJ[:, 0, 1]
        c = JtJ[:, 1, 0]
        d = JtJ[:, 1, 1] + 0.001

        JTJ_inverse = torch.zeros_like(jacobians).to(self.device)
        JTJ_inverse[:, 0, 0] = d
        JTJ_inverse[:, 0, 1] = -b
        JTJ_inverse[:, 1, 0] = -c
        JTJ_inverse[:, 1, 1] = a
        JTJ_inverse = JTJ_inverse / ((a * d - b * c).unsqueeze(-1).unsqueeze(-1))

        # Eq. 9 in the Paper:
        rigidity_loss = (JtJ ** 2).sum(1).sum(1).sqrt() + (JTJ_inverse ** 2).sum(1).sum(1).sqrt()

        return rigidity_loss.mean()

    def optical_flow_loss(self,
                          mapping_model: nn.Module,

                          pixel_coords: torch.Tensor,
                          uv_map: torch.Tensor,

                          optical_flows_forward: torch.Tensor,
                          optical_flows_forward_mask: torch.Tensor,

                          optical_flows_backward: torch.Tensor,
                          optical_flows_backward_mask: torch.Tensor,

                          alpha: torch.Tensor,

                          num_frames: np.int64,
                          uv_mapping_scale: np.float64):

        """ Compute Optical Flow Loss (Eq. 11 in Paper)

        :param mapping_model: nn.Module: Foreground/Background IMLP Mapping Model

        :param pixel_coords: torch.Tensor: Batch of Pixel Coordinates
        :param uv_map: torch.Tensor: Foreground/Background UV Map

        :param optical_flows_forward: torch.Tensor: Forward Optical Flow Vectors
        :param optical_flows_forward_mask: torch.Tensor: Forward Optical Flow Masks

        :param optical_flows_backward: torch.Tensor: Backward Optical Flow Vectors
        :param optical_flows_backward_mask: torch.Tensor: Backward Optical Flow Masks

        :param alpha: torch.Tensor: Alpha IMLP Model Output for Current Layer

        :param num_frames: np.int64: Number of Project Video Frames
        :param uv_mapping_scale:np.float64: Foreground/Background UV Mapping Scale

        :return: flow_loss: torch.Tensor: Computed Differentiable Flow Loss
        """

        uv_map_forward, pixel_coords_forward, relevant_indices_forward = self.compute_flow_matching_points(
            pixel_coords=pixel_coords,
            optical_flows=optical_flows_forward,
            optical_flows_mask=optical_flows_forward_mask,
            uv_map=uv_map,
            num_frames=num_frames,
            is_forward=True,
        )

        uv_map_backward, pixel_coords_backward, relevant_indices_backward = self.compute_flow_matching_points(
            pixel_coords=pixel_coords,
            optical_flows=optical_flows_backward,
            optical_flows_mask=optical_flows_backward_mask,
            num_frames=num_frames,
            is_forward=False,
            uv_map=uv_map,
        )

        uv_flow_forward = mapping_model(pixel_coords_forward.to(self.device))
        uv_flow_backward = mapping_model(pixel_coords_backward.to(self.device))

        # Eq. 11 in the Paper. (Term 1)
        flow_loss_forward = taxicab_norm(uv_flow_forward - uv_map_forward) * self.larger_dim / (2 * uv_mapping_scale)
        flow_loss_forward = (flow_loss_forward * alpha[relevant_indices_forward].squeeze()).mean()

        # Eq. 11 in the Paper. (Term 2)
        flow_loss_backward = taxicab_norm(uv_flow_backward - uv_map_backward) * self.larger_dim / (2 * uv_mapping_scale)
        flow_loss_backward = (flow_loss_backward * alpha[relevant_indices_backward].squeeze()).mean()

        # Eq. 11 in the Paper
        flow_loss = (flow_loss_forward + flow_loss_backward) * 0.5

        return flow_loss

    def optical_flow_alpha_loss(self,
                                alpha_model: nn.Module,
                                pixel_coords: torch.Tensor,
                                alpha: torch.Tensor,

                                optical_flows_forward: torch.Tensor,
                                optical_flows_forward_mask: torch.Tensor,

                                optical_flows_backward: torch.Tensor,
                                optical_flows_backward_mask: torch.Tensor,

                                num_frames: np.int64):

        """ Compute Optical Flow Alpha Loss (Eq. 12 in Paper)

        :param alpha_model: nn.Module: Alpha IMLP Model
        :param pixel_coords: torch.Tensor: Batch of Pixel Coordinates
        :param alpha: torch.Tensor: Alpha IMLP Model Output

        :param optical_flows_forward: torch.Tensor: Forward Optical Flow Vectors
        :param optical_flows_forward_mask: torch.Tensor: Forward Optical Flow Masks

        :param optical_flows_backward: torch.Tensor: Backward Optical Flow Vectors
        :param optical_flows_backward_mask: torch.Tensor: Backward Optical Flow Masks

        :param num_frames: np.int64: Number of Project Video Frames

        :return: alpha_flow_error: torch.Tensor: Computed Differentiable Optical Flow Alpha Loss
        """

        pixel_coords_forward, relevant_indices_forward = self.compute_flow_matching_points(
            pixel_coords=pixel_coords,
            optical_flows=optical_flows_forward,
            optical_flows_mask=optical_flows_forward_mask,
            num_frames=num_frames,
            is_forward=True,
            use_uv=False
        )

        pixel_coords_backward, relevant_indices_backward = self.compute_flow_matching_points(
            pixel_coords=pixel_coords,
            optical_flows=optical_flows_backward,
            optical_flows_mask=optical_flows_backward_mask,
            num_frames=num_frames,
            is_forward=False,
            use_uv=False
        )

        # Eq. 12 in the Paper (Forward).
        alpha_forward_flow = alpha_model(pixel_coords_forward.to(self.device)).unsqueeze(1)
        flow_alpha_loss_forward = (alpha[relevant_indices_forward] - alpha_forward_flow).abs().mean()

        # Eq. 12 in the Paper (Backward).
        alpha_backward_flow = alpha_model(pixel_coords_backward.to(self.device)).unsqueeze(1)
        flow_alpha_loss_backward = (alpha_backward_flow - alpha[relevant_indices_backward]).abs().mean()

        return (flow_alpha_loss_forward + flow_alpha_loss_backward) * 0.5

    def compute_flow_matching_points(self,
                                     pixel_coords: torch.Tensor,
                                     optical_flows_mask: torch.Tensor,
                                     optical_flows: torch.Tensor,
                                     num_frames: np.int64,
                                     is_forward: bool,
                                     uv_map: torch.Tensor = None,
                                     use_uv: bool = True):

        """ Compute Optical Flow Matching Points

        :param pixel_coords: torch.Tensor: Batch of Pixel Coordinates

        :param optical_flows_mask: torch.Tensor: Forward/Backward Optical Flow Masks
        :param optical_flows: torch.Tensor: Forward/Backward Optical Flow Vectors

        :param num_frames: np.int64: Number of Project Video Frames
        :param is_forward: bool: Is Current Optical Flow Array a Forward Flow?
        :param uv_map: torch.Tensor: Foreground/Background UV Map
        :param use_uv: bool: Return UV Map Matching Points?

        :return: Relevant Pixel Coordinates and Indices
        """

        forward_mask = torch.where(
            optical_flows_mask[
                pixel_coords[1, :, 0],
                pixel_coords[0, :, 0],
                pixel_coords[2, :, 0], :]
        )

        forward_frames_amount = 2 ** forward_mask[1]
        relevant_batch_indices = forward_mask[0]

        relevant_coords = pixel_coords[:, relevant_batch_indices, 0]
        forward_flows = optical_flows[relevant_coords[1], relevant_coords[0], :, relevant_coords[2], forward_mask[1]]

        if is_forward:
            pixel_coords_flow = torch.stack(
                (relevant_coords[0] + forward_flows[:, 0],
                 relevant_coords[1] + forward_flows[:, 1],
                 relevant_coords[2] + forward_frames_amount)
            )
        else:
            pixel_coords_flow = torch.stack(
                (relevant_coords[0] + forward_flows[:, 0],
                 relevant_coords[1] + forward_flows[:, 1],
                 relevant_coords[2] - forward_frames_amount)
            )

        norm_pixel_coords_flow = torch.stack((
            pixel_coords_flow[0] / (self.larger_dim / 2) - 1,
            pixel_coords_flow[1] / (self.larger_dim / 2) - 1,
            pixel_coords_flow[2] / (num_frames / 2) - 1)).T

        if use_uv:
            uv_map_flow = uv_map[forward_mask[0]]
            return uv_map_flow, norm_pixel_coords_flow, relevant_batch_indices

        else:
            return norm_pixel_coords_flow, relevant_batch_indices

    def sparsity_loss(self,
                      atlas_model: nn.Module,
                      batch_size: int):

        """ Compute Sparsity Loss (Eq. 14 in Paper)

        Note: Special Thanks to Yoni Kasten for helping me on
         Multi Foreground Layer Sparsity Loss Implementation.

         (https://github.com/ykasten/layered-neural-atlases/issues/6)

        :param atlas_model: nn.Module: Atlas IMLP Model
        :param batch_size: int: Sparsity Loss Multi Foreground Layer Batch Size

        :return: sparsity_loss: torch.Tensor: Computed Differentiable Sparsity Loss
        """

        random_uv_coords = []

        for layer_idx in range(self.config.foreground_layers):
            random_coords = torch.rand(batch_size // self.config.foreground_layers, 2, device=self.device)
            random_coords[:, 1] = random_coords[:, 1] - layer_idx

            random_uv_coords.append(random_coords)

        random_uv_coords = torch.cat(random_uv_coords, dim=0)
        random_foreground_rgb = atlas_model(random_uv_coords)

        return random_foreground_rgb.mean()

    def scribble_loss(self,
                      scribbles: torch.Tensor,
                      alpha: torch.Tensor,
                      color: tuple):

        """ Compute Scribbles Loss (Eq. 20-21 in Paper)

        :param scribbles: torch.Tensor: Batch of Scribbles Colors
        :param alpha: torch.Tensor: Alpha IMLP Model Output for Current Layer
        :param color: Tuple: A Tuple Indicating the RGB Values for Current Layer Color

        :return: scribbles_loss: torch.Tensor: Computed Differentiable Scribbles Loss
        """

        scribbles = (scribbles == torch.tensor(color, device=self.device)).all(dim=1)

        if scribbles.any():
            return -torch.log(alpha[scribbles] + 1e-8).mean()

        return 0

    def alpha_bootstrapping_loss(self,
                                 alpha: torch.Tensor,
                                 target_alpha: torch.Tensor,
                                 normalize: bool = False):

        """ Compute Alpha Bootstrapping Loss

        :param alpha: torch.Tensor: torch.Tensor: Alpha IMLP Model Output
        :param target_alpha: torch.Tensor: Mask Values from MaskRCNN/MODNet Model
        :param normalize: bool: Normalize Alpha IMLP Output?

        :return: alpha_bootstrapping_loss: torch.Tensor: Computed Differentiable Alpha Bootstrapping Loss
        """

        if self.config.foreground_layers > 1:
            foreground_alpha = alpha[:, :, -self.config.foreground_layers:].sum(dim=2)
        else:
            foreground_alpha = alpha[:, :, 0]

        if normalize:
            foreground_alpha = alpha_norm(foreground_alpha)

        return torch.mean(-target_alpha * torch.log(foreground_alpha) -
                          (1 - target_alpha) * torch.log(1 - foreground_alpha))
