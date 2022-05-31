from typing import List

import logging
from dataclasses import dataclass

import os
import cv2
import numpy as np

import torch
from models import RAFTWrapper, MODNet, MaskRCNN

from pathlib import Path

from core.config import Config

from core.utils import read_image, resize_flow
from core.utils import compute_consistency

from tqdm import tqdm


@dataclass
class Flow:
    forward: torch.Tensor
    backward: torch.Tensor


@dataclass
class Frames:
    frames: torch.Tensor
    derivative_x: torch.Tensor
    derivative_y: torch.Tensor
    num_frames: int


class DataManager:
    def __init__(self,
                 config: Config,
                 project_path: Path,
                 larger_dim: np.int64,
                 device: torch.device = torch.device("cpu")):

        self.device = device
        self.config = config
        self.larger_dim = larger_dim

        self.project_path = project_path

        os.makedirs(self.project_path / 'results', exist_ok=True)

    def find_images(self):
        """ Find Video Frames and Return them in Order """

        return sorted((self.project_path / 'frames').glob("*.*"))

    def preprocess_masks(self):
        """ Preprocess Foreground Masks of each Video Frame """

        images = self.find_images()

        output_path = self.project_path / 'results' / 'masks'
        output_path.mkdir(exist_ok=True)

        if self.config.foreground_class == 'modnet':
            model = MODNet(checkpoint='./models/weights/modnet.pth').to(self.device)
        else:
            model = MaskRCNN(class_name=self.config.foreground_class)

        for idx, image_path in enumerate(tqdm(images, desc="Computing Masks")):
            image = read_image(image_path, normalize=False)

            mask = model(image)

            cv2.imwrite(f"{output_path}/%05d.png" % idx, mask)

        del model
        torch.cuda.empty_cache()

    def preprocess_optical_flow(self):
        """ Preprocess Forward and Backward Optical Flows for Each Video Frame Pairs """

        images = self.find_images()

        output_path = self.project_path / 'results' / "optical_flow"
        output_path.mkdir(exist_ok=True)

        model = RAFTWrapper(
            checkpoint_path="./models/weights/raft.pth",
            max_resolution=self.larger_dim,
            device=self.device
        )

        for file_idx in tqdm(range(len(images) - 1), desc="Computing Optical Flow"):
            image_file = images[file_idx]
            next_image_file = images[file_idx + 1]

            image, next_image = model.load_images([image_file, next_image_file])

            forward_flow = model.compute_flow(image, next_image)
            backward_flow = model.compute_flow(next_image, image)

            np.save(
                file=str(output_path / f'{image_file.name}_{next_image_file.name}.npy'),
                arr=forward_flow
            )

            np.save(
                file=str(output_path / f'{next_image_file.name}_{image_file.name}.npy'),
                arr=backward_flow
            )

        del model
        torch.cuda.empty_cache()

    def load_frames(self):
        """ Load Video Frames and Compute Derivatives """

        size_x = self.config.size[0]
        size_y = self.config.size[1]
        max_frames = self.config.max_frames

        images = self.find_images()

        num_frames = min(len(images), max_frames)

        frames = torch.zeros((size_y, size_x, 3, num_frames))
        frames_derivative_x = torch.zeros((size_y, size_x, 3, num_frames))
        frames_derivative_y = torch.zeros((size_y, size_x, 3, num_frames))

        for idx in range(num_frames):
            image = read_image(images[idx])

            frames[:, :, :, idx] = torch.from_numpy(cv2.resize(image[:, :, :3], (size_x, size_y)))
            frames_derivative_y[:-1, :, :, idx] = frames[1:, :, :, idx] - frames[:-1, :, :, idx]
            frames_derivative_x[:, :-1, :, idx] = frames[:, 1:, :, idx] - frames[:, :-1, :, idx]

        frames = Frames(
            frames=frames,
            derivative_x=frames_derivative_x,
            derivative_y=frames_derivative_y,
            num_frames=np.int64(frames.shape[3])
        )

        return frames

    def load_mask(self):
        """ Load Precomputed Masks of each Video Frame """

        size_x = self.config.size[0]
        size_y = self.config.size[1]
        max_frames = self.config.max_frames

        mask_files = sorted((self.project_path / 'results' / 'masks').glob("*.png"))

        video_size = min(len(mask_files), max_frames)

        masks = torch.zeros((size_y, size_x, video_size))

        for idx in range(video_size):
            mask = read_image(mask_files[idx])
            masks[:, :, idx] = torch.from_numpy(cv2.resize(mask, (size_x, size_y), cv2.INTER_NEAREST))

        return masks

    def load_scribbles(self):
        """ Load User Annotated Scribbles for Multi Foreground Layer Splitting """

        size_x = self.config.size[0]
        size_y = self.config.size[1]
        max_frames = self.config.max_frames

        scribble_files = sorted((self.project_path / "scribbles").glob("*.png"))

        scribbles_colors = []
        scribbles = torch.zeros((size_y, size_x, 3, max_frames))

        for file in scribble_files:
            frame_idx = os.path.basename(file)
            frame_idx = int(os.path.splitext(frame_idx)[0])

            scribble = read_image(file)[:, :, :3]
            scribbles[:, :, :, frame_idx] = torch.from_numpy(cv2.resize(scribble, (size_x, size_y), cv2.INTER_NEAREST))

            colors = self.extract_colors(scribble)

            for color in colors:
                if color not in scribbles_colors:
                    scribbles_colors.append(color)

        if self.config.foreground_layers > 1 and not scribble_files:
            logging.warning("The multi layer feature is enabled but no scribble files were "
                            "found. This can produce unexpected results when splitting layers. "
                            "For better results, provide a image with scribbles for each video frame.")
        else:
            logging.info(f"Found {len(scribbles_colors)} Scribble Color Variants...")

        if len(scribbles_colors) > self.config.foreground_layers:
            logging.warning("Were found more scribble colors than the number of foreground layers. "
                            f"Using only the first {self.config.foreground_layers} colors.")

            scribbles_colors = scribbles_colors[:self.config.foreground_layers]

        return scribbles, scribbles_colors

    @staticmethod
    def extract_colors(img: np.array) -> List[np.array]:
        """ Find All Colors that Compose the User Scribbles """

        colors = img.reshape(-1, 3)
        colors = colors[(colors != np.array([0, 0, 0])).any(axis=1)]
        colors = np.unique(colors, axis=0).astype(int)

        return colors.tolist()

    def load_optical_flow(self, threshold: int = 1.0):
        """ Load Precomputed Optical Flow of each Video Frame and Compute Flow Consistency"""

        size_x = self.config.size[0]
        size_y = self.config.size[1]
        max_frames = self.config.max_frames

        images = self.find_images()

        video_size = min(len(images), max_frames)

        forward_flows = torch.zeros((size_y, size_x, 2, video_size, 1))
        backward_flows = torch.zeros((size_y, size_x, 2, video_size, 1))

        forward_flow_masks = torch.zeros((size_y, size_x, video_size, 1))
        backward_flow_masks = torch.zeros((size_y, size_x, video_size, 1))

        for idx in range(video_size - 1):
            image = images[idx]
            next_image = images[idx + 1]

            forward_flow = self.project_path / 'results' / "optical_flow" / f'{image.name}_{next_image.name}.npy'
            backward_flow = self.project_path / 'results' / "optical_flow" / f'{next_image.name}_{image.name}.npy'

            forward_flow = np.load(str(forward_flow))
            backward_flow = np.load(str(backward_flow))

            if forward_flow.shape[0] != size_y and forward_flow[1].shape != size_x:
                forward_flow = resize_flow(forward_flow, new_h=size_y, new_w=size_x)
                backward_flow = resize_flow(backward_flow, new_h=size_y, new_w=size_x)

            forward_flow_mask = compute_consistency(forward_flow, backward_flow) < threshold
            backward_flow_mask = compute_consistency(backward_flow, forward_flow) < threshold

            forward_flows[:, :, :, idx, 0] = torch.from_numpy(forward_flow)
            backward_flows[:, :, :, idx + 1, 0] = torch.from_numpy(backward_flow)

            forward_flow_masks[:, :, idx, 0] = torch.from_numpy(forward_flow_mask)
            backward_flow_masks[:, :, idx + 1, 0] = torch.from_numpy(backward_flow_mask)

        flows = Flow(
            forward=forward_flows,
            backward=backward_flows
        )

        masks = Flow(
            forward=forward_flow_masks,
            backward=backward_flow_masks
        )

        return flows, masks

    def load_video(self):
        """ Load Video Dataset (RGB Frames, Foreground Masks, Optical Flows and User Scribbles) """

        frames = self.load_frames()

        if frames.frames.shape[3] == 0:
            raise FileNotFoundError(f'No frames were found in: '
                                    f'"{self.project_path / "frames"}"')

        masks = self.load_mask()

        if masks.shape[2] == 0:
            raise FileNotFoundError(f'No alpha masks were found in: '
                                    f'"{self.project_path / "results" / "masks"}"')

        flows, flow_masks = self.load_optical_flow()

        if flows.forward.shape[3] == 0:
            raise FileNotFoundError(f'No optical flow files were found in: '
                                    f'"{self.project_path / "results" / "optical_flow"}"')

        scribbles, scribbles_colors = self.load_scribbles()

        return frames, masks, (scribbles, scribbles_colors), flows, flow_masks
