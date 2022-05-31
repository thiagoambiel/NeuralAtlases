import logging

import cv2

import torch
from torch import nn

from models.raft.raft import RAFT
from models.raft.utils.utils import InputPadder

from argparse import Namespace


class RAFTWrapper:
    def __init__(self,
                 checkpoint_path: str,
                 max_resolution: int,
                 device: torch.device = torch.device("cpu")):

        self.device = device

        self.args = Namespace()

        self.args.small = False
        self.args.mixed_precision = True
        self.args.model = checkpoint_path
        self.args.max_long_edge = max_resolution

        self.model = nn.DataParallel(RAFT(self.args))
        self.model.load_state_dict(torch.load(self.args.model))

        self.model = self.model.module
        self.model.to(self.device)
        self.model.eval()

    def load_image(self, image_path: str):
        image = cv2.imread(f'{image_path}')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if image is None:
            logging.error(f'Error reading file: {image_path}')
            exit(0)

        image_h = image.shape[0]
        image_w = image.shape[1]

        factor = max(image_w, image_h) / self.args.max_long_edge

        if factor > 1:
            new_width = int(image_w // factor)
            new_height = int(image_h // factor)

            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

        image = torch.from_numpy(image).permute(2, 0, 1).float()

        return image

    def load_image_list(self, image_files):
        images = []

        for image_path in sorted(image_files):
            images.append(self.load_image(image_path))

        images = torch.stack(images)
        images = images.to(self.device)

        padder = InputPadder(images.shape)
        return padder.pad(images)[0]

    def load_images(self, images):
        """ load and resize to multiple of 64 """
        images = self.load_image_list(images)
        return images[0, None], images[1, None]

    def compute_flow(self, image, next_image):
        padder = InputPadder(image.shape)
        image, next_image = padder.pad(image, next_image)
        _, flow = self.model(image, next_image, iters=20, test_mode=True)
        flow = flow[0].permute(1, 2, 0).detach().cpu().numpy()

        return flow
