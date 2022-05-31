import torch

import argparse
from pathlib import Path

from core.neural_atlases import NeuralAtlases


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train the Layered Neural Atlases models for a given video')

    parser.add_argument("-p", "--project", type=str, required=True,
                        help="Project name inside projects dir. Ex: 'lucia'")

    parser.add_argument("-s", "--steps", type=int, default=400_000,
                        help="Number of steps to train the models. (Default: 400_000)")

    parser.add_argument("-ps", "--pretrain_steps", type=int, default=100,
                        help="Number of steps to pretrain the foreground "
                             "and background models. (Default: 100)")

    parser.add_argument("-c", "--config", type=str,
                        help="Path to a custom config file ('.json') with "
                             "algorithm hyper-parameters. (Optional)")

    parser.add_argument("-d", "--device", type=str, default="cuda", choices=["cpu", "cuda"],
                        help="\nDevice to run the models. (Default: 'cuda')")

    args = parser.parse_args()

    neural_atlases = NeuralAtlases(
        config=args.config,
        project_path="projects" / Path(args.project),
        device=torch.device(args.device)
    )

    neural_atlases.train(steps=args.steps, pretrain_steps=args.pretrain_steps)
