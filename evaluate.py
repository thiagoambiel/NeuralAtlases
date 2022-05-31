import logging

import argparse
from pathlib import Path

import torch
from core.neural_atlases import NeuralAtlases


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Evaluation on a Project with Layered Neural Atlases models')

    parser.add_argument("-p", "--project", type=str, required=True,
                        help="Project name inside projects dir. Ex: 'lucia'")

    parser.add_argument("-d", "--device", type=str, default="cuda", choices=["cpu", "cuda"],
                        help="\nDevice to run the models. (Default: 'cuda')")

    args = parser.parse_args()

    neural_atlases = NeuralAtlases(
        project_path="./projects" / Path(args.project),
        device=torch.device(args.device)
    )

    logging.info("Starting Evaluation...")

    neural_atlases.evaluate(show_progress=True)

    logging.info("Finish! Check the Project 'atlases' and 'debug' folders")
