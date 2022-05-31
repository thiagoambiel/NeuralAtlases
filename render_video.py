import logging
import os.path

import torch

import argparse
from pathlib import Path

from core.neural_atlases import NeuralAtlases
from core.utils import read_atlas, save_video


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Render video from edited atlases.')

    parser.add_argument("-p", "--project", type=str, required=True,
                        help="Project name inside projects dir. Ex: 'lucia'")

    parser.add_argument("-f", "--fg_atlas", type=str, nargs='+',
                        help="Path to edited foreground atlas.")

    parser.add_argument("-b", "--bg_atlas", type=str,
                        help="Path to edited background atlas.")

    parser.add_argument("-r", "--remove", type=str, choices=["foreground", "background"],
                        help="Remove one layer from render. (Inpaint or Isolate Foreground)")

    parser.add_argument("-fps", "--frame_rate", type=int,
                        help="Frame Rate (FPS) to save the output video.")

    parser.add_argument("-o", "--output", type=str,
                        help="Path to save the output rendered video.")

    parser.add_argument("-d", "--device", type=str, default="cuda", choices=["cpu", "cuda"],
                        help="\nDevice to run the models. (Default: 'cuda')")

    args = parser.parse_args()

    if not args.fg_atlas and not args.bg_atlas and not args.remove:
        logging.warning("No edited atlas has been passed! This will result in a copy of the original video")
        response = input("Do you want to continue? [y/N]: ")

        if response.lower() == "n" or response == "":
            logging.info("Exiting...")
            exit()

    neural_atlases = NeuralAtlases(
        project_path="projects" / Path(args.project),
        device=torch.device(args.device)
    )

    if args.fg_atlas and len(args.fg_atlas) > neural_atlases.config.foreground_layers:
        raise ValueError(f"{len(args.fg_atlas)} Foreground Layers were passed, but the selected "
                         f"project only supports {neural_atlases.config.foreground_layers} layers")

    foreground_atlases = [None for i in range(neural_atlases.config.foreground_layers)]

    if args.fg_atlas:
        for layer_idx, atlas in enumerate(args.fg_atlas):
            if atlas.lower() != "none":
                foreground_atlases[layer_idx] = read_atlas(atlas)

    background_atlas = read_atlas(args.bg_atlas)

    # Check if a background atlas file need to be loaded
    if args.remove == "foreground" and background_atlas is None:
        background_atlas = read_atlas(
            "./projects" / Path(args.project) / "results" / "atlases" / "background_atlas.png")

        if background_atlas is None:
            logging.error(f"Could not find a background atlas in the project folder. To remove the foreground "
                          f"you will need to pass the path to a background atlas file or run the 'evaluate.py' script.")

            logging.info("Exiting...")
            exit()

    # Check if a foreground atlas file need to be loaded
    if args.remove == "background" and foreground_atlases[0] is None and neural_atlases.config.foreground_layers == 1:
        foreground_atlases[0] = read_atlas(
            "./projects" / Path(args.project) / "results" / "atlases" / "foreground_atlas_0.png")

        if foreground_atlases[0] is None:
            logging.error(f"Could not find a foreground atlas in the project folder. To remove the background "
                          f"you will need to pass the path to a foreground atlas file or run the 'evaluate.py' script.")

            logging.info("Exiting...")
            exit()

    video = neural_atlases.render_video_from_atlases(
        foreground_atlases=foreground_atlases,
        background_atlas=background_atlas,
        remove_layer=args.remove
    )

    if args.output is not None and Path(os.path.dirname(args.output)).exists():
        output = args.output
    else:
        if args.output is not None:
            logging.warning("Could not find the desired output folder. "
                            "The result will be saved in the default path...")

        output = "./projects" / Path(args.project) / "results" / "edited_video.mp4"

    logging.info(f"Saving video to '{output}'")

    frame_rate = 10 if not args.frame_rate else args.frame_rate
    save_video(video, save_path=str(output), fps=frame_rate)

    logging.info("Finish!")
