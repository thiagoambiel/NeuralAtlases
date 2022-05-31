import logging

import os
import torch

import argparse
from pathlib import Path

from core.neural_atlases import NeuralAtlases
from core.utils import read_image, save_atlas


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate atlases with Layered Neural Atlases models')

    parser.add_argument("-p", "--project", type=str, required=True,
                        help="Project name inside projects dir. Ex: 'lucia'")

    parser.add_argument("-cf", "--custom_frame", type=Path,
                        help="Path to a edited custom frame with the frame number in the name. Ex: 00001.jpg")

    parser.add_argument("-a", '--animated', action='store_true',
                        help="Generate animated atlases to enable animation support.")

    parser.add_argument("-f", '--fill', action='store_true',
                        help="Remove foreground layers transparency by disabling alpha channel.")

    parser.add_argument("-s", '--save_dir', type=Path,
                        help="Directory to save the generated atlases.")

    parser.add_argument("-d", "--device", type=str, default="cuda", choices=["cpu", "cuda"],
                        help="\nDevice to run the models. (Default: 'cuda')")

    args = parser.parse_args()

    neural_atlases = NeuralAtlases(
        project_path="./projects" / Path(args.project),
        device=torch.device(args.device)
    )

    frame_idx = 0

    if args.custom_frame is not None:
        if not args.custom_frame.exists():
            raise FileNotFoundError(f"The file '{args.custom_frame}' cannot be found.")

        frame = read_image(args.custom_frame)
        frame_idx = os.path.basename(args.custom_frame)

        try:
            frame_idx = int(os.path.splitext(frame_idx)[0])
        except ValueError:
            raise ValueError("The custom frame file name has an invalid format. Its name "
                             "must be the respective frame number on video. (e.g. 00001.jpg)")

        foreground_atlases, background_atlas = neural_atlases.generate_atlases_from_frame(
            frame=frame,
            frame_idx=frame_idx,
            animated=args.animated
        )

    else:
        foreground_atlases, background_atlas = neural_atlases.generate_atlases(animated=args.animated)

    if args.fill:
        # Fill Alpha Layer Values with Max Value (255)
        for foreground_atlas in foreground_atlases:
            if args.animated:
                foreground_atlas[:, :, :, 3].fill(255)
            else:
                foreground_atlas[:, :, 3].fill(255)

    output = "./projects" / Path(args.project) / 'results' / ("animated_atlases" if args.animated else "atlases")
    output = args.save_dir if args.save_dir is not None else output

    if not output.exists():
        output.mkdir(exist_ok=True)

    logging.info(f"Saving atlases to '{output}'")

    for layer_idx, fg_atlas in enumerate(foreground_atlases):
        name = f"foreground_atlas_{layer_idx}"
        name = f"{name}_{frame_idx:04d}" if args.custom_frame is not None else name
        name = name if args.animated else f"{name}.png"
        save_atlas(fg_atlas, str(output / name))

    name = "background_atlas"
    name = f"{name}_{frame_idx:04d}" if args.custom_frame is not None else name
    name = name if args.animated else f"{name}.png"
    save_atlas(background_atlas, str(output / name))

    logging.info("Finish! Now you can edit it and use to reconstruct the video frames.")
