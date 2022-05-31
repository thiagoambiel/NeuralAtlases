from dataclasses import dataclass


@dataclass
class Config:
    max_frames: int = 100
    size: tuple = (768, 432)
    foreground_class: str = "anything"
    foreground_layers: int = 1

    log_every: int = 0
    save_every: int = 1_000
    evaluate_every: int = 10_000

    batch_size: int = 10_000
    sparsity_batch_size: int = 8_000

    optical_flow_coeff: float = 5.0             # βf (Optical Flow)
    alpha_flow_coeff: float = 50.0              # βf-α (Alpha Optical Flow)

    rgb_coeff: float = 5_000.0                  # βr (RGB)
    rigidity_coeff: float = 5.0                 # βr (Rigidity)
    sparsity_coeff: float = 1_000.0             # βs (Sparsity)
    gradient_coeff: float = 1_000.0             # βg (Gradient)
    scribbles_coeff: float = 10_000.0

    global_rigidity_coeff_fg: float = 5.0       # Not in Paper (But in official code)
    global_rigidity_coeff_bg: float = 50.0      # Not in Paper (But in official code)
    alpha_bootstrapping_coeff: float = 2_000.0  # Not in Paper (But in official code)

    derivative_amount: int = 1
    global_derivative_amount: int = 100

    uv_mapping_scale: float = 0.8

    number_of_channels_atlas: int = 256
    number_of_layers_atlas: int = 8
    positional_encoding_num_atlas: int = 10

    number_of_channels_alpha: int = 256
    number_of_layers_alpha: int = 8
    positional_encoding_num_alpha: int = 5

    number_of_channels_foreground: int = 256
    number_of_layers_foreground: int = 6

    number_of_channels_background: int = 256
    number_of_layers_background: int = 4
