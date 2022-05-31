import torch


@torch.jit.script
def squared_euclidean_norm(x: torch.Tensor):
    """ Optimized Euclidean Normalization
    (https://en.wikipedia.org/wiki/Norm_(mathematics)#Euclidean_norm)

    :param x: torch.Tensor
    :return: normalized_x: torch.Tensor
    """

    return x.pow(2).sum(dim=1)


@torch.jit.script
def squared_taxicab_norm(x: torch.Tensor):
    """ Optimized Squared Taxicab Normalization
    (https://en.wikipedia.org/wiki/Norm_(mathematics)#Taxicab_norm_or_Manhattan_norm)

    :param x: torch.Tensor
    :return: normalized_x: torch.Tensor
    """

    return x.abs().sum(dim=1).pow(2)


@torch.jit.script
def taxicab_norm(x: torch.Tensor):
    """ Optimized Taxicab Normalization
    (https://en.wikipedia.org/wiki/Norm_(mathematics)#Taxicab_norm_or_Manhattan_norm)

    :param x: torch.Tensor
    :return: normalized_x: torch.Tensor
    """

    return x.abs().sum(dim=1)


@torch.jit.script
def foreground_range_norm(uv: torch.Tensor, layer_idx: int):
    """ Normalize foreground UV map for each layer

    :param uv: torch.Tensor
    :param layer_idx: int
    :return: normalized_foreground: torch.Tensor
    """

    uv = uv * 0.5 + 0.5
    uv[:, 1] = uv[:, 1] - layer_idx

    return uv


@torch.jit.script
def alpha_norm(alpha: torch.Tensor):
    """Normalize alpha to avoid alpha=0 && alpha=1

        :param alpha: torch.Tensor
        :return: normalized_alpha: torch.Tensor
    """

    alpha = alpha * 0.99
    alpha = alpha + 1e-3

    return alpha
