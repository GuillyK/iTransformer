from typing import Sequence
import matplotlib.cm
import numpy as np
import numpy.typing as npt


def get_scalar_mappable(
    values: Sequence[float] | npt.ArrayLike,
    colors: Sequence[str] | npt.ArrayLike,
    color_values: Sequence[float] | npt.ArrayLike | None = None,
    use_log_scale: bool = False,
    linear_width: float | None = None,
) -> matplotlib.cm.ScalarMappable:
    """Get a ScalarMappable with color map: colors[0] -> ... -> colors[-1].

    Args:
        values: The values to map to colors.
            Length: N
        colors: Colors to map the values to.
            The smallest value is mapped to colors[0].
            The largest value is mapped to colors[-1].
            For possible choices of colors, see:
            https://matplotlib.org/stable/gallery/color/named_colors.html
            Length: C
        color_values: With which values the middle colors correspond. Since
            the smallest value is mapped to colors[0] and the largest value is
            mapped to colors[-1], only the middle colors can be affected by
            this parameter. The values must be in [min(values), max(values)].
            If None, the colors will be evenly distributed. Useful if you want
            to unevenly distribute the colors.
            Length: C - 2
        use_log_scale: Whether to use a log scale for assigning colors.
        linear_width: The width of the linear part around zero if log scaling
            is used. If None, the linear width will be set to a small value
            automatically. Ignored if use_log_scale is False.

    Returns:
        scalar_mappable: ScalarMappable color map for the coefficients.
        - Use scalar_mappable.cmap to get the color map.
        - Use scalar_mappable.norm to get the norm.
            The norm maps the range [vmin, vmax] to [0, 1].
            To get vmin and vmax, use scalar_mappable.norm.vmin and
                scalar_mappable.norm.vmax respectively.
            To map a value from the range [vmin, vmax] to [0, 1], use
                scalar_mappable.norm(value).

    Examples:
        >>> # The following example will create a mapping to the colors red,
        >>> # green, and blue, which will be distributed as follows:
        >>> #                       0.7                            0.3
        >>> #  |------------------------------------------|------------------|
        >>> # red                                       green              blue
        >>> # The values will be mapped to the colors as follows:
        >>> #  |------------------------------------------|------------------|
        >>> #  1              2              3               4               5
        >>> # fully         orange        yellow           mostly         fully
        >>> #  red          -ish          /green            green          blue
        >>> get_scalar_mappable(
        >>>     [1, 2, 3, 4, 5],
        >>>     ["red", "green", "blue"],
        >>>     [0.7 * (5 - 1)],
        >>> )
    """
    values = np.array(values)
    colors = np.array(colors)
    if color_values is not None:
        color_values = np.array(color_values)
        if len(color_values) != len(colors) - 2:
            raise ValueError(
                "The number of color values must be len(colors) - 2."
            )
        if (
            color_values.min() < values.min()
            or color_values.max() > values.max()
        ):
            raise ValueError(
                "The color values must be between min(values) and max(values)."
            )

    vmin = values.min()
    vmax = values.max()

    # Use a log scale if specified, otherwise use a linear scale.
    # norm is a function that maps the range [vmin, vmax] to [0, 1].
    if use_log_scale:
        if linear_width is None:
            # Make the linear width as small as possible.
            linear_width = np.abs(values).min()
        norm = matplotlib.colors.AsinhNorm(
            vmin=vmin, vmax=vmax, linear_width=linear_width  # type: ignore
        )
    else:
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

    # Create a mapping from values to colors.
    name = "_".join(colors)
    if color_values is not None:
        all_norms = np.concatenate([[0], norm(color_values).data, [1]])
    else:
        all_norms = np.linspace(0, 1, len(colors))
    color_list = list(zip(all_norms, colors))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        name=name, colors=color_list
    )

    # Create the ScalarMappable color map.
    return matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)