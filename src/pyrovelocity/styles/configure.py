import shutil

import matplotlib as mpl
import matplotlib.pyplot as plt
from beartype import beartype
from beartype.typing import Any, Dict

__all__ = [
    "configure_matplotlib_style",
]


@beartype
def configure_matplotlib_style(
    override_defaults: Dict[str, Any] = {},
):
    """
    Configure the global matplotlib style.

    Accepts a dictionary to override default rcParams set in the
    `common.mplstyle` file.

    For example, to override the default figure size, use:

    ```
        {
            # Example: Set a global figure size
            "figure.figsize": (8, 6),
            # Add more global settings as needed
        },
    ```

    Args:
        override_defaults (Dict[str, Any], optional):
            Dictionary to update default rcParams. Defaults to {}.
    """
    plt.style.use("pyrovelocity.styles.common")

    if not shutil.which("latex"):
        mpl.rcParams.update(
            {
                "text.usetex": False,
            }
        )

    mpl.rcParams.update(
        override_defaults,
    )
