from dataclasses import asdict, dataclass

from beartype import beartype

__all__ = ["LARRY_CELL_TYPE_COLORS"]


@beartype
@dataclass(frozen=True)
class LarryCellTypeColors:
    Baso: str = "#1f77b4"  # Dark blue
    Ccr7_DC: str = "#ff7f0e"  # Orange
    Eos: str = "#2ca02c"  # Green
    Erythroid: str = "#d62728"  # Red
    Lymphoid: str = "#9467bd"  # Purple
    Mast: str = "#8c564b"  # Brown
    Meg: str = "#e377c2"  # Pink
    Monocyte: str = "#bcbd22"  # Olive
    Neutrophil: str = "#17becf"  # Teal
    Undifferentiated: str = "#aec7e8"  # Light blue
    pDC: str = "#ffbb78"  # Light orange


LARRY_CELL_TYPE_COLORS = asdict(LarryCellTypeColors())
