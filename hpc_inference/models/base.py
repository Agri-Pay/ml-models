"""Minimal base model interface for HPC inference.

Contains only data types and the abstract model interface needed for
standalone inference. No server dependencies (no FastAPI, no Supabase).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class InputType(str, Enum):
    GEOTIFF = "geotiff"
    IMAGE_RGB = "image_rgb"
    TABULAR = "tabular"


class OutputType(str, Enum):
    PLANT_COUNT = "plant_count"
    DENSITY_MAP = "density_map"
    SIZE_ESTIMATION = "size_estimation"
    CLASSIFICATION = "classification"
    SCALAR = "scalar"


@dataclass
class ModelSpec:
    model_id: str
    name: str
    crops: List[str]
    task: str
    input_type: InputType
    output_types: List[OutputType]
    weight_files: List[str]
    description: str = ""
    version: str = "1.0.0"
    input_size: tuple = (320, 320)
    extra: Dict[str, Any] = field(default_factory=dict)


ProgressCallback = Callable[[int, int, str], None]


class BaseModel(ABC):
    _loaded: bool = False
    _device: str = "cpu"

    @abstractmethod
    def spec(self) -> ModelSpec: ...

    @abstractmethod
    def load(self, device: str = "cpu") -> None: ...

    @abstractmethod
    def predict(
        self,
        input_data: Any,
        progress_cb: Optional[ProgressCallback] = None,
    ) -> Dict[str, Any]: ...

    def is_loaded(self) -> bool:
        return self._loaded

    def unload(self) -> None:
        self._loaded = False

    def get_device(self) -> str:
        return self._device
