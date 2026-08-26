"""Compatibilidade de visão para o co-processador seguro V5.

Não usa ``trust_remote_code`` e não fabrica uma descrição quando o VLM falha.
Para a API atual, use diretamente ``keilinks_v5.vision.VisionService``.
"""

from __future__ import annotations

import base64
import io
import mimetypes
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from keilinks_v5.vision import VisionService


@dataclass(frozen=True)
class VisualAnalysisResult:
    descricao: str
    texto_detectado_ocr: str
    objetos_detectados: list[str]
    confianca: float
    tempo_ms: float
    vram_usada_mb: float


class VisionProjector(nn.Module):
    """Projetor experimental para treino end-to-end futuro, não ligado à API."""

    def __init__(
        self,
        vision_dim: int = 1152,
        text_dim: int = 1152,
        compression_factor: int = 2,
    ) -> None:
        super().__init__()
        if compression_factor < 1:
            raise ValueError("compression_factor deve ser positivo")
        self.compression = compression_factor
        self.mlp = nn.Sequential(
            nn.Linear(vision_dim * compression_factor**2, text_dim),
            nn.GELU(),
            nn.Linear(text_dim, text_dim),
        )

    def forward(self, visual_features: torch.Tensor) -> torch.Tensor:
        if visual_features.ndim != 3:
            raise ValueError("visual_features deve ter formato [batch, patches, dimensão]")
        batch, patches, dimension = visual_features.shape
        group = self.compression**2
        if patches % group:
            raise ValueError("número de patches não é divisível pela compressão espacial")
        return self.mlp(visual_features.reshape(batch, patches // group, dimension * group))


class VisualCortex:
    """Adaptador fino para chamadas legadas, usando VLM sequencial e seguro."""

    MODEL_BY_MODE = {
        "smolvlm": "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
        "qwen3vl": "Qwen/Qwen3-VL-2B-Instruct",
    }

    def __init__(
        self,
        mode: str = "smolvlm",
        device: str | None = None,
        load_in_4bit: bool = True,
    ) -> None:
        if device and not device.startswith("cuda"):
            raise ValueError("o perfil VisualCortex local requer CUDA")
        if not load_in_4bit:
            raise ValueError("o perfil de 8 GB requer quantização 4-bit")
        if mode not in self.MODEL_BY_MODE:
            raise ValueError(f"modo visual desconhecido: {mode}")
        self.mode = mode
        self.model_id = self.MODEL_BY_MODE[mode]
        self._service = VisionService(True, self.model_id)

    def inicializar(self) -> None:
        """Valida a configuração sem reter o VLM na VRAM."""
        if not torch.cuda.is_available():
            raise RuntimeError("visão local exige GPU CUDA")

    @staticmethod
    def _data_url(image: object) -> str:
        if isinstance(image, str) and image.startswith("data:image/"):
            return image
        if isinstance(image, (str, Path)):
            path = Path(image)
            raw = path.read_bytes()
            mime = mimetypes.guess_type(path.name)[0] or "image/png"
        elif isinstance(image, bytes):
            raw = image
            mime = "image/png"
        else:
            try:
                from PIL import Image
            except ImportError as exc:
                raise RuntimeError("Pillow é necessário para converter a imagem") from exc
            if not isinstance(image, Image.Image):
                raise ValueError(f"formato de imagem não suportado: {type(image)!r}")
            buffer = io.BytesIO()
            image.convert("RGB").save(buffer, format="PNG")
            raw = buffer.getvalue()
            mime = "image/png"
        return f"data:{mime};base64," + base64.b64encode(raw).decode("ascii")

    def analisar_imagem(
        self,
        imagem: object,
        pergunta: str = "Descreva o que está visível na imagem em português brasileiro.",
    ) -> VisualAnalysisResult:
        started = time.perf_counter()
        evidence = self._service.inspect(self._data_url(imagem), pergunta)
        return VisualAnalysisResult(
            descricao=evidence.caption or "",
            texto_detectado_ocr="",
            objetos_detectados=[],
            confianca=0.0,
            tempo_ms=round((time.perf_counter() - started) * 1000, 1),
            vram_usada_mb=evidence.vram_peak_mb or 0.0,
        )
