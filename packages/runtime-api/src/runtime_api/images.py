"""Image data shared by Robin's input paths and model-call preparation."""

import base64
import binascii
from dataclasses import dataclass
from io import BytesIO
from typing import Self

from PIL import Image, UnidentifiedImageError


@dataclass(frozen=True)
class ImageData:
    data: bytes
    mime_type: str

    @classmethod
    async def normalize(cls, data: bytes, mime_type: str) -> Self:
        """Validate image content and constrain its longest side without changing its format."""
        formats = {
            "image/png": "PNG",
            "image/jpeg": "JPEG",
            "image/gif": "GIF",
            "image/webp": "WEBP",
        }
        image_format = formats.get(mime_type)
        if image_format is None:
            raise ValueError("unsupported image type")
        try:
            with Image.open(BytesIO(data)) as image:
                if image.format != image_format:
                    raise ValueError("image content does not match its MIME type")
                image.load()
                if max(image.size) <= 2000:
                    return cls(data, mime_type)
                image.thumbnail((2000, 2000), Image.Resampling.LANCZOS)
                output = BytesIO()
                if image_format == "JPEG":
                    image.convert("RGB").save(output, format=image_format, quality=85)
                else:
                    image.save(output, format=image_format)
                return cls(output.getvalue(), mime_type)
        except (OSError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
            raise ValueError("invalid image data") from exc

    @classmethod
    async def normalize_block(cls, block: dict[str, object]) -> dict[str, object]:
        """Normalize embedded image bytes while preserving the block's other fields."""
        kind = block.get("type")
        if kind not in {"image", "image_url"}:
            return block
        source = block.get("source")
        image_url = block.get("image_url") if kind == "image_url" else block.get("url")
        url = image_url.get("url") if isinstance(image_url, dict) else image_url
        if isinstance(url, str) and url.startswith("data:"):
            header, separator, encoded = url.partition(",")
            if not separator or not header.endswith(";base64"):
                raise ValueError("invalid image data URL")
            mime_type = header[5:-7]
        elif isinstance(source, dict) and source.get("type") == "base64":
            encoded, mime_type = source.get("data"), source.get("media_type")
        elif kind == "image" and "base64" in block:
            encoded = block.get("base64")
            mime_type = block.get("mime_type") or block.get("mimeType")
        else:
            return block
        if not isinstance(encoded, str) or not isinstance(mime_type, str):
            raise ValueError("invalid image data")
        try:
            data = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("invalid image data") from exc
        image = await cls.normalize(data, mime_type)
        if image.data == data:
            return block
        normalized = base64.b64encode(image.data).decode("ascii")
        if isinstance(url, str):
            value = f"data:{mime_type};base64,{normalized}"
            key = "image_url" if kind == "image_url" else "url"
            return {
                **block,
                key: {**image_url, "url": value} if isinstance(image_url, dict) else value,
            }
        if isinstance(source, dict):
            return {**block, "source": {**source, "data": normalized}}
        return {**block, "base64": normalized}
