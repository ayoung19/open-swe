import base64
from io import BytesIO

import pytest
from fastapi import HTTPException
from PIL import Image

from openswe.threads.runs import _dashboard_images_from_content, _validate_command_images


@pytest.mark.parametrize(
    "url", ["data:image/png;base64,aGVsbG8=", {"url": "data:image/png;base64,aGVsbG8="}]
)
async def test_native_image_blocks_are_validated_against_the_model(
    url: str | dict[str, str],
) -> None:
    content = [{"type": "image_url", "image_url": url}]
    images = _dashboard_images_from_content(content)
    assert len(images) == 1
    assert images[0].mime_type == "image/png"
    assert images[0].base64 == "aGVsbG8="
    with pytest.raises(HTTPException) as failure:
        await _validate_command_images(
            content, model_id="fireworks:accounts/fireworks/models/kimi-k3"
        )
    assert failure.value.status_code == 422


@pytest.mark.parametrize(
    "url", ["https://example.com/image.png", "data:image/png,hello", "not an image"]
)
def test_native_image_blocks_reject_unsupported_sources(url: str) -> None:
    with pytest.raises(HTTPException) as failure:
        _dashboard_images_from_content([{"type": "image_url", "image_url": {"url": url}}])
    assert failure.value.status_code == 422


@pytest.mark.parametrize("representation", ["url", "url_dict", "native"])
async def test_dashboard_images_are_normalized_without_changing_other_content(
    representation: str,
) -> None:
    output = BytesIO()
    Image.new("RGB", (4000, 1000)).save(output, format="PNG")
    encoded = base64.b64encode(output.getvalue()).decode("ascii")
    url = f"data:image/png;base64,{encoded}"
    block = (
        {"type": "image", "base64": encoded, "mime_type": "image/png"}
        if representation == "native"
        else {
            "type": "image_url",
            "image_url": {"url": url, "detail": "high"} if representation == "url_dict" else url,
        }
    )
    text = {"type": "text", "text": "Please inspect this image"}
    content = [text, block]
    await _validate_command_images(content, model_id="anthropic:claude-opus-5-5")
    assert content[0] == text
    images = _dashboard_images_from_content(content)
    with Image.open(BytesIO(base64.b64decode(images[0].base64))) as image:
        assert image.size == (2000, 500)
        assert image.format == "PNG"


@pytest.mark.parametrize("data", [b"invalid", b"GIF89a"])
async def test_dashboard_rejects_invalid_image_content(data: bytes) -> None:
    content = [
        {
            "type": "image",
            "base64": base64.b64encode(data).decode("ascii"),
            "mime_type": "image/png",
        }
    ]
    with pytest.raises(HTTPException) as failure:
        await _validate_command_images(content, model_id="anthropic:claude-opus-5-5")
    assert failure.value.status_code == 422
