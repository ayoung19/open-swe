import base64
import socket
from io import BytesIO
from typing import Any, cast
from urllib.parse import urlparse

import httpx2
import pytest
from PIL import Image

import openswe.utils.multimodal as multimodal
import openswe.utils.url_safety as url_safety
from openswe.utils.multimodal import (
    extract_image_urls,
    fetch_image_block,
)

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAIAAAABCAIAAAB7QOjdAAAAD0lEQVR4nGP8z8DAwMAAAAYIAQHLR3Z1AAAAAElFTkSuQmCC"
)


def test_extract_image_urls_markdown_and_direct_dedupes() -> None:
    text = (
        "Here is an image ![alt](https://example.com/a.png) and another "
        "![https://example.com/b.JPG?size=large plus a repeat https://example.com/a.png"
    )

    assert extract_image_urls(text) == [
        "https://example.com/a.png",
        "https://example.com/b.JPG?size=large",
    ]


def test_extract_image_urls_ignores_non_images() -> None:
    text = "Not images: https://example.com/file.pdf and https://example.com/noext"

    assert extract_image_urls(text) == []


def _addr_info(ip: str, port: int | None = None) -> tuple:
    return (
        socket.AF_INET,
        socket.SOCK_STREAM,
        6,
        "",
        (ip, port or 0),
    )


class FakeImageResponse:
    def __init__(
        self,
        *,
        status_code: int,
        url: str,
        headers: dict[str, str] | None = None,
        content: bytes = b"",
    ) -> None:
        self.status_code = status_code
        self.url = url
        self.headers = headers or {}
        self.content = content

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx2.HTTPStatusError(
                f"{self.status_code} error",
                request=httpx2.Request("GET", self.url),
                response=httpx2.Response(self.status_code, request=httpx2.Request("GET", self.url)),
            )


class FakeImageClient:
    def __init__(self, responder: Any) -> None:
        self._responder = responder
        self.calls: list[dict[str, Any]] = []

    async def request(self, method: str, url: str, **kwargs: Any) -> FakeImageResponse:
        self.calls.append({"method": method, "url": url, **kwargs})
        return self._responder(method, url, **kwargs)


def _patch_image_dns(monkeypatch: Any) -> None:
    def fake_getaddrinfo(host: str, port: int | None, *args: Any, **kwargs: Any) -> list[tuple]:
        public_hosts = {
            "cdn.example.com",
            "example.com",
            "files.slack.com",
            "private.files.slack.com",
        }
        ip = "93.184.216.34" if host in public_hosts else host
        return [_addr_info(ip, port)]

    monkeypatch.setattr(url_safety.socket, "getaddrinfo", fake_getaddrinfo)


async def test_fetch_image_block_blocks_redirect_to_internal_url(monkeypatch: Any) -> None:
    _patch_image_dns(monkeypatch)

    def responder(method: str, url: str, **kwargs: Any) -> FakeImageResponse:
        return FakeImageResponse(
            status_code=302,
            url=url,
            headers={"Location": "http://169.254.169.254/latest/meta-data"},
        )

    client = FakeImageClient(responder)

    result = await fetch_image_block(
        "https://example.com/start.png", cast(httpx2.AsyncClient, client)
    )

    assert result is None
    assert len(client.calls) == 1
    assert urlparse(client.calls[0]["url"]).hostname == "93.184.216.34"
    assert client.calls[0]["headers"]["Host"] == "example.com"


async def test_fetch_image_block_tries_each_validated_address(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        url_safety.socket,
        "getaddrinfo",
        lambda host, port, *args, **kwargs: [
            _addr_info("93.184.216.34", port),
            _addr_info("93.184.216.35", port),
        ],
    )
    monkeypatch.setattr(multimodal, "create_image_block", lambda **kwargs: kwargs)

    def responder(method: str, url: str, **kwargs: Any) -> FakeImageResponse:
        if urlparse(url).hostname == "93.184.216.34":
            raise httpx2.ConnectError("address unreachable")
        return FakeImageResponse(
            status_code=200,
            url=url,
            headers={"Content-Type": "image/png"},
            content=PNG,
        )

    client = FakeImageClient(responder)

    result = await fetch_image_block(
        "https://example.com/image.png", cast(httpx2.AsyncClient, client)
    )

    assert result == {"base64": base64.b64encode(PNG).decode("ascii"), "mime_type": "image/png"}
    assert [urlparse(call["url"]).hostname for call in client.calls] == [
        "93.184.216.34",
        "93.184.216.35",
    ]


async def test_fetch_image_block_accepts_image_at_size_limit(monkeypatch: Any) -> None:
    _patch_image_dns(monkeypatch)
    monkeypatch.setattr(multimodal, "_MAX_IMAGE_BYTES", len(PNG))
    monkeypatch.setattr(multimodal, "create_image_block", lambda **kwargs: kwargs)

    def responder(method: str, url: str, **kwargs: Any) -> FakeImageResponse:
        return FakeImageResponse(
            status_code=200,
            url=url,
            headers={"Content-Type": "image/png"},
            content=PNG,
        )

    result = await fetch_image_block(
        "https://example.com/image.png", cast(httpx2.AsyncClient, FakeImageClient(responder))
    )

    assert result == {"base64": base64.b64encode(PNG).decode("ascii"), "mime_type": "image/png"}


async def test_fetch_image_block_warns_about_image_above_size_limit(monkeypatch: Any) -> None:
    _patch_image_dns(monkeypatch)
    monkeypatch.setattr(multimodal, "_MAX_IMAGE_BYTES", 3)
    monkeypatch.setattr(multimodal, "create_image_block", lambda **kwargs: kwargs)

    def responder(method: str, url: str, **kwargs: Any) -> FakeImageResponse:
        return FakeImageResponse(
            status_code=200,
            url=url,
            headers={"Content-Type": "image/png"},
            content=b"large",
        )

    result = await fetch_image_block(
        "https://example.com/image.png", cast(httpx2.AsyncClient, FakeImageClient(responder))
    )

    assert result is not None
    assert result["type"] == "text"
    assert result["text"] == (
        "An attached image was skipped because it exceeded the 10 MiB size limit."
    )


async def test_fetch_image_block_does_not_forward_slack_auth_to_redirect_host(
    monkeypatch: Any,
) -> None:
    _patch_image_dns(monkeypatch)
    monkeypatch.setenv("SLACK_BOT_TOKEN", "test-slack-token")
    monkeypatch.setattr(multimodal, "create_image_block", lambda **kwargs: kwargs)

    def responder(method: str, url: str, **kwargs: Any) -> FakeImageResponse:
        host = kwargs["headers"]["Host"]
        if host == "files.slack.com":
            return FakeImageResponse(
                status_code=302,
                url=url,
                headers={"Location": "https://cdn.example.com/image.png"},
            )
        return FakeImageResponse(
            status_code=200,
            url=url,
            headers={"Content-Type": "image/png"},
            content=PNG,
        )

    client = FakeImageClient(responder)

    result = await fetch_image_block(
        "https://files.slack.com/image.png", cast(httpx2.AsyncClient, client)
    )

    assert result == {"base64": base64.b64encode(PNG).decode("ascii"), "mime_type": "image/png"}
    assert len(client.calls) == 2
    assert client.calls[0]["headers"]["Authorization"] == "Bearer test-slack-token"
    assert "Authorization" not in client.calls[1]["headers"]
    assert client.calls[1]["headers"]["Host"] == "cdn.example.com"


async def test_fetch_image_block_does_not_add_slack_auth_after_untrusted_redirect(
    monkeypatch: Any,
) -> None:
    _patch_image_dns(monkeypatch)
    monkeypatch.setenv("SLACK_BOT_TOKEN", "test-slack-token")
    monkeypatch.setattr(multimodal, "create_image_block", lambda **kwargs: kwargs)

    def responder(method: str, url: str, **kwargs: Any) -> FakeImageResponse:
        if kwargs["headers"]["Host"] == "example.com":
            return FakeImageResponse(
                status_code=302,
                url=url,
                headers={"Location": "https://files.slack.com/private.png"},
            )
        return FakeImageResponse(
            status_code=200,
            url=url,
            headers={"Content-Type": "image/png"},
            content=PNG,
        )

    client = FakeImageClient(responder)

    result = await fetch_image_block(
        "https://example.com/start.png", cast(httpx2.AsyncClient, client)
    )

    assert result == {"base64": base64.b64encode(PNG).decode("ascii"), "mime_type": "image/png"}
    assert len(client.calls) == 2
    assert all("Authorization" not in call["headers"] for call in client.calls)


async def test_fetch_image_block_keeps_auth_within_slack_host_family(monkeypatch: Any) -> None:
    _patch_image_dns(monkeypatch)
    monkeypatch.setenv("SLACK_BOT_TOKEN", "test-slack-token")
    monkeypatch.setattr(multimodal, "create_image_block", lambda **kwargs: kwargs)

    def responder(method: str, url: str, **kwargs: Any) -> FakeImageResponse:
        if kwargs["headers"]["Host"] == "files.slack.com":
            return FakeImageResponse(
                status_code=302,
                url=url,
                headers={"Location": "https://private.files.slack.com/image.png"},
            )
        return FakeImageResponse(
            status_code=200,
            url=url,
            headers={"Content-Type": "image/png"},
            content=PNG,
        )

    client = FakeImageClient(responder)

    result = await fetch_image_block(
        "https://files.slack.com/image.png", cast(httpx2.AsyncClient, client)
    )

    assert result == {"base64": base64.b64encode(PNG).decode("ascii"), "mime_type": "image/png"}
    assert len(client.calls) == 2
    assert all(
        call["headers"]["Authorization"] == "Bearer test-slack-token" for call in client.calls
    )


@pytest.mark.parametrize("image_format", ["PNG", "JPEG", "GIF", "WEBP"])
@pytest.mark.parametrize("size", [(4000, 1000), (1000, 4000), (2000, 500)])
async def test_fetched_images_keep_format_and_fit_model_dimensions(
    monkeypatch: pytest.MonkeyPatch, image_format: str, size: tuple[int, int]
) -> None:
    _patch_image_dns(monkeypatch)
    output = BytesIO()
    Image.new("RGB", size, "red").save(output, format=image_format)
    data = output.getvalue()
    mime = f"image/{image_format.lower()}"
    response = FakeImageResponse(
        status_code=200,
        url="https://example.com/image",
        headers={"Content-Type": mime},
        content=data,
    )
    client = FakeImageClient(lambda *args, **kwargs: response)
    block = await fetch_image_block("https://example.com/image", cast(httpx2.AsyncClient, client))
    assert block is not None and block["type"] == "image"
    assert block["mime_type"] == mime
    decoded = base64.b64decode(block["base64"])
    with Image.open(BytesIO(decoded)) as image:
        assert image.format == image_format
        assert image.size == (size if max(size) <= 2000 else (size[0] // 2, size[1] // 2))
    if max(size) <= 2000:
        assert decoded == data


@pytest.mark.parametrize(
    ("data", "mime"), [(b"not an image", "image/png"), (PNG[:40], "image/png"), (PNG, "image/jpeg")]
)
async def test_fetched_invalid_images_are_not_forwarded(
    monkeypatch: pytest.MonkeyPatch, data: bytes, mime: str
) -> None:
    _patch_image_dns(monkeypatch)
    response = FakeImageResponse(
        status_code=200,
        url="https://example.com/image.png",
        headers={"Content-Type": mime},
        content=data,
    )
    block = await fetch_image_block(
        "https://example.com/image.png",
        cast(httpx2.AsyncClient, FakeImageClient(lambda *args, **kwargs: response)),
    )
    assert block is None
