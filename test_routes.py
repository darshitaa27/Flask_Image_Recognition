"""Test the upload page and prediction route."""

from io import BytesIO
from unittest.mock import Mock

import pytest
from PIL import Image

import app as app_module


@pytest.fixture
def client(monkeypatch):
    """Create a Flask client with testing enabled."""
    monkeypatch.setitem(app_module.app.config, "TESTING", True)
    with app_module.app.test_client() as test_client:
        yield test_client


def test_home_page_contains_upload_form(client):
    """The home page should provide an image upload form."""
    response = client.get("/")

    assert response.status_code == 200
    assert response.mimetype == "text/html"
    assert b"<form" in response.data
    assert b'name="file"' in response.data


def test_valid_upload_displays_prediction(client, monkeypatch):
    """A valid image should reach the predictor and result template."""
    predictor = Mock(return_value=5)
    renderer = Mock(wraps=app_module.render_template)
    monkeypatch.setattr(app_module, "predict_result", predictor)
    monkeypatch.setattr(app_module, "render_template", renderer)

    with BytesIO() as upload:
        with Image.new("RGB", (32, 48), (128, 128, 128)) as image:
            image.save(upload, format="PNG")
        upload.seek(0)
        response = client.post(
            "/prediction",
            data={"file": (upload, "sample.png")},
            content_type="multipart/form-data",
        )

    assert response.status_code == 200
    predictor.assert_called_once()
    batch = predictor.call_args.args[0]
    assert batch.shape == (1, 224, 224, 3)
    renderer.assert_called_once_with("result.html", predictions="5")
    assert b"5" in response.data


@pytest.mark.parametrize(
    "contents, filename",
    [
        (b"", "empty.png"),
        (b"\x89PNG\r\n\x1a\nbroken", "corrupted.png"),
        (b"This is a text document.", "document.txt"),
    ],
    ids=["empty-file", "corrupted-image", "text-file"],
)
def test_invalid_upload_does_not_call_predictor(
    client, monkeypatch, contents, filename
):
    """Unreadable uploads should show an error without prediction."""
    predictor = Mock()
    monkeypatch.setattr(app_module, "predict_result", predictor)

    response = client.post(
        "/prediction",
        data={"file": (BytesIO(contents), filename)},
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    assert b"File cannot be processed." in response.data
    predictor.assert_not_called()


def test_empty_filename_does_not_call_predictor(client, monkeypatch):
    """An upload without a filename should show an error."""
    predictor = Mock()
    monkeypatch.setattr(app_module, "predict_result", predictor)

    response = client.post(
        "/prediction",
        data={"file": (BytesIO(b""), "")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    assert b"File cannot be processed." in response.data
    predictor.assert_not_called()
