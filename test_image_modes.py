"""Check image preprocessing across common colour modes."""

from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from model import preprocess_img


@pytest.mark.parametrize(
    "mode, colour",
    [
        ("RGB", (128, 128, 128)),
        ("L", 128),
        ("RGBA", (128, 128, 128, 255)),
    ],
    ids=["rgb", "grayscale", "rgba"],
)
def test_preprocessing_returns_normalized_rgb_batch(mode, colour):
    """Readable images should produce three normalized RGB channels."""
    with BytesIO() as upload:
        with Image.new(mode, (32, 48), colour) as image:
            image.save(upload, format="PNG")
        upload.seek(0)

        result = preprocess_img(upload)

    assert result.shape == (1, 224, 224, 3)
    np.testing.assert_allclose(result, 128 / 255.0, rtol=1e-6)
