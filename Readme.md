# Hand Digit Image Recognition

A Flask application that predicts the digit represented by an uploaded
hand-sign image using a saved Keras model.

This project extends the original application with code-quality checks,
automated tests, pre-commit hooks, and GitHub Actions.

## Repository

- Original: https://github.com/Sachin-crypto/Flask_Image_Recognition
- Assignment fork: https://github.com/darshitaa27/Flask_Image_Recognition
- CI runs: https://github.com/darshitaa27/Flask_Image_Recognition/actions

## Requirements

- Windows with Python 3.10, 64-bit
- Git
- Local setup verified using Python 3.10.11

Runtime dependencies are listed in `requirements.txt`.
Development dependencies are listed in `requirements-dev.txt`.

Werkzeug is pinned to 2.2.3 for compatibility with Flask 2.2.1.

## Windows Setup

Run these commands in Command Prompt:

```cmd
git clone https://github.com/darshitaa27/Flask_Image_Recognition.git
cd Flask_Image_Recognition
py -3.10 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install -r requirements-dev.txt
python -m pip check
```

For an existing installation, open the project folder and activate `.venv`
before running the commands below.

## Run the Application

From the project folder:

```cmd
python -m flask --app app run
```

Open http://127.0.0.1:5000 in a browser.

1. Select a sample image from `test_images`.
2. Check the image preview.
3. Click Submit to display the predicted digit.

Press Ctrl+C in the terminal to stop the server.

Run from the project folder so the application can locate `digit_model.h5`.
TensorFlow may display CUDA-related messages when running on a CPU.

## Main Components

| Component | Purpose |
| --- | --- |
| `app.py` | Flask routes and upload handling |
| `model.py` | Image preprocessing and prediction |
| `digit_model.h5` | Saved trained Keras model |
| `templates/` | Upload and result pages |
| `static/` | Styles and image-preview JavaScript |
| `test_images/` | Sample hand-sign images |

Preprocessing converts readable images to RGB, resizes them to 224 by 224,
normalizes pixel values, and returns a batch shaped `(1, 224, 224, 3)`.

Missing or unreadable uploads display "File cannot be processed."

## Quality Checks

```cmd
python -m pylint --rcfile=.pylintrc app.py model.py
python -m pytest -q --tb=short
```

Verified results:

- Pylint: 10.00/10
- Pytest: 26 passing cases locally and in GitHub Actions
- 56 deprecation warnings remain visible in the test output

The suite retains 17 existing cases and adds nine cases:

- Three image-mode cases in `test_image_modes.py`
- Six route cases in `test_routes.py`

The valid-upload route test mocks prediction while retaining real image
preprocessing. Invalid-upload tests verify that prediction is not called.

## Pre-commit

Install the hooks after cloning:

```cmd
python -m pre_commit install
python -m pre_commit run --all-files
```

Hooks check trailing whitespace, final newlines, YAML syntax, Python syntax,
and merge-conflict markers.

If a hook modifies a file, review the change, stage it again, and retry
the commit. Pylint runs separately through its command and CI.

## Continuous Integration

`.github/workflows/ci.yml` runs on pushes to `main`, pull requests targeting
`main`, and manual dispatch.

It checks out the repository, sets up Python 3.10 on Ubuntu 22.04,
installs dependencies, checks dependency compatibility, and runs Pylint
and the complete pytest suite.

Regression tests exposed grayscale and RGBA preprocessing failures.
Converting images to RGB corrected both failures.

The tests verify selected application behaviours; they do not measure
the model's accuracy across all possible images.
