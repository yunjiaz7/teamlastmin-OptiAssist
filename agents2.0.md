# Backend Python Coding Standards

These standards apply to all backend Python files in this project.

---

## 1. Type Hints

All functions must include type hints for parameters and return values.

```python
def process_image(image_path: str, threshold: float) -> dict:
    ...
```

---

## 2. Docstrings

All functions must have docstrings that explain their purpose, inputs, and outputs.

```python
def process_image(image_path: str, threshold: float) -> dict:
    """
    Run defect detection on a single image.

    Args:
        image_path: Absolute path to the input image file.
        threshold: Confidence threshold for filtering detections (0.0–1.0).

    Returns:
        A dict containing detected defects and their confidence scores.
    """
```

---

## 3. Async/Await for I/O Operations

Use `async`/`await` for all I/O-bound operations, including model inference and file reading.

```python
async def run_inference(image: np.ndarray) -> list:
    result = await model.predict_async(image)
    return result
```

---

## 4. Module-Level Docstrings

Every file must begin with a module-level docstring explaining its purpose.

```python
"""
detection_service.py

Handles image loading, preprocessing, and defect detection inference
for the visual inspection pipeline.
"""
```

---

## 5. Error Handling

Wrap all model inference calls in `try/except`. Raise descriptive, specific exceptions — never silently swallow errors.

```python
try:
    result = await model.infer(image)
except RuntimeError as e:
    raise RuntimeError(f"Model inference failed for image '{image_path}': {e}") from e
```

---

## 6. No Hardcoded Paths

Never hardcode file paths. Use environment variables or module-level constants.

```python
import os

MODEL_WEIGHTS_PATH = os.environ.get("MODEL_WEIGHTS_PATH", "/default/path/to/weights.pt")
```

---

## 7. Logging

Use Python's built-in `logging` module instead of `print` statements.

```python
import logging

logger = logging.getLogger(__name__)

logger.info("Starting inference for batch of %d images", len(images))
logger.error("Inference failed: %s", str(e))
```

---

## 8. Small, Single-Purpose Functions

Keep functions small and focused. Each function should do exactly one thing. If a function needs a long comment to explain what it's doing, it should probably be split.

---

## 9. Model Loading at Module Level

Load models once at module initialization, not inside functions. Loading inside a function causes repeated overhead on every call.

```python
# Load once at module level
model = load_model(MODEL_WEIGHTS_PATH)

async def run_inference(image: np.ndarray) -> list:
    # Reuse the already-loaded model
    return await model.predict_async(image)
```

---

## 10. Comments Explain WHY, Not WHAT

Comments should explain the reasoning or intent behind a decision, not restate what the code already says.

```python
# Bad: increment counter
counter += 1

# Good: skip the first frame because it is often corrupted during camera warm-up
counter += 1
```
