# 🧠 Faster R-CNN Inference API (FastAPI) - Under development

This repository provides a RESTful API for performing object detection using a pre-trained Faster R-CNN model via FastAPI. The API supports image uploads, inference, and optional visualizations of detected objects.

---

## 🚀 Features

* FastAPI-based asynchronous server
* Supports image uploads via `/infer_image` endpoint
* Returns bounding boxes, confidence scores, and class predictions
* Optionally returns visualizations as base64-encoded images
* Loads pretrained model at startup
* Device-aware (automatically uses GPU if available)

---

## 📦 Requirements

* Python 3.8+
* Torch
* torchvision
* FastAPI
* uvicorn
* Pillow
* matplotlib

Install dependencies:

```bash
pip install torch torchvision fastapi uvicorn pillow matplotlib
```

---

## 🛠️ Setup

1. **Clone the repository** and ensure the following files exist:

   * `model_utils.py`: Defines `InferFasterRCNN` class with `load_model()` and `infer_image()`.
   * `config.py`: Defines constants like `NUM_CLASSES`, `CLASSNAMES`, `CHECKPOINT_PATH`, `IMAGE_SIZE`.

2. **Start the API server:**

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

Replace `main` with the filename containing your FastAPI code.

---

## 📤 Endpoint: `/infer_image`

### Method: `POST`

### Description:

Upload an image and get object detections with optional visualization.

### Query Parameters:

| Name      | Type  | Default | Description                                               |
| --------- | ----- | ------- | --------------------------------------------------------- |
| visualize | bool  | False   | If true, returns a base64 PNG with bounding boxes         |
| threshold | float | 0.5     | Confidence threshold for filtering detections (0.0 - 1.0) |

### Form Data:

| Name | Type         | Description                   |
| ---- | ------------ | ----------------------------- |
| file | `UploadFile` | Image file (.jpg, .png, etc.) |

### Example `curl`:

```bash
curl -X POST "http://localhost:8000/infer_image?visualize=true&threshold=0.6" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@example.jpg"
```

### Example JSON Response:

```json
{
  "boxes": [[34.0, 55.2, 123.4, 210.1]],
  "scores": [0.92],
  "classes": [1],
  "visualization": "iVBORw0KGgoAAAANSUhEUgA..."  // base64 PNG (if visualize=true)
}
```

---

## 🧩 Customization

Update the following in `config.py`:

```python
NUM_CLASSES = 3  # including background
CLASSNAMES = ["person", "car"]
CHECKPOINT_PATH = "./checkpoints/model.pth"
IMAGE_SIZE = 512
```

Ensure the `InferFasterRCNN` class implements:

```python
def load_model(self, checkpoint_path: str, device: torch.device): ...
def infer_image(self, input_dict: dict, detection_threshold: float, visualize: bool): ...
```

---

## 📬 Root Endpoint

### `/`

Returns a simple health check message:

```json
{"message": "Inference API is running."}
```

---

## 📝 Notes

* Class index 0 is assumed to be background and is not included in output.
* All bounding boxes are returned in the format `[x1, y1, x2, y2]`.

---

## 📸 Visualization

If `visualize=true`, the response includes a base64-encoded PNG image showing bounding boxes and class labels. You can render it in HTML or save it as an image:

```python
import base64
from PIL import Image
import io

b64_str = response['visualization']
img_bytes = base64.b64decode(b64_str)
img = Image.open(io.BytesIO(img_bytes))
img.show()
```

---

## 📄 License

This project is open-source and free to use under the [MIT License](LICENSE).