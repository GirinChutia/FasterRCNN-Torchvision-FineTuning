from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from typing import List, Tuple, Optional
import base64
import io
from PIL import Image
import torch
from torchvision import transforms as T
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from model_utils import InferFasterRCNN

app = FastAPI()

# Configuration
CLASSNAMES = ['fish', 'jellyfish', 'penguin', 'puffin', 'shark', 'starfish', 'stingray']
NUM_CLASSES = len(CLASSNAMES) + 1
CHECKPOINT_PATH = 'best_model.pth'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMAGE_SIZE = 640

class InferenceResult(BaseModel):
    boxes: List[List[float]]
    scores: List[float]
    classes: List[int]
    visualization: Optional[str] = None  # Base64-encoded string if visualize=True

@app.get("/")
def read_root():
    return {"message": "Inference API is running."}

@app.on_event('startup')
async def load_model():
    """
    Asynchronous event handler that loads the Faster R-CNN model at application startup.

    This function initializes an instance of `InferFasterRCNN` with the specified number of classes and class names,
    loads the model weights from the checkpoint path onto the specified device, and attaches the loaded model to the
    application state for later use in inference endpoints.

    Returns:
        None
    """
    model = InferFasterRCNN(num_classes=NUM_CLASSES, classnames=CLASSNAMES)
    model.load_model(CHECKPOINT_PATH, device=DEVICE)
    app.state.model = model


def preprocess_image(file_bytes: bytes, size: int = IMAGE_SIZE) -> Tuple[Tuple[int, int], torch.Tensor, torch.Tensor]:
    """
    Converts uploaded image bytes into original and resized PyTorch tensors.

    Args:
        file_bytes (bytes): The image file in bytes format.
        size (int, optional): The target size for resizing the image (height and width). Defaults to IMAGE_SIZE.

    Returns:
        Tuple[Tuple[int, int], torch.Tensor, torch.Tensor]:
            - original_size: A tuple (width, height) representing the original image size.
            - original: The original image as a PyTorch tensor.
            - resized: The resized image as a PyTorch tensor with shape (3, size, size).
    """
    img = Image.open(io.BytesIO(file_bytes)).convert('RGB')
    original_size = img.size  # (width, height)
    to_tensor = T.ToTensor()
    resized = T.Compose([T.Resize((size, size)), to_tensor])(img)
    original = to_tensor(img)
    return original_size, original, resized


@app.post('/infer_image', response_model=InferenceResult)
async def infer_image(
    file: UploadFile = File(...),
    visualize: bool = Query(False),
    threshold: float = Query(0.5, ge=0.0, le=1.0)
):
    """
    Endpoint to perform object detection inference on an uploaded image.
    Args:
        file (UploadFile): The image file to be processed.
        visualize (bool, optional): Whether to return a visualization of detections. Defaults to False.
        threshold (float, optional): Detection confidence threshold (between 0.0 and 1.0). Defaults to 0.5.
    Returns:
        dict: A dictionary containing:
            - boxes (List[List[float]]): List of bounding boxes [x1, y1, x2, y2] for detected objects.
            - scores (List[float]): Confidence scores for each detected object.
            - classes (List[int]): Class indices for each detected object.
            - visualization (Optional[str]): Base64-encoded PNG image with visualized detections if visualize=True, otherwise None.
    Raises:
        HTTPException: If the uploaded file is not a valid image.
    """
    # Read bytes and preprocess
    content = await file.read()
    try:
        (orig_w, orig_h), orig_tensor, resized_tensor = preprocess_image(content)
    except Exception:
        raise HTTPException(status_code=400, detail='Invalid image file')

    # Perform inference
    model = app.state.model
    result = model.infer_image(
        {
            'original_width': orig_w,
            'original_height': orig_h,
            'resized_width': resized_tensor.size(2),
            'resized_height': resized_tensor.size(1),
            'resized_image': resized_tensor,
            'original_image': orig_tensor,
        },
        detection_threshold=threshold,
        visualize=False
    )

    boxes = result.get('unscaled_boxes', [])
    scores = result.get('scores', [])
    labels = result.get('labels', [])

    vis_b64 = None

    # Visualization
    if visualize:
        img = Image.open(io.BytesIO(content)).convert('RGB')
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(img)
        for box, label, score in zip(boxes, labels, scores):
            x1, y1, x2, y2 = box
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                     linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
            ax.text(x1, y1 - 5, f"{CLASSNAMES[int(label)-1]} {score:.2f}",
                    color='white', bbox=dict(facecolor='red', alpha=0.5))
        ax.axis('off')
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        buf.seek(0)

        # Encode image as base64 string
        vis_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    # Format JSON response
    cleaned_boxes = [[float(coord) for coord in box] for box in boxes]
    cleaned_scores = [float(s) for s in scores]
    cleaned_labels = [int(l) for l in labels]

    return {
        "boxes": cleaned_boxes,
        "scores": cleaned_scores,
        "classes": cleaned_labels,
        "visualization": vis_b64  # Will be None if visualize=False
    }
