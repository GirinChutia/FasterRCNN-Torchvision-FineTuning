
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import warnings
warnings.filterwarnings("ignore", category=UserWarning) 

def create_model(num_classes):
    """
    Create a model for object detection using the Faster R-CNN architecture.

    Parameters:
    - num_classes (int): The number of classes for object detection. (Total classes + 1 [for background class])

    Returns:
    - model (torchvision.models.detection.fasterrcnn_resnet50_fpn): The created model for object detection.
    """
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True,
                                                                 pretrained_backbone=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes) 
    return model