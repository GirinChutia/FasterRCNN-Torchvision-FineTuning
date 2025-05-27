from model_utils import InferFasterRCNN
import torch
import bentoml
import cv2
from PIL import Image
import copy
from torchvision import transforms as T
from pathlib import Path

def transform_image_for_inference(image_path,width,height):
        
    image = cv2.imread(image_path)
    ori_h, ori_w, _ = image.shape
    
    oimage = copy.deepcopy(image)
    oimage = Image.fromarray(oimage)
    oimage = T.ToTensor()(oimage)
    
    rimage = cv2.cvtColor(
        image, cv2.COLOR_BGR2RGB
    )
    rimage = cv2.resize(rimage, (width,height))
    rimage = Image.fromarray(rimage)
    rimage = T.ToTensor()(rimage)
    # rimage = torch.unsqueeze(rimage, 0)
    
    transform_info = {'original_width':ori_w,
                        'original_height':ori_h,
                        'resized_width':width,
                        'resized_height':height,
                        'resized_image':rimage,
                        'original_image':oimage}
    
    return transform_info # this can directly go to model for inference
    
@bentoml.service(
    resources={"gpu": 1, "memory": "4GiB"},
    traffic={"timeout": 20},
)
class InferFasterRCNNService:
    
    def __init__(self) -> None:
        
        classnames = ['fish', 'jellyfish', 'penguin', 'puffin', 'shark', 'starfish', 'stingray']
        num_classes = len(classnames)+1
        checkpoint = r'D:\Work\Build\FasterRCNN-Torchvision-FineTuning\weight_outputs_best\best_model.pth'
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.num_classes = num_classes
        self.classnames = classnames
        self.checkpoint = checkpoint
        self.device = device
        self.model = InferFasterRCNN(num_classes=num_classes, classnames=classnames)
        self.model.load_model(checkpoint, device=device)
        
    @bentoml.api
    def infer_image(self, image: Path):
        transform_info = transform_image_for_inference(image, width=640, height=640)
        
        result = self.model.infer_image(transform_info,
                                      detection_threshold=0.5,
                                      visualize=False)
        
        if len(result) == 0:
            all_results = {'boxes':[], # XYXY
                            'scores':[],
                            'classes':[]}
        else:
            pred_boxes = result['unscaled_boxes']
            pred_classes = result['pred_classes']
            pred_scores = result['scores']
            pred_labels = result['labels']
            
            pred_boxes = [[float(i[0]),float(i[1]),
                        float(i[2]),float(i[3])] for i in pred_boxes] # xyxy
            pred_scores = [float(i) for i in pred_scores]
            pred_labels = [int(i) for i in pred_labels]
            
            all_results = {
                'boxes':pred_boxes, # XYXY
                'scores':pred_scores,
                'classes':pred_labels}
    
        return all_results
        