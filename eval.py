from utils.dataset import CocoDataset
import torch
from utils.model_utils import InferFasterRCNN,display_gt_pred
from pycocotools.coco import COCO
import os
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm
import json
import gc

def save_json(data, file_path):
    with open(file_path, 'w') as file:
        json.dump(data, file)

def evaluate_model(image_dir,
                   gt_ann_file,
                   model_weight):
    
    _ds = CocoDataset(
            image_folder=image_dir,
            annotations_file=gt_ann_file,
            height=640,
            width=640,
        )
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    IF_C = InferFasterRCNN(num_classes=_ds.get_total_classes_count() + 1,
                        classnames=_ds.get_classnames())

    IF_C.load_model(checkpoint=model_weight,
                    device=device)

    image_dir = image_dir

    cocoGt=COCO(annotation_file=gt_ann_file)
    imgIds = cocoGt.getImgIds() # all image ids

    res_id = 1
    res_all = []
        
    for id in tqdm(imgIds,total=len(imgIds)):
        id = id
        img_info = cocoGt.loadImgs(imgIds[id])[0]
        annIds = cocoGt.getAnnIds(imgIds=img_info['id'])
        ann_info = cocoGt.loadAnns(annIds)
        image_path = os.path.join(image_dir, 
                                img_info['file_name'])
        transform_info = CocoDataset.transform_image_for_inference(image_path,width=640,height=640)
        result = IF_C.infer_image(transform_info=transform_info,
                                visualize=False)

        if len(result)>0:
            pred_boxes_xyxy = result['unscaled_boxes']
            pred_boxes_xywh = [[i[0],i[1],i[2]-i[0],i[3]-i[1]] for i in pred_boxes_xyxy]
            pred_classes = result['pred_classes']
            pred_scores = result['scores']
            pred_labels = result['labels']

            for i in range(len(pred_boxes_xywh)):
                res_temp = {"id":res_id,
                            "image_id":id,
                            "bbox":pred_boxes_xywh[i],
                            "segmentation":[],
                            "iscrowd": 0,
                            "category_id": int(pred_labels[i]),
                            "area":pred_boxes_xywh[i][2]*pred_boxes_xywh[i][3],
                            "score": float(pred_scores[i])}
                res_all.append(res_temp)
                res_id+=1

    save_json_path = 'test_dect.json'
    save_json(res_all,save_json_path)
    
    cocoGt=COCO(gt_ann_file)
    cocoDt=cocoGt.loadRes(save_json_path)

    cocoEval = COCOeval(cocoGt,cocoDt,iouType='bbox')
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    
    AP_50_95 = cocoEval.stats.tolist()[0]
    AP_50 = cocoEval.stats.tolist()[1]
    
    del IF_C,_ds
    os.remove(save_json_path)
    
    torch.cuda.empty_cache()
    gc.collect()
    
    return {'AP_50_95':AP_50_95,
            'AP_50':AP_50}