import torch
import cv2
import os
from .model import create_model
from .dataset import CocoDataset
import torch
import numpy as np
import copy
import matplotlib.pyplot as plt


class InferFasterRCNN:
    def __init__(self, num_classes=None, classnames=[]):

        assert type(num_classes) != type(None), "Define number of classes"

        self.num_classes = num_classes  # total_class_no + 1 (for background)

        self.classnames = ["__background__"]
        self.classnames.extend(classnames)

        self.colors = np.random.uniform(0, 255, size=(len(self.classnames), 3))

        assert (
            len(self.classnames) == self.num_classes
        ), f"num_classes: {self.num_classes}, len(classnames): {len(self.classnames)}.\
            num_classes should be equal to count of actual classes in classnames list without background + 1"

    def load_model(self, checkpoint, device="cpu"):
        self.device = device
        self.model = create_model(
            self.num_classes, checkpoint=checkpoint, device=self.device
        )
        self.model = self.model.eval()

    def infer_image(self, image, detection_threshold=0.5, visualize=False):

        orig_image = copy.deepcopy(image)
        orig_image = orig_image.cpu().numpy()
        orig_image = np.transpose(orig_image, (1, 2, 0))
        orig_image = np.ascontiguousarray(orig_image, dtype=np.float32)

        image = torch.unsqueeze(image, 0)

        with torch.no_grad():
            self.model = self.model.to(self.device)
            outputs = self.model(image.to(self.device))

        # load all detection to CPU for further operations
        outputs = [{k: v.to("cpu") for k, v in t.items()} for t in outputs]

        
        # carry further only if there are detected boxes
        if len(outputs[0]["boxes"]) != 0:
            boxes = outputs[0]["boxes"].data.numpy()
            scores = outputs[0]["scores"].data.numpy()

            # filter out boxes according to `detection_threshold`
            boxes = boxes[scores >= detection_threshold].astype(np.int32)
            draw_boxes = boxes.copy()

            # get all the predicited class names
            pred_classes = [
                self.classnames[i] for i in outputs[0]["labels"].cpu().numpy()
            ]

            # draw the bounding boxes and write the class name on top of it
            for j, box in enumerate(draw_boxes):
                class_name = pred_classes[j]
                color = self.colors[self.classnames.index(class_name)]
                cv2.rectangle(
                    orig_image,
                    (int(box[0]), int(box[1])),
                    (int(box[2]), int(box[3])),
                    color,
                    2,
                )
                cv2.putText(
                    orig_image,
                    class_name,
                    (int(box[0]), int(box[1] - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2,
                    lineType=cv2.LINE_AA,
                )

            if visualize:
                plt.figure(figsize=(10, 10))
                plt.imshow(orig_image)  # [:,:,::-1])
                plt.show()

        return outputs,orig_image

    def infer_image_path(self, image_path, detection_threshold=0.5, visualize=False):

        image = cv2.imread(image_path)
        orig_image = image.copy()

        # BGR to RGB
        image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB).astype(np.float32)
        # make the pixel range between 0 and 1
        image /= 255.0
        # bring color channels to front
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        # convert to tensor
        image = torch.tensor(image, dtype=torch.float).cpu()

        # add batch dimension
        image = torch.unsqueeze(image, 0)
        with torch.no_grad():
            self.model = self.model.to(self.device)
            outputs = self.model(image.to(self.device))

        # load all detection to CPU for further operations
        outputs = [{k: v.to("cpu") for k, v in t.items()} for t in outputs]

        
        # carry further only if there are detected boxes
        if len(outputs[0]["boxes"]) != 0:
            boxes = outputs[0]["boxes"].data.numpy()
            scores = outputs[0]["scores"].data.numpy()

            # filter out boxes according to `detection_threshold`
            boxes = boxes[scores >= detection_threshold].astype(np.int32)
            draw_boxes = boxes.copy()

            # get all the predicited class names
            pred_classes = [
                self.classnames[i] for i in outputs[0]["labels"].cpu().numpy()
            ]

            # draw the bounding boxes and write the class name on top of it
            for j, box in enumerate(draw_boxes):
                class_name = pred_classes[j]
                color = self.colors[self.classnames.index(class_name)]
                cv2.rectangle(
                    orig_image,
                    (int(box[0]), int(box[1])),
                    (int(box[2]), int(box[3])),
                    color,
                    2,
                )
                cv2.putText(
                    orig_image,
                    class_name,
                    (int(box[0]), int(box[1] - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2,
                    lineType=cv2.LINE_AA,
                )

            if visualize:
                plt.figure(figsize=(10, 10))
                plt.imshow(orig_image[:, :, ::-1])
                plt.show()

        return outputs,orig_image
    
    
    def draw_bounding_boxes(self,image, bboxes, class_labels, figsize=(12,12)):
        class_labels = class_labels.cpu().numpy()
        bboxes = bboxes.cpu().numpy()
        for j, box in enumerate(bboxes):
            label = class_labels[j]
            color = self.colors[label]
            cv2.rectangle(
                    image,
                    (int(box[0]), int(box[1])),
                    (int(box[2]), int(box[3])),
                    (0,200,0),
                    1,
                )
            cv2.putText(
                image,
                self.classnames[int(label)],
                (int(box[0] + 15), int(box[1] + 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0,200,0),
                2,
                lineType=cv2.LINE_AA,
            )
        
        plt.figure(figsize=figsize)
        plt.imshow(image)
        plt.show()


class SaveBestModel:
    """
    Class to save the best model while training. If the current epoch's 
    validation loss is less than the previous least less, then save the
    model state.
    """

    def __init__(
        self, best_valid_loss=float("inf"), output_dir="weight_outputs",
    ):
        self.best_valid_loss = best_valid_loss

        os.makedirs(output_dir, exist_ok=True)

        self.output_dir = output_dir

    def __call__(self, current_valid_loss, epoch, model, optimizer):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            print(f"\nBest validation loss: {self.best_valid_loss}")
            print(f"\nSaving best model for epoch: {epoch+1}\n")
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                f"{self.output_dir}/best_model.pth",
            )
