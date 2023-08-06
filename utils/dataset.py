from collections import namedtuple
import abc, cv2, glob,copy
import torch, os, json
from torch.utils.data import Dataset, DataLoader
import numpy as np
from collections import defaultdict
from torchvision import ops
import matplotlib.patches as patches
from torchvision import transforms as T
from PIL import Image
import matplotlib.pyplot as plt

COCOBox_base = namedtuple("COCOBox", ["xmin", "ymin", "width", "height"])
VOCBox_base = namedtuple("VOCBox", ["xmin", "ymin", "xmax", "ymax"])

class COCOBox(COCOBox_base):
    def area(self):
        return self.width * self.height


class VOCBox(VOCBox_base):
    def area(self):
        return (self.xmax - self.xmin) * (self.ymax - self.ymin)


# Define the abstract base class for loading datasets
class DatasetLoader(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def load_images(self):
        pass

    @abc.abstractmethod
    def load_annotations(self):
        pass


# the dataset class
class CocoDataset(Dataset):
    def __init__(self, image_folder, annotations_file, width, height, transforms=None):

        self.transforms = transforms
        self.image_folder = image_folder
        self.annotations_file = annotations_file
        self.height = height
        self.width = width

        if not isinstance(self.image_folder, str):
            raise ValueError("image_folder should be a string")

        if not isinstance(annotations_file, str):
            raise ValueError("annotations_file should be a string")

        self.annotations_file = annotations_file
        self.image_folder = image_folder
        self.width = width
        self.height = height

        with open(annotations_file, "r") as f:
            self.annotations = json.load(f)

        self.image_ids = defaultdict(list)
        for i in self.annotations["images"]:
            self.image_ids[i["id"]] = i  # key = image_id

        self.annotation_ids = defaultdict(list)
        for i in self.annotations["annotations"]:
            self.annotation_ids[i["image_id"]].append(i)  # key = image_id

        self.cats_id2label = {}
        self.label_names = []

        first_label_id = self.annotations["categories"][0]["id"]
        if first_label_id == 0:
            for i in self.annotations["categories"][1:]:
                self.cats_id2label[i["id"]] = i["name"]
                self.label_names.append(i["name"])
        if first_label_id == 1:
            for i in self.annotations["categories"]:
                self.cats_id2label[i["id"]] = i["name"]
                self.label_names.append(i["name"])
        if first_label_id > 1:
            raise AssertionError(
                "Something went wrong in categories, check the annotation file!"
            )

    def get_total_classes_count(self):
        return len(self.cats_id2label)

    def get_classnames(self):
        return [v for k, v in self.cats_id2label.items()]

    def load_images_annotations(self, index):
        image_info = self.image_ids[index]
        image_path = os.path.join(self.image_folder, image_info["file_name"])

        image = cv2.imread(image_path)
        rimage = cv2.cvtColor(
            image, cv2.COLOR_BGR2RGB
        )  # .astype(np.float32) # convert BGR to RGB color format
        rimage = cv2.resize(rimage, (self.width, self.height))
        # rimage /= 255.0
        rimage = Image.fromarray(rimage)

        image_height, image_width = (
            image_info["height"],
            image_info["width"],
        )  # original height & width
        anno_info = self.annotation_ids[index]

        if len(anno_info) == 0:  # for negative images (Images without annotations)
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0, 1), dtype=torch.int64)
            iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        else:
            boxes = []
            labels_id = []

            for ainfo in anno_info:
                xmin, ymin, w, h = ainfo["bbox"]
                xmax, ymax = xmin + w, ymin + h

                xmin_final = (xmin / image_width) * self.width
                xmax_final = (xmax / image_width) * self.width
                ymin_final = (ymin / image_height) * self.height
                ymax_final = (ymax / image_height) * self.height

                category_id = ainfo["category_id"]

                boxes.append([xmin_final, ymin_final, xmax_final, ymax_final])
                labels_id.append(category_id)

            boxes = torch.as_tensor(
                boxes, dtype=torch.float32
            )  # bounding box to tensor
            area = (boxes[:, 3] - boxes[:, 1]) * (
                boxes[:, 2] - boxes[:, 0]
            )  # area of the bounding boxes
            iscrowd = torch.zeros(
                (boxes.shape[0],), dtype=torch.int64
            )  # no crowd instances
            labels = torch.as_tensor(labels_id, dtype=torch.int64)  # labels to tensor

        # final `target` dictionary
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["area"] = area
        target["iscrowd"] = iscrowd
        image_id = torch.tensor([index])
        target["image_id"] = image_id

        return {
            "image": rimage,
            "height": image_height,
            "width": image_width,
            "target": target,
        }
    
    @staticmethod
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

    @staticmethod
    def display_bbox(
        bboxes, fig, ax, classes=None, in_format="xyxy", color="y", line_width=3
    ):
        if type(bboxes) == np.ndarray:
            bboxes = torch.from_numpy(bboxes)
        if classes:
            assert len(bboxes) == len(classes)
        # convert boxes to xywh format
        bboxes = ops.box_convert(bboxes, in_fmt=in_format, out_fmt="xywh")
        c = 0
        for box in bboxes:
            x, y, w, h = box.numpy()
            # display bounding box
            rect = patches.Rectangle(
                (x, y), w, h, linewidth=line_width, edgecolor=color, facecolor="none"
            )
            ax.add_patch(rect)
            # display category
            if classes:
                if classes[c] == "pad":
                    continue
                ax.text(
                    x + 5, y + 20, classes[c], bbox=dict(facecolor="yellow", alpha=0.5)
                )
            c += 1

        return fig, ax

    def __getitem__(self, idx):

        sample = self.load_images_annotations(idx)
        image_resized = sample["image"]
        target = sample["target"]

        # apply the image transforms
        if self.transforms:
            sample = self.transforms(
                image=image_resized, bboxes=target["boxes"], labels=sample["labels"]
            )
            image_resized = sample["image"]
            target["boxes"] = torch.Tensor(sample["bboxes"])

        return T.ToTensor()(image_resized), target

    def __len__(self):
        return len(self.image_ids)
