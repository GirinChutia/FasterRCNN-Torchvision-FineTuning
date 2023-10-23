from tqdm import tqdm
from time import sleep
from utils.dataset import CocoDataset
from utils.model import create_model
from utils.training_utils import SaveBestModel,train_one_epoch,val_one_epoch,get_datasets
import torch
import os
import time
from datetime import datetime
from dataclasses import dataclass
from simple_parsing import ArgumentParser
from eval import evaluate_model
from torch import nn
from torch.utils.tensorboard import SummaryWriter

def train(
    train_dataset,
    val_dataset,
    epochs=2,
    batch_size=8,
    exp_folder="exp",
    val_eval_freq=1,
):

    date_format = "%d-%m-%Y-%H-%M-%S"
    date_string = time.strftime(date_format)

    exp_folder = os.path.join("exp", "summary", date_string)
    writer = SummaryWriter(exp_folder)

    def custom_collate(data):
        return data

    # Dataloaders --
    train_dl = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=custom_collate,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    val_dl = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=custom_collate,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    # Device --
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Model --
    faster_rcnn_model = create_model(train_dataset.get_total_classes_count() + 1)
    faster_rcnn_model = faster_rcnn_model.to(device)

    # Optimizer --
    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups

    for k, v in faster_rcnn_model.named_modules():
        if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
            pg2.append(v.bias)  # biases
        if isinstance(v, nn.BatchNorm2d) or "bn" in k:
            pg0.append(v.weight)  # no decay
        elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
            pg1.append(v.weight)  # apply decay

    optimizer = torch.optim.SGD(
        pg0, lr=0.001, momentum=0.9, nesterov=True
    ) # BN
    
    optimizer.add_param_group(
        {"params": pg1, "weight_decay":  5e-4}
    )  # add pg1 with weight_decay # Weights
    
    optimizer.add_param_group({"params": pg2}) # Biases
    

    num_epochs = epochs
    save_best_model = SaveBestModel(output_dir=exp_folder)

    for epoch in range(num_epochs):

        faster_rcnn_model, optimizer, writer, epoch_loss = train_one_epoch(
            faster_rcnn_model,
            train_dl,
            optimizer,
            writer,
            epoch + 1,
            num_epochs,
            device,
        )
        
        sleep(0.1)

        if (epoch % val_eval_freq == 0) and epoch != 0:  # Do evaluation of validation set
            eval_result = evaluate_model(image_dir=val_dataset.image_folder,
                                         gt_ann_file=val_dataset.annotations_file,
                                         model_weight=save_best_model.model_save_path)
            
            sleep(0.1)
            
            writer.add_scalar("Val/AP_50_95", eval_result['AP_50_95'], epoch + 1)
            writer.add_scalar("Val/AP_50", eval_result['AP_50'], epoch + 1)
        
        else:
            writer, val_epoch_loss = val_one_epoch(
                faster_rcnn_model,
                val_dl,
                writer,
                epoch + 1,
                num_epochs,
                device,
                log=True,
            )
        
            sleep(0.1)
            
            save_best_model(val_epoch_loss, 
                            epoch, 
                            faster_rcnn_model, 
                            optimizer)
            

    _, _ = val_one_epoch(
        faster_rcnn_model, val_dl, writer, epoch + 1, num_epochs, device, log=False
    )

    writer.add_hparams(
        {"epochs": epochs, "batch_size": batch_size},
        {"Train/total_loss": epoch_loss, "Val/total_loss": val_epoch_loss},
    )

@dataclass
class DatasetPaths:
    train_image_dir: str = r"D:\Work\work\FasterRCNN-Torchvision-FineTuning\dataset\AquariumDataset\train\images"
    val_image_dir: str = r"D:\Work\work\FasterRCNN-Torchvision-FineTuning\dataset\AquariumDataset\valid\images"
    train_coco_json: str = r"D:\Work\work\FasterRCNN-Torchvision-FineTuning\dataset\AquariumDataset\train\_annotations.coco_neg.json"
    val_coco_json: str = r"D:\Work\work\FasterRCNN-Torchvision-FineTuning\dataset\AquariumDataset\valid\_annotations.coco.json"

@dataclass
class TrainingConfig:
    epochs: int = 15
    batch_size: int = 6
    val_eval_freq: int = 2
    exp_folder: str = 'exp'

if __name__ == "__main__":
    
    parser = ArgumentParser()
    parser.add_arguments(DatasetPaths,dest='dataset_config')
    parser.add_arguments(TrainingConfig,dest='training_config')
    args = parser.parse_args()
    
    dataset_config: DatasetPaths = args.dataset_config
    training_config: TrainingConfig = args.training_config
    
    train_ds, val_ds = get_datasets(train_image_dir=dataset_config.train_image_dir,
                                    train_coco_json=dataset_config.train_coco_json,
                                    val_image_dir=dataset_config.val_image_dir,
                                    val_coco_json=dataset_config.val_coco_json)
    train(train_ds, val_ds,
          epochs=training_config.epochs, 
          batch_size=training_config.batch_size,
          val_eval_freq=training_config.val_eval_freq,
          exp_folder=training_config.exp_folder)
    
