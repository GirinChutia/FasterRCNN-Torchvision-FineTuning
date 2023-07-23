from tqdm import tqdm
from time import sleep
from utils.dataset import CocoDataset
from utils.model import create_model
import torch
import os
import time
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

class SaveBestModel:
    """
    Class to save the best model while training. If the current epoch's 
    validation loss is less than the previous least less, then save the
    model state.
    """
    def __init__(
        self, best_valid_loss=float('inf'), output_dir = 'weight_outputs',
    ):
        self.best_valid_loss = best_valid_loss
    
        os.makedirs(output_dir,exist_ok=True)
        
        self.output_dir = output_dir
        
    def __call__(
        self, current_valid_loss, 
        epoch, model, optimizer
    ):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            print(f"\nBest validation loss: {self.best_valid_loss}")
            print(f"\nSaving best model for epoch: {epoch+1}\n")
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, f'{self.output_dir}/best_model.pth')


def get_datasets():

    train_ds = CocoDataset(
        image_folder=r"D:\Work\work\FasterRCNN-Torchvision-FineTuning\dataset\AquariumDataset\train\images",
        annotations_file=r"D:\Work\work\FasterRCNN-Torchvision-FineTuning\dataset\AquariumDataset\train\_annotations.coco_neg.json",
        height=640,
        width=640,
    )

    val_ds = CocoDataset(
        image_folder=r"D:\Work\work\FasterRCNN-Torchvision-FineTuning\dataset\AquariumDataset\train\images",
        annotations_file=r"D:\Work\work\FasterRCNN-Torchvision-FineTuning\dataset\AquariumDataset\train\_annotations.coco_neg.json",
        height=640,
        width=640,
    )

    return train_ds, val_ds


def train_one_epoch(model, train_dl, optimizer, writer, epoch_no, total_epoch, device):
    with tqdm(train_dl, unit="batch") as tepoch:
        epoch_loss = 0
        _classifier_loss = 0
        _loss_box_reg = 0
        _loss_rpn_box_reg = 0
        _loss_objectness = 0
        for data in tepoch:
            tepoch.set_description(f"Train:Epoch {epoch_no}/{total_epoch}")
            imgs = []
            targets = []
            for d in data:
                imgs.append(d[0].to(device))
                targ = {}
                targ["boxes"] = d[1]["boxes"].to(device)
                targ["labels"] = d[1]["labels"].to(device)
                targets.append(targ)
            loss_dict = model(imgs, targets)

            loss = sum(v for v in loss_dict.values())
            classifier_loss = loss_dict.get("loss_classifier").cpu().detach().numpy()
            loss_box_reg = loss_dict.get("loss_box_reg").cpu().detach().numpy()
            loss_objectness = loss_dict.get("loss_objectness").cpu().detach().numpy()
            loss_rpn_box_reg = loss_dict.get("loss_rpn_box_reg").cpu().detach().numpy()

            epoch_loss += loss.cpu().detach().numpy()
            _classifier_loss += classifier_loss
            _loss_box_reg += loss_box_reg
            _loss_objectness += loss_objectness
            _loss_rpn_box_reg += loss_rpn_box_reg

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tepoch.set_postfix(
                total_loss=epoch_loss,
                loss_classifier=_classifier_loss,
                boxreg_loss=_loss_box_reg,
                obj_loss=_loss_objectness,
                rpn_boxreg_loss=_loss_rpn_box_reg,
            )

        writer.add_scalar("Train/total_loss", epoch_loss, epoch_no)
        writer.add_scalar("Train/classifier_loss", _classifier_loss, epoch_no)
        writer.add_scalar("Train/box_reg_loss", _loss_box_reg, epoch_no)
        writer.add_scalar("Train/objectness_loss", _loss_objectness, epoch_no)
        writer.add_scalar("Train/rpn_box_reg_loss", _loss_rpn_box_reg, epoch_no)

    return model, optimizer, writer, epoch_loss


@torch.inference_mode()
def val_one_epoch(model, val_dl, writer, epoch_no, total_epoch, device, log=True):
    with tqdm(val_dl, unit="batch") as tepoch:
        epoch_loss = 0
        _classifier_loss = 0
        _loss_box_reg = 0
        _loss_rpn_box_reg = 0
        _loss_objectness = 0
        for data in tepoch:
            tepoch.set_description(f"Val:Epoch {epoch_no}/{total_epoch}")
            imgs = []
            targets = []
            for d in data:
                imgs.append(d[0].to(device))
                targ = {}
                targ["boxes"] = d[1]["boxes"].to(device)
                targ["labels"] = d[1]["labels"].to(device)
                targets.append(targ)
            loss_dict = model(imgs, targets)

            loss = sum(v for v in loss_dict.values())
            classifier_loss = loss_dict.get("loss_classifier").cpu().detach().numpy()
            loss_box_reg = loss_dict.get("loss_box_reg").cpu().detach().numpy()
            loss_objectness = loss_dict.get("loss_objectness").cpu().detach().numpy()
            loss_rpn_box_reg = loss_dict.get("loss_rpn_box_reg").cpu().detach().numpy()

            epoch_loss += loss.cpu().detach().numpy()
            _classifier_loss += classifier_loss
            _loss_box_reg += loss_box_reg
            _loss_objectness += loss_objectness
            _loss_rpn_box_reg += loss_rpn_box_reg

            tepoch.set_postfix(
                total_loss=epoch_loss,
                loss_classifier=_classifier_loss,
                boxreg_loss=_loss_box_reg,
                obj_loss=_loss_objectness,
                rpn_boxreg_loss=_loss_rpn_box_reg,
            )

        if log:
            writer.add_scalar("Val/total_loss", epoch_loss, epoch_no)
            writer.add_scalar("Val/classifier_loss", _classifier_loss, epoch_no)
            writer.add_scalar("Val/box_reg_loss", _loss_box_reg, epoch_no)
            writer.add_scalar("Val/objectness_loss", _loss_objectness, epoch_no)
            writer.add_scalar("Val/rpn_box_reg_loss", _loss_rpn_box_reg, epoch_no)

    return writer, epoch_loss


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

    writer = SummaryWriter(os.path.join("exp", "summary", date_string))

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
    optimizer = torch.optim.SGD(
        faster_rcnn_model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005
    )

    num_epochs = epochs
    save_best_model = SaveBestModel()

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

        if epoch % val_eval_freq == 0:  # Do evaluation of validation set
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
            
            sleep(0.1)

    _, _ = val_one_epoch(
        faster_rcnn_model, val_dl, writer, epoch + 1, num_epochs, device, log=False
    )

    writer.add_hparams(
        {"epochs": epochs, "batch_size": batch_size},
        {"Train/total_loss": epoch_loss, "Val/total_loss": val_epoch_loss},
    )


if __name__ == "__main__":
    train_ds, val_ds = get_datasets()
    train(train_ds, val_ds, epochs=15, batch_size=6)
