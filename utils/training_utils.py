import torch
from tqdm import tqdm
import os
from .dataset import CocoDataset

def get_datasets(train_image_dir:str,
                 val_image_dir:str,
                 train_coco_json:str,
                 val_coco_json:str):

    train_ds = CocoDataset(
        image_folder=train_image_dir,
        annotations_file=train_coco_json,
        height=640,
        width=640,
    )

    val_ds = CocoDataset(
        image_folder=val_image_dir,
        annotations_file=val_coco_json,
        height=640,
        width=640,
    )

    return train_ds, val_ds

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
        self.model_save_path = f'{self.output_dir}/best_model.pth'
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            print(f"\nBest validation loss: {self.best_valid_loss}")
            print(f"\nSaving best model for epoch: {epoch+1}\n")
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, self.model_save_path)

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