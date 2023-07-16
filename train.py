from tqdm import tqdm
from time import sleep
from utils.dataset import CocoDataset
from utils.model import create_model
import torch
import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

def get_datasets():
    
    train_ds = CocoDataset(image_folder=r'D:\Work\work\FasterRCNN-Torchvision-FineTuning\dataset\AquariumDataset\train\images',
                     annotations_file=r'D:\Work\work\FasterRCNN-Torchvision-FineTuning\dataset\AquariumDataset\train\_annotations.coco_neg.json',
                     height=640,width=640)

    val_ds = CocoDataset(image_folder=r'D:\Work\work\FasterRCNN-Torchvision-FineTuning\dataset\AquariumDataset\train\images',
                     annotations_file=r'D:\Work\work\FasterRCNN-Torchvision-FineTuning\dataset\AquariumDataset\train\_annotations.coco_neg.json',
                     height=640,width=640)
    
    return train_ds, val_ds
    

def train(train_dataset, val_dataset, epochs =2, batch_size=8, exp_folder='exp'):

    date_string = "16-07-2023-10-30-45"
    date_format = "%d-%m-%Y-%H-%M-%S"
    datetime_obj = datetime.strptime(date_string, date_format)
    date_string = datetime_obj.strftime(date_format)

    writer = SummaryWriter(os.path.join('exp', "summary",date_string))
    writer.add_hparams({'epochs':epochs, 'batch_size' : batch_size, 'experiment_folder':exp_folder},{})
    
    def custom_collate(data):
        return data
    
    # Dataloaders --
    train_dl=torch.utils.data.DataLoader(train_dataset,
                                        batch_size=batch_size,
                                        shuffle=True,
                                        collate_fn=custom_collate,
                                        pin_memory=True if torch.cuda.is_available() else False)

    val_dl=torch.utils.data.DataLoader(val_dataset,
                                        batch_size=batch_size,
                                        shuffle=False,
                                        collate_fn=custom_collate,
                                        pin_memory=True if torch.cuda.is_available() else False)
    
    # Device --
    device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # Model --
    faster_rcnn_model = create_model(train_dataset.get_total_classes_count()+1)
    faster_rcnn_model = faster_rcnn_model.to(device)
    
    # Optimizer --
    optimizer=torch.optim.SGD(faster_rcnn_model.parameters(),lr=0.001,momentum=0.9,weight_decay=0.0005)
    
    num_epochs=epochs
   
    for epoch in range(num_epochs):
        with tqdm(train_dl, unit="batch") as tepoch:
            
            epoch_loss=0
            _classifier_loss=0
            _loss_box_reg=0
            _loss_rpn_box_reg=0
            _loss_objectness=0
            
            for data in tepoch:
                tepoch.set_description(f"Epoch {epoch}/{num_epochs}")
                imgs=[]
                targets=[]
                for d in data:
                    imgs.append(d[0].to(device))
                    targ={}
                    targ['boxes']=d[1]['boxes'].to(device)
                    targ['labels']=d[1]['labels'].to(device)
                    targets.append(targ)
                loss_dict=faster_rcnn_model(imgs,targets)
                
                loss=sum(v for v in loss_dict.values())
                classifier_loss = loss_dict.get('loss_classifier').cpu().detach().numpy()
                loss_box_reg = loss_dict.get('loss_box_reg').cpu().detach().numpy()
                loss_objectness = loss_dict.get('loss_objectness').cpu().detach().numpy()
                loss_rpn_box_reg = loss_dict.get('loss_rpn_box_reg').cpu().detach().numpy()
                
                epoch_loss+=loss.cpu().detach().numpy()
                _classifier_loss+=classifier_loss
                _loss_box_reg+=loss_box_reg
                _loss_objectness+=loss_objectness
                _loss_rpn_box_reg+=loss_rpn_box_reg
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                tepoch.set_postfix(total_loss=epoch_loss, 
                                   loss_classifier=_classifier_loss,
                                   boxreg_loss = _loss_box_reg, 
                                   obj_loss = _loss_objectness,
                                   rpn_boxreg_loss=_loss_rpn_box_reg)
            
            writer.add_scalar("Train/total_loss", epoch_loss, epoch)
            writer.add_scalar("Train/classifier_loss", _classifier_loss, epoch)
            writer.add_scalar("Train/box_reg_loss", _loss_box_reg, epoch)
            writer.add_scalar("Train/objectness_loss", _loss_objectness, epoch)
            writer.add_scalar("Train/rpn_box_reg_loss", _loss_rpn_box_reg, epoch)
            
            sleep(0.1)

if __name__ == '__main__':
    train_ds, val_ds = get_datasets()
    train(train_ds,val_ds,epochs=2,batch_size=4)
