import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision
from PIL import Image
import pandas as pd
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from config import Config
from net import GeoModel
from data import GeoDataset

if __name__=="__main__":
    df = pd.read_csv(Config.INPUT_PATH)

    train_data = GeoDataset(df[df["fold"]!=Config.FOLD])
    eval_data = GeoDataset(df[df["fold"]==Config.FOLD],eval=True)

    train_dataloader = DataLoader(train_data,\
                        batch_size=Config.BATCH_SIZE,\
                        shuffle=True)
    eval_dataloader = DataLoader(eval_data,\
                        batch_size=Config.BATCH_SIZE,\
                        shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=Config.LR,weight_decay=0.0001)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',\
        factor=0.5, patience=5, threshold=0.0001, threshold_mode='rel',\
             cooldown=0, min_lr=0, eps=1e-08, verbose=True)

    best_loss = 1000
    best_accuracy = 0
    writer = SummaryWriter(log_dir="runs")
    for epoch in range(Config.EPOCHS):
        # TRAIN ONE EPOC
        model.train()
        train_loss = 0
        for i, (X,y) in tqdm(enumerate(train_dataloader),total=len(train_dataloader)):
            X = X.to(DEVICE)
            y = y.to(DEVICE)

            y_pred = model(X)
            loss = criterion(y_pred,y)
            train_loss+=loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_loss = train_loss/len(train_dataloader)
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)
        # EVAL ONE EPOCHS
        model.eval()
        val_loss = 0
        correct = 0
        size = len(eval_dataloader.dataset)
        predictions = np.empty((0,58))
        actuals = []
        with torch.no_grad():
            for i,(X,y) in tqdm(enumerate(eval_dataloader),total=len(eval_dataloader)):
                X = X.to(DEVICE)
                y = y.to(DEVICE)

                y_pred = model(X)
                loss = criterion(y_pred,y)
                val_loss+=loss.item()
                correct += (y_pred.argmax(1) == y).type(torch.float).sum().item()
                predictions = np.append(predictions,y_pred.cpu().numpy(),axis=0)
                actuals.extend(list(y.cpu().numpy()))
            val_loss = val_loss/len(eval_dataloader)
            accuracy = correct/size
            lr_scheduler.step(val_loss)
            # lr_scheduler.step()

            predictions_argmax = np.argmax(predictions,axis=1) 
            cf_matrix = confusion_matrix(np.array(actuals), predictions_argmax)
            fig = plt.figure(figsize=(30,30))
            plt.imshow(cf_matrix)
            writer.add_figure('cm', fig,epoch)
            writer.flush()

            if val_loss<best_loss:
                best_loss = val_loss
                torch.save(model.state_dict(), Config.BEST_LOSS_MODEL.format(epoch))
                print("Saving Model.....BL")
            if accuracy>best_accuracy:
                best_accuracy = accuracy
                torch.save(model.state_dict(), Config.BEST_ACC_MODEL.format(epoch))
                print("Saving Model.....BA")
        writer.add_scalar("Loss/Eval", val_loss, epoch)
        writer.add_scalar("Loss/Accuracy", accuracy*100, epoch)
        writer.flush()
        
        print(f"Epoch {epoch}:: Train Loss: {train_loss}; Eval Loss: {val_loss}; Accuracy: {accuracy*100}")
    writer.close()



















