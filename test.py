import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, matthews_corrcoef, accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, auc, precision_recall_curve,roc_curve
from data import ProteinLigandData, LigandData, BatchCollate
from torch.utils.data import Dataset, DataLoader
from losses import TripletCenterLoss, FocalLoss, CrossEntropy, InfoNCELoss
import numpy as np
import pytorch_lightning as pl
import count
from sklearn.model_selection import KFold
pl.seed_everything(42)
import torch.utils.data as data
from torch.nn.utils.rnn import pad_sequence
from model import *
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

class Getdata(data.Dataset):
    def __init__(self, ID_list):
        super(Getdata, self).__init__()
        self.IDs = ID_list
    
    def __len__(self):
        return len(self.IDs)

    def __getitem__(self, idx):
        return self._feature_(idx)
    
    def _feature_(self, idx):
        name = self.IDs[idx]
        with torch.no_grad():
            embedding1 = torch.load("../prostt5_embedding/" + name + ".tensor")
            embedding2 = torch.load("../ankh_embedding/" + name + ".tensor")
        return embedding2, embedding1

class BatchCollate(object):
    def __call__(self, batch):
        features1, features2 = zip(*batch)
        # 填充特征数据
        features1 = [torch.tensor(feature) for feature in features1]
        feature1 = pad_sequence(features1, batch_first=True)  # ([bz, length, dim])
        features2 = [torch.tensor(feature) for feature in features2]
        feature2 = pad_sequence(features2, batch_first=True)  # ([bz, length, dim])
        return feature1, feature2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_text = "ATP17.txt"
protein_ids = []
with open(test_text, 'r') as file:
    lines = file.readlines()
    for i in range(0, len(lines), 2):
        protein_id = lines[i].split(" ")[0]
        protein_ids.append(protein_id)

test_dataset = Getdata(ID_list=protein_ids)
test_loader = DataLoader(test_dataset, collate_fn=BatchCollate(), shuffle=True, num_workers=8)

best_model = Plan_Ankh_ProstT5()
if torch.cuda.device_count() > 1:
    best_model = nn.DataParallel(best_model)
best_model.eval()
best_model.to(device)
best_model.load_state_dict(torch.load(f"../save_new/model/227_Ankh_ProstT5.pt"))

with torch.no_grad():

    for data1, data2 in test_loader:
        data1 = data1.to(device)
        data2 = data2.to(device)
        score = best_model(data1, data2)
        print(score)
