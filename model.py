import re, torch
import torch.nn as nn
import torch.nn.functional as F
from KAN import KANLinear



class TextCNN(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_sizes):
        super(TextCNN, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(input_dim, output_dim, kernel_size=k, padding=k // 2) for k in kernel_sizes
        ])
        self.bn = nn.BatchNorm1d(output_dim)
        self.act = nn.ReLU()
        

    def forward(self, x):
        features = [self.act(self.bn(conv(x))) for conv in self.convs]
        return torch.cat(features, dim=1)

    
class Plan_Ankh_ProstT5(nn.Module):
    def __init__(self):
        super(Plan_Ankh_ProstT5, self).__init__()
        self.textcnn_esm2 = TextCNN(input_dim=1536, output_dim=256, kernel_sizes=[1, 3, 5])
        self.textcnn_prot = TextCNN(input_dim=1024, output_dim=256, kernel_sizes=[1, 3, 5])
        self.bn = nn.BatchNorm1d(256)
        self.bn_con = nn.BatchNorm1d(512)
        self.act = nn.ReLU()
        self.lstmcell = nn.LSTM(256 * 3, 128, bidirectional=True) 
        self.classifier = nn.Sequential(
            KANLinear(512, 128),
            KANLinear(128, 2),
            nn.Softmax(-1)
        )

    def forward(self, Ankh_embeding, prostT5_embeding):
        Ankh_features = Ankh_embeding.permute(0, 2, 1)  # [batch_size, 1280, seq_len]
        Ankh_features = self.textcnn_esm2(Ankh_features)  # [batch_size, 256 * len(kernel_sizes), seq_len]
        Ankh_features = Ankh_features.permute(0, 2, 1)  # [batch_size, seq_len, 256 * len(kernel_sizes)]

        prost_features = prostT5_embeding.permute(0, 2, 1)  # [batch_size, 1024, seq_len]
        prost_features = self.textcnn_prot(prost_features)  # [batch_size, 256 * len(kernel_sizes), seq_len]
        prost_features = prost_features.permute(0, 2, 1)  # [batch_size, seq_len, 256 * len(kernel_sizes)]

        fea1,_ = self.lstmcell(Ankh_features)
        fea2,_ = self.lstmcell(prost_features)
        fin_fea = torch.cat([fea1, fea2], dim=2)
        out = self.classifier(fin_fea)
        return out
