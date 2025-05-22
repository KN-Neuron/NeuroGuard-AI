import torch
import torch.nn as nn

class StatisticsPooling(nn.Module):
    def __init__(self, eps=1e-6):
        super(StatisticsPooling, self).__init__()
        self.eps = eps

    def forward(self, x):

        mean = torch.mean(x, dim=2)

        std = torch.sqrt(torch.var(x, dim=2, unbiased=False) + self.eps)

        pooled = torch.cat((mean, std), dim=1)
        return pooled

class XVectorEmbeddingModel(nn.Module):
    def __init__(self,
                 input_features,
                 num_classes=None,
                 timesteps=751,
                 embedding_dim=32,
                 layer1_filt=512,
                 layer2_filt=512,
                 layer3_filt=1024,
                 layer4_filt=1024,
                 layer5_filt=1500,
                 layer_1_dilatation=1,
                 layer_2_dilatation=2,
                 layer_3_dilatation=3,
                 layer_1_stride=1,
                 layer_2_stride=1,
                 layer_3_stride=2,
                 dropout1=0.25,
                 dropout2=0.25,
                 dropout3=0.25,
                 dropout4=0.25,
                 ):

        super(XVectorEmbeddingModel, self).__init__()

        self.relu = nn.ReLU()
        self.Frame1 = nn.Conv1d(in_channels=input_features,
                                out_channels=layer1_filt,
                                kernel_size=5,
                                stride=layer_1_stride,
                                dilation=layer_1_dilatation,
                                padding=2,
                                )
        # self.bn1 = nn.BatchNorm1d(layer1_filt)
        self.bn1 = nn.LayerNorm(layer1_filt)
        self.dropout1 = nn.Dropout(p=dropout1)
        self.Frame2 = nn.Conv1d(in_channels=layer1_filt,
                                out_channels=layer2_filt,
                                kernel_size=3,
                                stride=layer_2_stride,
                                dilation=layer_2_dilatation,
                                padding=2)
        self.bn2 = nn.LayerNorm(layer2_filt)

        self.dropout2 = nn.Dropout(p=dropout2)

        self.Frame3 = nn.Conv1d(in_channels=layer2_filt,
                                out_channels=layer3_filt,
                                kernel_size=3,
                                stride=layer_3_stride,
                                dilation=layer_3_dilatation,
                                padding=3)
        self.bn3 = nn.LayerNorm(layer3_filt)

        self.dropout3 = nn.Dropout(p=dropout3)

        self.Frame4 = nn.Conv1d(in_channels=layer3_filt,
                                out_channels=layer4_filt,
                                kernel_size=1,
                                stride=1,
                                padding=0)

        self.bn4 = nn.LayerNorm(layer4_filt)

        self.dropout4 = nn.Dropout(p=dropout4)

        self.Frame5 = nn.Conv1d(in_channels=layer4_filt,
                                out_channels=layer5_filt,
                                kernel_size=1,
                                stride=1,
                                padding=0)

        # self.bn5 = nn.BatchNorm1d(layer5_filt)
        self.bn5 = nn.LayerNorm(layer5_filt)

        self.statPooling = StatisticsPooling(eps=1e-6)

        self.embeddingLayer = nn.Linear(in_features=layer5_filt*2,
                                        out_features=embedding_dim)
        if num_classes is not None:
            self.classifier = nn.Linear(embedding_dim, num_classes)
        else:
            self.classifier = None
    def forward(self, x, return_embedding=False, no_norm=False):
        x = self.Frame1(x)
        if not no_norm:
            x = x.permute(0, 2, 1)
            x = self.bn1(x)
            x = x.permute(0, 2, 1)
        x = self.dropout1(x)
        # x = self.bn1(x)
        x = self.relu(x)
        # print(x.shape)
        x = self.Frame2(x)
        if not no_norm:
            x = x.permute(0, 2, 1)
            x = self.bn2(x)
            x = x.permute(0, 2, 1)
        x = self.dropout2(x)
        # x = self.bn2(x)
        x = self.relu(x)
        # print(x.shape)
        x = self.Frame3(x)
        if not no_norm:
            x = x.permute(0, 2, 1)
            x = self.bn3(x)
            x = x.permute(0, 2, 1)
        x = self.dropout3(x)
        # x = self.bn3(x)
        x = self.relu(x)
        # print(x.shape)
        x = self.Frame4(x)
        if not no_norm:
            x = x.permute(0, 2, 1)
            x = self.bn4(x)
            x = x.permute(0, 2, 1)
        x = self.dropout4(x)
        # x = self.bn4(x)
        x = self.relu(x)
        # print(x.shape)
        x = self.Frame5(x)
        if not no_norm:
            x = x.permute(0, 2, 1)
            x = self.bn5(x)
            x = x.permute(0, 2, 1)
        x = self.relu(x)
        # print(x.shape)
        x = self.statPooling(x)
        # print(x.shape)
        x = self.embeddingLayer(x)
        if return_embedding or self.classifier is None:
            return x
        else:
            return self.classifier(x)

