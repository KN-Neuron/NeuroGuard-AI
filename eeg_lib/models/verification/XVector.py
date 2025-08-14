import torch
import torch.nn as nn
from torch.nn import LayerNorm
from torch.optim.lr_scheduler import StepLR, ExponentialLR, ReduceLROnPlateau
from torch import optim
from eeg_lib.losses.ProxyNCALoss import ProxyNCALoss
import torch.nn.functional as F
import math
from eeg_lib.losses.ArcFaceLoss import ArcMarginProduct

class StatisticsPooling(nn.Module):
    '''
    Last layer of standard TDNN model, computes mean and vairance across timesteps
    and concatenates the results
    '''
    def __init__(self, eps=1e-6):
        super(StatisticsPooling, self).__init__()
        self.eps = eps

    def forward(self, x):

        mean = torch.mean(x, dim=2)
        std = torch.sqrt(torch.var(x, dim=2, unbiased=False) + self.eps)
        pooled = torch.cat((mean, std), dim=1)
        return pooled


class EcapaRes2NetBlock(nn.Module):
    '''
    Ecapa Res2Net block for timeseries data uses Conv1d, used in ECAPA_TDNN model
    '''
    def __init__(self, in_channels, out_channels, split=4, kernel_size=3, padding=1):
        '''
        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param split: how many "slices" do we want to split the data into across feature channel
        :param kernel_size: kernel size
        :param padding: kernel padding
        '''
        super(EcapaRes2NetBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.split = split
        self.kernel_size = kernel_size
        self.w = self.in_channels // self.split

        self.pre_conv = nn.Conv1d(in_channels=in_channels,
                                  out_channels=in_channels,
                                  kernel_size=1,
                                  bias=False)
        self.pre_bn = nn.BatchNorm1d(in_channels)


        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=self.w,
                      out_channels=self.w,
                      kernel_size=kernel_size,
                      padding=padding,
                      bias=False)
            for _ in range(split-1)
        ])

        self.bns = nn.ModuleList([
            nn.BatchNorm1d(self.w) for _ in range(split-1)
        ])

        self.post_conv = nn.Conv1d(in_channels=in_channels,
                                   out_channels=out_channels,
                                   kernel_size=1,
                                   bias=False)
        self.post_bn = nn.BatchNorm1d(out_channels)

        if in_channels != out_channels:
            self.skip_conv = nn.Sequential(
                nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.skip_conv = nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        '''
        :param x: input tensor
        '''
        out = self.pre_conv(x)
        out = self.pre_bn(out)
        out = self.relu(out)

        slices = out.split(self.w, dim=1)

        outputs = []
        for i, Xi in enumerate(slices):
            if i == 0:
                Yi = Xi
            else:
                Yi = Xi + outputs[i-1]
                Yi = self.relu(self.bns[i-1](self.convs[i-1](Yi)))
            outputs.append(Yi)
        out = torch.cat(outputs, dim=1)
        out = self.post_conv(out)
        out = self.post_bn(out)
        out = self.relu(out)
        skip = self.skip_conv(x)
        return skip + out

class EcapaSEModule(nn.Module):
    '''
    Squeeze and excitation module used in ECAPA_TDNN model and in EcapaSEResBlock
    '''
    def __init__(self, in_channels, reduction=8):
        """
        :param in_channels: number of input channels
        :param reduction: how much do we want to lower the dimensionality of the input
        """
        super(EcapaSEModule, self).__init__()
        self.in_channels = in_channels
        self.reduction = reduction

        self.squeeze = nn.AdaptiveAvgPool1d(1)

        self.excitation = nn.Sequential( # Squeeze, Dense1, Relu, Dense2, Sigmoid
            nn.Conv1d(in_channels=in_channels, out_channels=in_channels//reduction, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv1d(in_channels=in_channels//reduction, out_channels=in_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        s = self.squeeze(x)
        e = self.excitation(s)
        return e * x

class EcapaSEResBlock(nn.Module):
    '''
    Res2Net block and squeeze-excitation integrated together, used in ECAPA_TDNNv2 model,
    this is the building block of the standard ECAPA_TDNN
    '''
    def __init__(self, in_channels, out_channels, split=4, kernel_size=3, padding=1, dilation=0):
        """
        :param in_channels input channels
        :param out_channels output channels
        :param split how many "slices" do we want to split the data into across feature channel
        :param kernel_size kernel size in the Res2Net block
        :param padding in the Res2net block
        :param dilation in the Res2net block
        """
        super(EcapaSEResBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.GroupNorm(1,out_channels)

        self.width = out_channels // split
        self.split = split
        self.convs = nn.ModuleList([
            nn.Conv1d(self.width, self.width, kernel_size=kernel_size, dilation=dilation, padding=(dilation * (kernel_size - 1)) // 2)
            for _ in range(split-1)
        ])

        self.bns = nn.ModuleList([
            nn.GroupNorm(1,self.width) for _ in range(split-1)
        ])
        self.conv2 = nn.Conv1d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=1)
        self.bn2 = nn.GroupNorm(1,out_channels)

        self.se =  EcapaSEModule(out_channels)

        self.shortcut = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, no_norm=False):
        residual = self.shortcut(x)
        x = self.relu(self.conv1(x))
        if not no_norm:
            x = self.bn1(x)

        frames = torch.split(x, self.width, 1)
        outputs = [frames[0]]
        y = frames[0]

        for i in range(len(self.convs)):
            y_conv = self.convs[i](y)
            if not no_norm:
                y_conv = self.bns[i](y_conv)
            y_conv = self.relu(y_conv)

            y = y_conv + frames[i+1]
            outputs.append(y)
        x = torch.cat(outputs, 1)

        x = self.conv2(x)
        if not no_norm:
            x = self.bn2(x)
        x = self.relu(x)

        x = self.se(x)
        return self.relu(x + residual)


class AttentiveStatisticsPooling(nn.Module):
    """
    Final layer in the ECAPA_TDNN architecture aggregates statistics across timesteps
    and scales them by the attention factor
    """
    def __init__(self, in_channels, eps=1e-6, reduction=128):
        '''
        :param in_channels - int number of input channels
        :param eps - float for stddev calculation
        :reduction - int dimensionality of the attention layer
        '''
        super(AttentiveStatisticsPooling, self).__init__()
        self.in_channels = in_channels
        self.eps = eps
        self.reduction = reduction
        self.att = nn.Sequential(
            nn.Linear(in_channels, reduction),
            nn.ReLU(),
            nn.Linear(reduction, 1)
        )

    def forward(self, x):
        batch_size, num_channels, time_steps = x.shape
        frames = x.transpose(1,2)
        scores = self.att(frames)
        weights = F.softmax(scores.squeeze(-1), dim=1)
        weights = weights.unsqueeze(-1)

        weighted_sum = torch.sum(weights * frames, dim=1)
        sqr_frames = frames * frames
        weighted_sqr_sum = torch.sum(weights * sqr_frames, dim=1)
        variance = weighted_sqr_sum - weighted_sum * weighted_sum
        stddev = torch.sqrt(torch.clamp(variance, min=self.eps))
        pooled = torch.cat((weighted_sum, stddev), dim=1)
        return pooled


class ECAPA_TDNN(nn.Module):
    """
    A non-standard ECAPA_TDNN model with separated SE and Res2Net blocks
    """
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
                 dropout1=0.25,
                 dropout2=0.25,
                 dropout3=0.25,
                 dropout4=0.25,
                 ):
        '''
        :param input_features - int number of input channels
        :param num_classes - int number of classes needed for pretraining with softmax, if None just output embeddings
        :param timesteps - int number of timesteps // redundant
        :param embedding_dim - int dimensionality of the output embedding
        :param layer1_filt - int number of filters in the first convolutional layer
        :param layer2_filt - int number of filters in the second convolutional layer
        :param layer3_filt - int number of filters in the third convolutional layer
        :param layer4_filt - int number of filters in the fourth convolutional layer
        :param layer5_filt - int number of filters in the fifth convolutional layer
        :param dropout1 - float for dropout
        :param dropout2 - float for dropout
        :param dropout3 - float for dropout
        :param dropout4 - float for dropout
        '''
        super(ECAPA_TDNN, self).__init__()
        self.layer1 = nn.Sequential(
            EcapaRes2NetBlock(in_channels=input_features,
                              out_channels=layer1_filt),
            EcapaSEModule(in_channels=layer1_filt)
        )
        self.dropout1 = nn.Dropout(dropout1)
        self.layer2 = nn.Sequential(
            EcapaRes2NetBlock(in_channels=layer1_filt,
                              out_channels=layer2_filt),
            EcapaSEModule(in_channels=layer2_filt)
        )
        self.dropout2 = nn.Dropout(dropout2)
        self.layer3 = nn.Sequential(
            EcapaRes2NetBlock(in_channels=layer2_filt,
                              out_channels=layer3_filt),
            EcapaSEModule(in_channels=layer3_filt)
        )
        self.dropout3 = nn.Dropout(dropout3)
        self.layer4 = nn.Sequential(
            EcapaRes2NetBlock(in_channels=layer3_filt,
                              out_channels=layer4_filt),
            EcapaSEModule(in_channels=layer4_filt)
        )
        self.dropout4 = nn.Dropout(dropout4)
        self.layer5 = nn.Sequential(
            EcapaRes2NetBlock(in_channels=layer4_filt,
                              out_channels=layer5_filt),
            EcapaSEModule(in_channels=layer5_filt)
        )
        self.stat_pooling = AttentiveStatisticsPooling(in_channels=layer5_filt)
        self.embeddingLayer = nn.Linear(in_features=layer5_filt*2, out_features=embedding_dim)
        if num_classes is not None:
            self.classifier = nn.Linear(in_features=embedding_dim, out_features=num_classes)
        else:
            self.classifier = None

    def forward(self, x, return_embedding=False, no_norm=False):
        x = self.layer1(x)
        x = self.dropout1(x)
        x = self.layer2(x)
        x = self.dropout2(x)
        x = self.layer3(x)
        x = self.dropout3(x)
        x = self.layer4(x)
        x = self.dropout4(x)
        x = self.layer5(x)
        x = self.stat_pooling(x)
        x = self.embeddingLayer(x)
        if self.classifier is not None:
            x = self.classifier(x)
        return x


class ECAPA_TDNNv2(nn.Module):
    """
    Standard ECAPA_TDNN model with SE and Res2Net blocks integrated together
    """
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
                 dropout1=0.25,
                 dropout2=0.25,
                 dropout3=0.25,
                 dropout4=0.25):
        super().__init__()
        '''
        :param input_features - int numeber of input features
        :param num_classes - int number of classes needed for pretraining with softmax, if None just output embeddings
        :param timesteps - int number of timesteps // redundant
        :param embedding_dim - int dimensionality of the output embedding
        :param layer1_filt - int number of filters in the first EcapaSEResBlock
        :param layer2_filt - int number of filters in the second EcapaSEResBlock
        :param layer3_filt - int number of filters in the third EcapaSEResBlock
        :param layer4_filt - int number of filters in the fourth EcapaSEResBlock
        :param layer5_filt - int number of filters in the fifth EcapaSEResBlock // redundant
        :param dropout1 - float for dropout
        :param dropout2 - float for dropout
        :param dropout3 - float for dropout
        :param dropout4 - float for dropout // redundant
        '''

        self.relu = nn.ReLU()

        self.layer1 = EcapaSEResBlock(
            in_channels=input_features,
            out_channels=layer1_filt,
            kernel_size=3,
            dilation=2
        )
        self.dropout1 = nn.Dropout(dropout1)
        self.layer2 = EcapaSEResBlock(
            in_channels=layer1_filt,
            out_channels=layer2_filt,
            kernel_size=3,
            dilation=3
        )
        self.dropout2 = nn.Dropout(dropout2)
        self.layer3 = EcapaSEResBlock(
            in_channels=layer2_filt,
            out_channels=layer3_filt,
            kernel_size=3,
            dilation=4
        )

        self.dropout3 = nn.Dropout(dropout3)

        self.layer4 = EcapaSEResBlock(
            in_channels=layer3_filt,
            out_channels=layer4_filt,
            kernel_size=3,
            dilation=5
        )
        #
        # self.layer5 = EcapaSEResBlock(
        #     in_channels=layer4_filt,
        #     out_channels=layer5_filt,
        #     kernel_size=3,
        #     dilation=6
        # )

        self.cat_conv = nn.Conv1d(layer1_filt + layer2_filt + layer3_filt + layer4_filt, 1500, kernel_size=1)
        self.pooling = AttentiveStatisticsPooling(in_channels=1500)

        self.ln = nn.GroupNorm(1,3000)
        self.dense = nn.Linear(in_features=3000, out_features=1024)
        self.embeddingLayer = nn.Linear(in_features=1024, out_features=embedding_dim)
        if num_classes is not None:
            self.classifier = nn.Linear(in_features=embedding_dim, out_features=num_classes)
        else:
            self.classifier = None

    def forward(self, x, return_embedding=False, no_norm=True):
        out1 = self.layer1(x)
        x = self.dropout1(out1)
        out2 = self.layer2(x)
        x = self.dropout2(out2)
        out3 = self.layer3(x)
        x = self.dropout3(out3)
        out4 = self.layer4(x)


        x_cat = torch.cat((out1,out2,out3, out4) ,dim=1)
        x = self.relu(self.cat_conv(x_cat))

        x = self.pooling(x)
        if not no_norm:
            x = self.ln(x)
        x = self.dense(x)
        x = self.relu(x)
        x = self.embeddingLayer(x)
        x = F.normalize(x,p=2,dim=1)
        if return_embedding or self.classifier is None:
            return x
        else:
            x = self.classifier(x)
            return x


class XVectorEmbeddingModel(nn.Module):
    """
    Standard TDNN Model
    """
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


def get_standard_model(hparams, input_features, num_classes):
    """
    Utility function for creating standard TDNN model with hparams dictionary, instead of writing it all explicitly
    """
    embedding_dim = hparams["embedding_dim"]
    dropout_rate = hparams["dropout_rate"]

    model = XVectorEmbeddingModel(input_features=input_features,
                                  num_classes=num_classes,
                                  embedding_dim=embedding_dim,
                                  dropout1=dropout_rate,
                                  dropout2=dropout_rate,
                                  dropout3=dropout_rate,
                                  dropout4=dropout_rate,
                                  layer1_filt=hparams["layer1_filters"],
                                  layer2_filt=hparams["layer2_filters"],
                                  layer3_filt=hparams["layer3_filters"],
                                  layer4_filt=hparams["layer4_filters"],
                                  layer5_filt=hparams["layer5_filters"],
                                  layer_1_dilatation=hparams["layer_1_dilatation"],
                                  layer_2_dilatation=hparams["layer_2_dilatation"],
                                  layer_3_dilatation=hparams["layer_3_dilatation"],
                                  layer_1_stride=hparams["layer_1_stride"],
                                  layer_2_stride=hparams["layer_2_stride"],
                                  layer_3_stride=hparams["layer_3_stride"], )
    return model


def get_ecapa_model(hparams, input_features, num_classes):
    """
    Utility function for creating ECAPA_TDNN model with hparams dictionary, instead of writing it all explicitly
    """
    embedding_dim = hparams["embedding_dim"]
    dropout_rate = hparams["dropout_rate"]

    model = ECAPA_TDNN(input_features=input_features,
                                  num_classes=num_classes,
                                  embedding_dim=embedding_dim,
                                  dropout1=dropout_rate,
                                  dropout2=dropout_rate,
                                  dropout3=dropout_rate,
                                  dropout4=dropout_rate,
                                  layer1_filt=hparams["layer1_filters"],
                                  layer2_filt=hparams["layer2_filters"],
                                  layer3_filt=hparams["layer3_filters"],
                                  layer4_filt=hparams["layer4_filters"],
                                  layer5_filt=hparams["layer5_filters"])
    return model


def get_ecapa2_model(hparams, input_features, num_classes):
    """
    Utility function for creating ECAPA_TDNNv2 model with hparams dictionary, instead of writing it all explicitly
    """
    embedding_dim = hparams["embedding_dim"]
    dropout_rate = hparams["dropout_rate"]

    model = ECAPA_TDNNv2(input_features=input_features,
                                  num_classes=num_classes,
                                  embedding_dim=embedding_dim,
                                  dropout1=dropout_rate,
                                  dropout2=dropout_rate,
                                  dropout3=dropout_rate,
                                  dropout4=dropout_rate,
                                  layer1_filt=hparams["layer1_filters"],
                                  layer2_filt=hparams["layer2_filters"],
                                  layer3_filt=hparams["layer3_filters"],
                                  layer4_filt=hparams["layer4_filters"],
                                  layer5_filt=hparams["layer5_filters"])
    return model


def pretrain(hparams, device, input_features, num_classes, dataloader, writer=None, type="standard", fold=""):
    """
    function creates a model and then pretrains the model using softmax as Loss Function
    :param hparams: dict of hyperparameters
    :param device: cpu or gpu
    :param input_features: input features
    :param num_classes: number of classes
    :param dataloader: dataloader
    :param writer: optional SummaryWriter object
    :param type: "standard" for standard TDNN model, ECAPA for ECAPA_TDNN model anything else for ECAPA_TDNNv2 model
    """
    lr = hparams["softmax_learning_rate"]
    weight_decay = hparams["softmax_learning_rate_decay"]
    epochs = hparams["softmax_epochs"]

    if type == "standard":
        model = get_standard_model(hparams, input_features, num_classes).to(device)
    elif type == "ECAPA":
        model = get_ecapa_model(hparams, input_features, num_classes).to(device)
    else:
        model = get_ecapa2_model(hparams, input_features, num_classes).to(device)
    if writer is not None and (fold=="" or fold==1):
        rnd_sample = torch.randn(1, input_features, num_classes).to(device)
        writer.add_graph(model, rnd_sample)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = ExponentialLR(optimizer, gamma=weight_decay)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        total_correct = 0
        total_seen = 0

        for data, labels in dataloader:
            data = data.to(device)
            labels = labels.to(device).long()
            optimizer.zero_grad()
            logits = model(data)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * data.size(0)
            pred = logits.argmax(dim=1)
            total_correct += (pred == labels).sum().item()
            total_seen += data.size(0)

        avg_loss = total_loss / total_seen
        acc = total_correct / total_seen
        if writer is not None:
            writer.add_scalar(f"Pretrain/Loss{fold}", avg_loss, epoch)
            writer.add_scalar(f"Pretrain/Accuracy{fold}", acc, epoch)
        print(f"[Pretrain] Epoch {epoch + 1}/{epochs}  Loss={avg_loss:.4f}  Acc={acc:.4f}")
        scheduler.step()
    return model


def fine_tune(model, hparams, device, dataloader, num_classes, writer=None, fold=""):
    """
    Function training a model with ProxyNCALoss

    :param model: nn.Module -  model to fine-tune
    :param hparams: dict - dictionary of hyperparameters
    :param device: cpu or gpu
    :param dataloader: dataloader
    :param num_classes: int - number of classes
    :param writer: SummaryWriter - optional SummaryWriter object
    """
    lr = hparams["proxy_learning_rate"]
    weight_decay = hparams["proxy_learning_rate_decay"]
    epochs = hparams["proxy_epochs"]
    scale = hparams["scale"]
    embedding_dim = hparams["embedding_dim"]

    model.classifier = None
    proxy_loss = ProxyNCALoss(num_classes=num_classes,
                              embedding_dim=embedding_dim, scale=scale).to(device)
    optimizer = optim.Adam(list(model.parameters()) + list(proxy_loss.parameters()), lr=lr)
    scheduler = ExponentialLR(optimizer, gamma=weight_decay)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for data, labels in dataloader:
            data, labels = data.to(device), labels.to(device).long()
            optimizer.zero_grad()
            emb = model(data, return_embedding=True, no_norm=True)
            loss = proxy_loss(emb, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * data.size(0)
        avg_loss = total_loss / len(dataloader.dataset)
        if writer is not None:
            writer.add_scalar(f"Finetune/Loss{fold}", avg_loss, epoch)
        print(f"[Fine-tune] Epoch {epoch + 1}/{epochs}  Loss={avg_loss:.4f}")
        scheduler.step()
    return model


def create_embeddings(model, X_train, X_test, hparams):
    """
    Function creates embedding vectors using an already trained model

    :param model: nn.Module -  model used to create embeddings
    :param X_train: np.array - training data
    :param X_test: np.array - test data
    :param hparams: dict - dictionary of hyperparameters
    """
    embeddings = []
    test_embeddings = []

    with torch.no_grad():
        for epoch in X_train:
            embeddings.append(
                model(torch.tensor(epoch, dtype=torch.float, requires_grad=False).unsqueeze(0), return_embedding=True,
                      no_norm=False))
        for epoch in X_test:
            test_embeddings.append(
                model(torch.tensor(epoch, dtype=torch.float, requires_grad=False).unsqueeze(0), return_embedding=True,
                      no_norm=False))
    embd = torch.stack(embeddings).reshape((X_train.shape[0], hparams["embedding_dim"])).numpy()
    test_embd = torch.stack(test_embeddings).reshape((X_test.shape[0], hparams["embedding_dim"])).numpy()
    return embd, test_embd




def fine_tune_arcface(model, hparams, device, dataloader, num_classes, writer=None, return_final_loss=False, fold=""):
    """
    Function training an exisiting model using ArcFace loss

    :param model: nn.Module - model to be trained
    :param hparams: dict - dictionary of hyperparameters
    :param device - cpu or gpu
    :param dataloader: dataloader object - training set
    :param num_classes: int - number of classes
    :param writer: tensorboard writer object - optional
    """
    lr      = hparams["proxy_learning_rate"]
    decay   = hparams["proxy_learning_rate_decay"]
    epochs  = hparams["proxy_epochs"]
    emb_dim = hparams["embedding_dim"]

    # remove old classifier / proxies
    model.classifier = None

    # create ArcFace head
    arcface = ArcMarginProduct(
        in_features=emb_dim,
        out_features=num_classes,
        s=hparams.get("scale", 1.0),
        m=hparams.get("margin", 0.001),
        easy_margin=hparams.get("easy_margin", False)
    ).to(device)

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(arcface.parameters()),
        lr=lr
    )
    scheduler = ExponentialLR(optimizer, gamma=decay)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for data, labels in dataloader:
            data, labels = data.to(device), labels.to(device).long()
            optimizer.zero_grad()
            # get embeddings (no classifier)
            emb = model(data, return_embedding=True, no_norm=False)
            # compute arcface loss
            loss = arcface(emb, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * data.size(0)

        avg_loss = total_loss / len(dataloader.dataset)
        if writer:
            writer.add_scalar(f"Finetune/ArcFaceLoss{fold}", avg_loss, epoch)
        print(f"[ArcFace Fine‑tune] Epoch {epoch+1}/{epochs}  Loss={avg_loss:.4f}")
        scheduler.step()
    if return_final_loss:
        return model, avg_loss
    return model
