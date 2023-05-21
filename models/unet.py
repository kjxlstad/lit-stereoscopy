import torch
from torch import nn
from torch.nn import functional as F

from lightning import LightningModule
from lightning.pytorch.utilities.types import STEP_OUTPUT

from data.dataset import Batch
from models.metrics import masked_l1, threshold_error, end_point_error, d1_error


def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Encoder, self).__init__()
        self.conv_block = conv_block(in_channels, out_channels)
        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        skip = self.conv_block(x)
        x = self.downsample(skip)
        return x, skip


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()
        self.conv_block = conv_block(in_channels, out_channels)
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x, skip):
        x = self.upsample(x)
        x = torch.cat((x, skip), dim=1)
        x = self.conv_block(x)
        return x


class UNet(LightningModule):
    def __init__(self, in_channels=3, out_channels=1, init_features: int = 32) -> None:
        super(UNet, self).__init__()
        self.max_disp = 192
        features = init_features

        self.encoder_1 = Encoder(in_channels, features)
        self.encoder_2 = Encoder(features, features * 2)
        self.encoder_3 = Encoder(features * 2, features * 4)
        self.encoder_4 = Encoder(features * 4, features * 8)

        self.bottleneck = conv_block(features * 8, features * 16)

        self.decoder_4 = Decoder(features * 16, features * 8)
        self.decoder_3 = Decoder(features * 8, features * 4)
        self.decoder_2 = Decoder(features * 4, features * 2)
        self.decoder_1 = Decoder(features * 2, features)

        self.conv = nn.Conv2d(in_channels=features, out_channels=out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, skip_1 = self.encoder_1(x)
        x, skip_2 = self.encoder_2(x)
        x, skip_3 = self.encoder_3(x)
        x, skip_4 = self.encoder_4(x)

        x = self.bottleneck(x)

        x = self.decoder_4(x, skip_4)
        x = self.decoder_3(x, skip_3)
        x = self.decoder_2(x, skip_2)
        x = self.decoder_1(x, skip_1)

        x = self.conv(x)

        return torch.sigmoid(x)

    def training_step(self, batch: Batch, batch_idx: int) -> STEP_OUTPUT:
        input_image = torch.cat((batch["image_left"], batch["image_right"]), dim=1)
        disp_pred = self(input_image)

        mask = (batch["disparity"] < self.max_disp) & (batch["disparity"] > 0)
        loss = masked_l1(disp_pred, batch["disparity"], mask)

        with torch.no_grad():
            metrics = {
                "epe": end_point_error(disp_pred, batch["disparity"], mask),
                "d1": d1_error(disp_pred, batch["disparity"], mask),
                "thres_1": threshold_error(disp_pred, batch["disparity"], mask, 1.0),
                "thres_2": threshold_error(disp_pred, batch["disparity"], mask, 2.0),
                "thres_3": threshold_error(disp_pred, batch["disparity"], mask, 3.0),
            }
            self.log_dict({f"train/{m}": v.item() for m, v in metrics.items()}, on_step=False, on_epoch=True)

        self.log("loss", loss, on_step=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Batch, batch_idx: int) -> STEP_OUTPUT:
        input_image = torch.cat((batch["image_left"], batch["image_right"]), dim=1)
        disp_pred = self(input_image)

        mask = (batch["disparity"] < self.max_disp) & (batch["disparity"] > 0)

        metrics = {
            "loss": masked_l1(disp_pred, batch["disparity"], mask),
            "epe": end_point_error(disp_pred, batch["disparity"], mask),
            "d1": d1_error(disp_pred, batch["disparity"], mask),
            "thres_1": threshold_error(disp_pred, batch["disparity"], mask, 1.0),
            "thres_2": threshold_error(disp_pred, batch["disparity"], mask, 2.0),
            "thres_3": threshold_error(disp_pred, batch["disparity"], mask, 3.0),
        }

        self.log_dict({f"val/{m}": v.item() for m, v in metrics.items()}, on_step=False, on_epoch=True)
        return metrics

    def test_step(self, batch: Batch, batch_idx: int) -> STEP_OUTPUT:
        input_image = torch.cat((batch["image_left"], batch["image_right"]), dim=1)
        disp_pred = self(input_image)

        mask = (batch["disparity"] < self.max_disp) & (batch["disparity"] > 0)

        metrics = {
            "loss": masked_l1(disp_pred, batch["disparity"], mask),
            "epe": end_point_error(disp_pred, batch["disparity"], mask),
            "d1": d1_error(disp_pred, batch["disparity"], mask),
            "thres_1": threshold_error(disp_pred, batch["disparity"], mask, 1.0),
            "thres_2": threshold_error(disp_pred, batch["disparity"], mask, 2.0),
            "thres_3": threshold_error(disp_pred, batch["disparity"], mask, 3.0),
        }

        self.log_dict({f"test/{m}": v.item() for m, v in metrics.items()}, on_step=False, on_epoch=True)
        return metrics

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=1e-3)
