import torch
from torch import nn
from core.modules import LayerNorm
from typing import Optional
import torch.nn.functional as F

class UtteranceEncoder(nn.Module):

    def __init__(self, idim: int,
                 n_layers: int = 2,
                 n_chans: int = 256,
                 kernel_size: int = 5,
                 pool_kernel: int = 3,
                 dropout_rate: float = 0.5,
                 stride: int = 3):
        super(UtteranceEncoder, self).__init__()
        self.conv = torch.nn.ModuleList()
        for idx in range(n_layers):
            in_chans = idim if idx == 0 else n_chans
            self.conv += [
                torch.nn.Sequential(
                    torch.nn.Conv1d(
                        in_chans,
                        n_chans,
                        kernel_size,
                        stride=stride,
                        padding=(kernel_size - 1) // 2,
                    ),
                    torch.nn.ReLU(),
                    LayerNorm(n_chans),
                    torch.nn.Dropout(dropout_rate),
                )
            ]

    def forward(self,
                xs: torch.Tensor,
                x_masks: Optional[torch.Tensor] = None
                ) -> torch.Tensor:

        for f in self.conv:
            xs = f(xs)  # (B, C, Tmax)

        # NOTE: calculate in log domain
        xs = F.avg_pool1d(xs, xs.size(-1))  # (B, C, 1)

        return xs


class PhonemeLevelEncoder(nn.Module):

    def __init__(self, idim: int,
                    n_layers: int = 2,
                    n_chans: int = 256,
                    out: int = 4,
                    kernel_size: int = 3,
                    dropout_rate: float = 0.5,
                    stride: int = 1):
        super(PhonemeLevelEncoder, self).__init__()
        self.conv = torch.nn.ModuleList()
        for idx in range(n_layers):
            in_chans = idim if idx == 0 else n_chans
            self.conv += [
                torch.nn.Sequential(
                    torch.nn.Conv1d(
                        in_chans,
                        n_chans,
                        kernel_size,
                        stride=stride,
                        padding=(kernel_size - 1) // 2,
                    ),
                    torch.nn.ReLU(),
                    LayerNorm(n_chans),
                    torch.nn.Dropout(dropout_rate),
                )
            ]

        self.linear = torch.nn.Linear(n_chans, out)

    def forward(self,
                xs: torch.Tensor,
                x_masks: Optional[torch.Tensor] = None
                ) -> torch.Tensor:

        for f in self.conv:
            xs = f(xs)  # (B, C, Lmax)


        xs = self.linear(xs.transpose(1, 2))  # (B, Lmax, 4)

        return xs


class PhonemeLevelPredictor(nn.Module):

    def __init__(self, idim: int,
                 n_layers: int = 2,
                 n_chans: int = 256,
                 out: int = 4,
                 kernel_size: int = 3,
                 dropout_rate: float = 0.5,
                 stride: int = 1):
        super(PhonemeLevelPredictor, self).__init__()
        self.conv = torch.nn.ModuleList()
        for idx in range(n_layers):
            in_chans = idim if idx == 0 else n_chans
            self.conv += [
                torch.nn.Sequential(
                    torch.nn.Conv1d(
                        in_chans,
                        n_chans,
                        kernel_size,
                        stride=stride,
                        padding=(kernel_size - 1) // 2,
                    ),
                    torch.nn.ReLU(),
                    LayerNorm(n_chans),
                    torch.nn.Dropout(dropout_rate),
                )
            ]

        self.linear = torch.nn.Linear(n_chans, out)

    def forward(self,
                xs: torch.Tensor,
                x_masks: Optional[torch.Tensor] = None
                ) -> torch.Tensor:

        for f in self.conv:
            xs = f(xs)  # (B, C, Tmax)

        xs = self.linear(xs.transpose(1, 2))  # (B, Tmax)

        return xs

class AcousticPredictorLoss(torch.nn.Module):
    """Loss function module for duration predictor.
    The loss value is Calculated in log domain to make it Gaussian.
    """

    def __init__(self, offset=1.0):
        """Initilize duration predictor loss module.
        Args:
            offset (float, optional): Offset value to avoid nan in log domain.
        """
        super(AcousticPredictorLoss, self).__init__()
        self.criterion = torch.nn.MSELoss()
        self.offset = offset

    def forward(self, outputs, targets):
        """Calculate forward propagation.
        Args:
            outputs (Tensor): Batch of prediction durations in log domain (B, T)
            targets (LongTensor): Batch of groundtruth durations in linear domain (B, T)
        Returns:
            Tensor: Mean squared error loss value.
        Note:
            `outputs` is in log domain but `targets` is in linear domain.
        """
        # NOTE: outputs is in log domain while targets in linear
        loss = self.criterion(outputs, targets)

        return loss