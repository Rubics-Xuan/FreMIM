import torch
import torch.nn as nn
import logging
import torch.fft as fft


# TODO attention the output of model to use loss

class DiceLoss(nn.Module):
    """2D Cross Entropy Loss with Multi-L1oss"""
    def __init__(self):
        super(DiceLoss, self).__init__()

        self.Softmax = nn.Softmax(dim=1)

    def forward(self, *inputs):
        pred, target = tuple(inputs)
        pred = self.Softmax(pred)
        loss = softmax_dice(pred, target)
        return loss


def softmax_dice(output, target):
    '''
    The dice loss for using softmax activation function
    :param output: (b, num_class, d, h, w)
    :param target: (b, d, h, w)
    :return: softmax dice loss
    '''
    loss1 = Dice(output[:, 1, ...], (target == 1).float())
    loss2 = Dice(output[:, 2, ...], (target == 2).float())
    loss3 = Dice(output[:, 3, ...], (target == 4).float())

    return loss1 + loss2 + loss3, 1-loss1.data, 1-loss2.data, 1-loss3.data


def Dice(output, target, eps=1e-5):
    target = target.float()
    num = 2 * (output * target).sum()
    den = output.sum() + target.sum() + eps
    return 1.0 - num/den



class FocalFrequencyLoss_2D_pooling(nn.Module):
    """The torch.nn.Module class that implements focal frequency loss - a
    frequency domain loss function for optimizing generative models.
    Ref:
    Focal Frequency Loss for Image Reconstruction and Synthesis. In ICCV 2021.
    <https://arxiv.org/pdf/2012.12821.pdf>
    Args:
        loss_weight (float): weight for focal frequency loss. Default: 1.0
        alpha (float): the scaling factor alpha of the spectrum weight matrix for flexibility. Default: 1.0
        patch_factor (int): the factor to crop image patches for patch-based focal frequency loss. Default: 1
        ave_spectrum (bool): whether to use minibatch average spectrum. Default: False
        log_matrix (bool): whether to adjust the spectrum weight matrix by logarithm. Default: False
        batch_matrix (bool): whether to calculate the spectrum weight matrix using batch-based statistics. Default: False
    """

    def __init__(self, loss_weight=1.0, alpha=1.0, patch_factor=1, pass_band=20, log_matrix=False, batch_matrix=False):
        super(FocalFrequencyLoss_2D_pooling, self).__init__()
        self.loss_weight = loss_weight
        self.alpha = alpha
        self.patch_factor = patch_factor
        self.log_matrix = log_matrix
        self.batch_matrix = batch_matrix
        self.passband = pass_band
        self.pooling = nn.AdaptiveAvgPool2d((int(128 / 4), int(128 / 4)))

    def tensor2freq_image(self, x, pass_manner, stage_manner):

        if stage_manner == 'high_level':
            x = self.pooling(x)

        if pass_manner == 'high':
            freq = torch.fft.fft2(x, dim=[2,3], norm='ortho')
            freq = fft.fftshift(freq)
            H, W = x.size(2), x.size(3)
            h_crop, w_crop = int(H/2), int(W/2)
            freq[:, :, h_crop-self.passband:h_crop+self.passband, w_crop-self.passband:w_crop+self.passband] = 0
        elif pass_manner == 'low':
            freq = torch.fft.fft2(x, dim=[2,3], norm='ortho')
            freq = fft.fftshift(freq)
            H, W = x.size(2), x.size(3)
            h_crop, w_crop = int(H/2), int(W/2)
            mask = torch.zeros((x.size(0), x.size(1), H, W), dtype=torch.uint8).cuda()
            mask[:, :, h_crop-self.passband:h_crop+self.passband, w_crop-self.passband:w_crop+self.passband] = 1
            freq = freq * mask
        else:
            freq = torch.fft.fft2(x, dim=[2,3], norm='ortho')
            freq = fft.fftshift(freq)

        freq = torch.stack([freq.real, freq.imag], -1)

        return freq

    def tensor2freq_predict(self, x, pass_manner):

        freq = torch.fft.fft2(x, dim=[2,3], norm='ortho')
        freq = fft.fftshift(freq)

        
        if pass_manner == 'high':
            H, W = x.size(2), x.size(3)
            h_crop, w_crop = int(H/2), int(W/2)
            freq[:, :, h_crop-self.passband:h_crop+self.passband, w_crop-self.passband:w_crop+self.passband] = 0
        elif pass_manner == 'low':
            H, W = x.size(2), x.size(3)
            h_crop, w_crop = int(H/2), int(W/2)
            mask = torch.zeros((x.size(0), x.size(1), H, W), dtype=torch.uint8).cuda()
            mask[:, :, h_crop-self.passband:h_crop+self.passband, w_crop-self.passband:w_crop+self.passband] = 1
            freq = freq * mask
        else:
            pass

        freq = torch.stack([freq.real, freq.imag], -1)

        return freq

    def loss_formulation(self, recon_freq, real_freq, matrix=None):
        # spectrum weight matrix
        if matrix is not None:
            # if the matrix is predefined
            weight_matrix = matrix.detach()
        else:
            # if the matrix is calculated online: continuous, dynamic, based on current Euclidean distance
            matrix_tmp = (recon_freq - real_freq) ** 2
            matrix_tmp = torch.sqrt(matrix_tmp[..., 0] + matrix_tmp[..., 1]) ** self.alpha

            # whether to adjust the spectrum weight matrix by logarithm
            if self.log_matrix:
                matrix_tmp = torch.log(matrix_tmp + 1.0)

            # whether to calculate the spectrum weight matrix using batch-based statistics
            if self.batch_matrix:
                matrix_tmp = matrix_tmp / matrix_tmp.max()
            else:
                # print(matrix_tmp.size())
                matrix_tmp = matrix_tmp / matrix_tmp.max(-1).values.max(-1).values[:, :, None, None]

            matrix_tmp[torch.isnan(matrix_tmp)] = 0.0
            matrix_tmp = torch.clamp(matrix_tmp, min=0.0, max=1.0)
            weight_matrix = matrix_tmp.clone().detach()

        assert weight_matrix.min().item() >= 0 and weight_matrix.max().item() <= 1, (
            'The values of spectrum weight matrix should be in the range [0, 1], '
            'but got Min: %.10f Max: %.10f' % (weight_matrix.min().item(), weight_matrix.max().item()))

        # frequency distance using (squared) Euclidean distance
        tmp = (recon_freq - real_freq) ** 2
        freq_distance = tmp[..., 0] + tmp[..., 1]

        # dynamic spectrum weighting (Hadamard product)
        loss = weight_matrix * freq_distance
        return torch.mean(loss)


    def forward(self, pred, target, pass_manner, stage_manner, ave_spectrum, matrix=None, **kwargs):
        """Forward function to calculate focal frequency loss.
        Args:
            pred (torch.Tensor): of shape (N, C, H, W). Predicted tensor.
            target (torch.Tensor): of shape (N, C, H, W). Target tensor.
            matrix (torch.Tensor, optional): Element-wise spectrum weight matrix.
                Default: None (If set to None: calculated online, dynamic).
        """
        pred_freq = self.tensor2freq_predict(pred, pass_manner)
        target_freq = self.tensor2freq_image(target, pass_manner, stage_manner)

        # whether to use minibatch average spectrum
        if ave_spectrum:
            pred_freq = torch.mean(pred_freq, 0, keepdim=True)
            target_freq = torch.mean(target_freq, 0, keepdim=True)

        return self.loss_formulation(pred_freq, target_freq, matrix) * self.loss_weight