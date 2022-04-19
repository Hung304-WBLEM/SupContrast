"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""
from __future__ import print_function

import torch
import torch.nn as nn


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            # print('[+] labels:', labels, labels.shape)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
            # print('[+] mask:', mask, mask.shape)
        else:
            mask = mask.float().to(device)
        # print('-' * 100)

        contrast_count = features.shape[1]
        # print('[+] contrast_count:', contrast_count)
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        # print('[+] contrast_feature:', contrast_feature.shape)
        # print('-' * 100)

        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # print('[+] self.contrast_mode:', self.contrast_mode)
        # print('[+] anchor_feature:', anchor_feature.shape)
        # print('[+] anchor_count:', anchor_count)
        # print('-' * 100)

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # print('[+] anchor_dot_contrast:', anchor_dot_contrast.shape)
        # print('-' * 100)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        # print('[+] logits_max:', logits_max.shape)
        logits = anchor_dot_contrast - logits_max.detach()
        # print('[+] logits:', logits.shape)
        # print('-' * 100)

        # tile mask
        # print('[+] mask before:', mask, mask.shape)
        mask = mask.repeat(anchor_count, contrast_count)
        # print('[+] mask after:', mask, mask.shape)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        # print('[+] logits_mask:', logits_mask, logits_mask.shape)
        mask = mask * logits_mask
        # print('-' * 100)

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        # mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        # avoid nan loss when there's one sample for a certain class, e.g., 0,1,...1 for bin-cls , this produce nan for 1st in Batch
        # which also results in batch total loss as nan. such row should be dropped
        pos_per_sample=mask.sum(1) #B
        pos_per_sample[pos_per_sample<1e-6]=1.0
        mean_log_prob_pos = (mask * log_prob).sum(1) / pos_per_sample #mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        # print('[+] loss before:', loss.shape)
        # print('[+] loss before:', loss.view(anchor_count, batch_size).shape)
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


if __name__ == '__main__':
    criterion = SupConLoss(temperature=0.5)

    bsz, n_views, f_dim = 4, 4, 10 
    features = torch.rand(bsz, n_views, f_dim)

    labels = torch.randint(3, (bsz,))

    # print('[+] Input features:', features, features.shape)
    # print('[+] Input labels:', labels, labels.shape)
    # print('-' * 100)

    loss = criterion(features, labels)

    print('[+] Loss:', loss)
