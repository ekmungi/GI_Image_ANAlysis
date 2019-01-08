import torch
import torch.nn as nn


def make_one_hot(labels, num_classes=2):
    # labels.unsqueeze_(1)
    one_hot = torch.FloatTensor(labels.size(0), num_classes, labels.size(2), labels.size(3)).zero_()

    labels_onehot = one_hot.scatter_(1, labels.data.long(), 1)
    # print(labels_onehot.size())

    return labels_onehot

    # labels_dims_number = labels.dim()

    # # Add a singleton dim -- we need this for scatter
    # labels_ = labels.unsqueeze(labels_dims_number).long()
    
    # # We add one more dim to the end of tensor with the size of 'number_of_classes'
    # one_hot_shape = list(labels.size())
    # one_hot_shape.append(num_classes)
    # one_hot_encoding = torch.zeros(one_hot_shape)
    
    # # Filling out the tensor with ones
    # print(labels_.shape)
    # one_hot_encoding.scatter_(dim=labels_dims_number, index=labels_, value=1)
    
    # one_hot_encoding = one_hot_encoding.squeeze().permute(0,3,1,2)
    # return one_hot_encoding






class SoftDiceLoss(nn.Module):
    def __init__(self):
        super(SoftDiceLoss, self).__init__()

    def forward(self, predictions, targets):
        return 1-dice_score(predictions, targets)




def dice_score(predictions, targets):
    smooth = 1.
    iflat = predictions.view(-1)
    tflat = targets.view(-1)
    intersection = (iflat * tflat).sum()
    
    return (2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth)



def soft_dice_per_batch_2(net_output, gt, smooth=1., smooth_in_nom=1., background_weight=1, rebalance_weights=None):
    if rebalance_weights is not None and len(rebalance_weights) != gt.shape[1]:
        rebalance_weights = rebalance_weights[1:] # this is the case when use_bg=False
    axes = tuple([0] + list(range(2, len(net_output.size()))))
    tp = sum_tensor(net_output * gt, axes, keepdim=False)
    fn = sum_tensor((1 - net_output) * gt, axes, keepdim=False)
    fp = sum_tensor(net_output * (1 - gt), axes, keepdim=False)
    weights = torch.ones(tp.shape)
    weights[0] = background_weight
    if net_output.device.type == "cuda":
        weights = weights.cuda(net_output.device.index)
    if rebalance_weights is not None:
        rebalance_weights = torch.from_numpy(rebalance_weights).float()
        if net_output.device.type == "cuda":
            rebalance_weights = rebalance_weights.cuda(net_output.device.index)
        tp = tp * rebalance_weights
        fn = fn * rebalance_weights
    result = (- ((2 * tp + smooth_in_nom) / (2 * tp + fp + fn + smooth)) * weights).mean()
    return result