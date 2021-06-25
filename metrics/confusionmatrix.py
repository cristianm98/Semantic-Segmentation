import numpy as np
import torch


def validate_pred(predicted, num_classes):
    if np.ndim(predicted) != 1:
        assert predicted.shape[1] == num_classes, \
            '[ConfusionMatrix]: Number of classes from the prediction batch is different than the actual number of classes'
        predicted = np.argmax(predicted, 1)
    else:
        assert (predicted.max() < num_classes) and (predicted.min() >= 0), \
            '[ConfusionMatrix]: Predicted values are not in range [0, num_classes - 1]'
    return predicted


def validate_target(target, num_classes):
    if np.ndim(target) != 1:
        assert target.shape[1] == num_classes, \
            '[ConfusionMatrix]: Number of classes from the target batch is different than the actual number of classes'
        assert (target >= 0).all() and (target <= 1).all(), \
            '[ConfusionMatrix]: One hot encoded values should be in range [0, 1]'
        assert (target.sum(1) == 1).all(), \
            '[ConfusionMatrix]: Multi-label target is not supported'
        target = np.argmax(target, 1)
    else:
        assert (target.max() < num_classes) and (target.min() >= 0), \
            '[ConfusionMatrix]: Target values are not in range [0, num_classes - 1]'
    return target


class ConfusionMatrix:
    def __init__(self, num_classes):
        self.conf_matrix = np.ndarray((num_classes, num_classes), dtype=np.int64)
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.conf_matrix.fill(0)

    def add(self, predicted, target):
        if torch.is_tensor(predicted):
            predicted = predicted.cpu().numpy()
        if torch.is_tensor(target):
            target = target.cpu().numpy()
        assert predicted.shape[0] == target.shape[0], \
            '[ConfusionMatrix]: Different batch sizes for predicted and target'
        predicted = validate_pred(predicted, self.num_classes)
        target = validate_target(target, self.num_classes)
        x = predicted + self.num_classes * target
        bincount_2d = np.bincount(
            x.astype(np.int64), minlength=self.num_classes ** 2)
        assert bincount_2d.size == self.num_classes ** 2
        conf_matrix = bincount_2d.reshape((self.num_classes, self.num_classes))
        self.conf_matrix += conf_matrix

    def value(self):
        return self.conf_matrix
