import torch
from tqdm.auto import tqdm


class Tester:
    def __init__(self, model, data_loader, criterion, metric, device):
        self.model = model
        self.data_loader = data_loader
        self.criterion = criterion
        self.metric = metric
        self.device = device

    def run_epoch(self, use_tqdm=True):
        loop = tqdm(self.data_loader) if use_tqdm else self.data_loader
        if self.device.type == 'cuda':
            self.model.cuda()
        self.model.eval()
        epoch_loss = 0.0
        self.metric.reset()
        for idx, batch in enumerate(loop):
            data = batch[0].to(self.device)
            target = batch[1].to(self.device)
            with torch.no_grad():
                pred = self.model(data)
                loss = self.criterion(pred, target)
            epoch_loss += loss.item()
            self.metric.add(pred.detach(), target.detach())
        return epoch_loss / len(self.data_loader), self.metric.value()
