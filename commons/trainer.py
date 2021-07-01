from tqdm.auto import tqdm


class Trainer:
    def __init__(self, model, data_loader, optimizer, criterion, metric, device):
        super(Trainer, self).__init__()
        self.model = model
        self.data_loader = data_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.metric = metric
        self.device = device

    def run_epoch(self, use_tqdm=True):
        train_data = tqdm(self.data_loader) if use_tqdm else self.data_loader
        if self.device.type == 'cuda':
            self.model.cuda()
        self.model.train()
        epoch_loss = 0.0
        self.metric.reset()
        for idx, batch in enumerate(train_data):
            data = batch[0].to(self.device)
            target = batch[1].to(self.device)
            if self.model.__class__.__name__.lower() == 'fcn':
                pred = self.model(data)['out']
            else:
                pred = self.model(data)
            loss = self.criterion(pred, target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()
            self.metric.add(pred.detach(), target.detach())
        return epoch_loss / len(self.data_loader), self.metric.value()
