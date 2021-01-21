import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader

class AirplaneBirdClassifier:
    def __init__(self):
        self.model = nn.Sequential(
            nn.Linear(3072, 512),
            nn.Tanh(),
            nn.Linear(512, 2),
            nn.LogSoftmax(dim=1)
        )
        self.loss_fn  = nn.NLLLoss()

    def _get_accuracy(self, val_dataloader):
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, labels in val_dataloader:
                batch_size = imgs.shape[0]
                outputs = self.model(imgs.view(batch_size, -1))
                _, predicted = torch.max(outputs, dim=1)
                total += labels.shape[0]
                correct += int((predicted == labels).sum())

        return correct / total


    def training_loop(self,
                      n_epochs: int,
                      train_dataloader: DataLoader,
                      val_dataloader: DataLoader,
                      optimizer_class: torch.optim.Optimizer.__class__,
                      learning_rate: float):
        optimizer = optimizer_class(self.model.parameters(), lr=learning_rate)

        for epoch in range(n_epochs):
            for imgs, labels in train_dataloader:
                batch_size = imgs.shape[0]
                outputs = self.model(imgs.view(batch_size, -1))
                loss = self.loss_fn(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print(f"Epoch: {epoch}, Loss: {float(loss)}, Accuracy {self._get_accuracy(val_dataloader)}")




