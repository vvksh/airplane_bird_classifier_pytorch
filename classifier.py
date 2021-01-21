import torch
import torch.nn as nn

class AirplaneBirdClassifier:
    def __init__(self):
        self.model = nn.Sequential(
            nn.Linear(3072, 512),
            nn.Tanh(),
            nn.Linear(512, 2),
            nn.LogSoftmax(dim=1)
        )
        self.loss_fn  = nn.NLLLoss()

    def training_loop(self,
                      n_epochs: int,
                      dataset_train,
                      optimizer_class: torch.optim.Optimizer.__class__,
                      learning_rate: float):
        optimizer = optimizer_class(self.model.parameters(), lr=learning_rate)

        for epoch in range(n_epochs):
            for image, label in dataset_train:
                out = self.model(image.view(-1).unsqueeze(0))
                loss = self.loss_fn(out, torch.tensor([label]))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print("Epoch: %d, Loss: %f" % (epoch, float(loss)))




