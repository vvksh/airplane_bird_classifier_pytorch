import data_utils
from classifier import AirplaneBirdClassifier
import torch
from torch.utils.data.dataloader import DataLoader

def run_classifier():
    cifar2_train_dataset, cifar2_val_dataset = data_utils.get_cifar2_train_test_dataset()
    cifar2_train_dataloader = DataLoader(dataset=cifar2_train_dataset,
                                         batch_size=64,
                                         shuffle=True)
    cifar2_val_loader = torch.utils.data.DataLoader(dataset=cifar2_val_dataset,
                                                    batch_size=64,
                                                    shuffle=False)

    classifier = AirplaneBirdClassifier()
    classifier.training_loop(n_epochs=100,
                             train_dataloader=cifar2_train_dataloader,
                             val_dataloader=cifar2_val_loader,
                             optimizer_class=torch.optim.SGD,
                             learning_rate=1e-2)

if __name__ == '__main__':
    run_classifier()