import data_utils
from classifier import AirplaneBirdClassifier
import torch

def run_classifier():
    cifar2_train, _ = data_utils.get_cifar2_train_test_dataset()
    classifier = AirplaneBirdClassifier()
    classifier.training_loop(n_epochs=100,
                             dataset_train=cifar2_train,
                             optimizer_class=torch.optim.SGD,
                             learning_rate=1e-2)

if __name__ == '__main__':
    run_classifier()