from torchvision import datasets, transforms
import torch

# tensor_cifar10 = datasets.CIFAR10(data_path, train=True, download=True,
#                            transform=transforms.ToTensor())
# tensor_cifar10_val = datasets.CIFAR10(data_path, train=False, download=True,
#                                       transform=transforms.Compose(
#                                           [transforms.ToTensor(),
#                                            transforms.Normalize((0.4915, 0.4823, 0.4468),
#                                                                 (0.2470, 0.2435, 0.2616))
#                                            ]))
#


# just to show mean and sd is calculated
def get_mean_and_sd(tensor_cifar10) -> (torch.Tensor, torch.Tensor):
    imgs = torch.stack([img_t for img_t, _ in tensor_cifar10], dim=3)
    # Recall that view(3, -1) keeps the three channels and merges all the
    # remaining dimensions into one, figuring out the appropriate size.
    # Here our 3 × 32 × 32 image is transformed into a 3 × 1,024 vector,
    # and then the mean is taken over the 1,024 elements of each channel.
    return imgs.view(3, -1).mean(dim=1), imgs.view(3, -1).std(dim=1)

def get_cifar10(is_train: bool):
    data_path = 'cifar10/'

    return datasets.CIFAR10(data_path, train=is_train, download=True,
                                          transform=transforms.Compose(
                                              [transforms.ToTensor(),
                                               # copying this from the book
                                               transforms.Normalize((0.4915, 0.4823, 0.4468),
                                                                    (0.2470, 0.2435, 0.2616))
                                               ]))

def get_cifar10_train_valid_dataset():
    return get_cifar10(True), get_cifar10(False)

def get_cifar2_train_test_dataset():
    cifar10_train, cifar10_test = get_cifar10_train_valid_dataset()
    label_map = {0: 0, 2: 1}
    class_names = ['airplane', 'bird']
    cifar2 = [(img, label_map[label])
              for img, label in cifar10_train
              if label in [0, 2]]
    cifar2_val = [(img, label_map[label])
                  for img, label in cifar10_test
                  if label in [0, 2]]
    return cifar2, cifar2_val


# def get_train_valid_dataloader():

