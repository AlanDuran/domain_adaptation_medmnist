import random

import numpy as np

from proyecto.ewc_fine_tuner import EWCFineTuner
from proyecto.medmnist_data import MedMnistData

if __name__ == "__main__":

    num_classes = 6
    learning_rate = 0.001
    momentum = 0.9
    num_epochs = 3
    ewc_lambda = 0
    rehearsal = 0
    base_model = 'resnet50'
    preprocess_resnet = True

    # Load and preprocess your datasets
    dataset_names = ['PathMNIST', 'DermaMNIST', 'BloodMNIST', 'TissueMNIST',
                     'OrganCMNIST', 'OrganAMNIST', 'OrganSMNIST']
    random.Random(42).shuffle(dataset_names)
    datasets = []
    smaller_size = 1e6
    for name in dataset_names:
        dataset = MedMnistData(name.lower())
        dataset.select_n_classes(num_classes)
        dataset_size = len(dataset.train_dataset.imgs)
        smaller_size = dataset_size if dataset_size < smaller_size else smaller_size
        datasets.append(dataset)

    for dataset in datasets:
        imgs = dataset.train_dataset.imgs
        labels = dataset.train_dataset.labels
        n_samples = len(imgs)
        # Randomly sample a subset of the data
        random_idx = np.random.choice(n_samples, smaller_size, replace=False)
        # Concatenate the subset to your dataset
        dataset.train_dataset.imgs = imgs[random_idx]
        dataset.train_dataset.labels = labels[random_idx]

    # Initialize the class
    fine_tuner = EWCFineTuner(num_classes, learning_rate, momentum, base_model)

    # Train on the datasets with EWC
    fine_tuner.train_on_datasets(datasets=datasets[:-1],
                                 num_epochs=num_epochs,
                                 ewc_lambda=ewc_lambda,
                                 eval_dataset=datasets[-1],
                                 rehearsal=rehearsal)
