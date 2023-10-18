import random

from proyecto.ewc_fine_tuner import EWCFineTuner
from proyecto.medmnist_data import MedMnistData

if __name__ == "__main__":

    num_classes = 6
    learning_rate = 0.01
    momentum = 0.9
    num_epochs = 1
    ewc_lambda = 0.4

    # Initialize the class
    fine_tuner = EWCFineTuner(num_classes, learning_rate, momentum,
                              'customNet')

    # Load and preprocess your datasets
    dataset_names = ['PathMNIST', 'DermaMNIST', 'BloodMNIST', 'TissueMNIST',
                     'OrganCMNIST', 'OrganAMNIST', 'OrganSMNIST']
    random.shuffle(dataset_names)
    datasets = []
    for name in dataset_names:
        dataset = MedMnistData(name.lower())
        dataset.select_n_classes(num_classes)
        datasets.append(dataset)

    # Train on the datasets with EWC
    fine_tuner.train_on_datasets(datasets=datasets[:-1],
                                 num_epochs=num_epochs,
                                 ewc_lambda=ewc_lambda,
                                 eval_dataset=datasets[-1],
                                 rehearsal=0.2)
