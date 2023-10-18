import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.data import DataLoader
from tqdm import tqdm


class EWCFineTuner:
    def __init__(self, num_classes, learning_rate=0.001, momentum=0.9,
                 model_name='resnet50'):
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.initialize_model(model_name)
        self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate,
                                   momentum=momentum)
        self.fisher_dict = {}
        self.optpar_dict = {}

    def initialize_model(self, model_name='resnet50'):
        if model_name == 'resnet50':
            return self.get_resnet_model()
        elif model_name == 'customNet':
            return Net(3, self.num_classes)

    def train_with_ewc(self, dataloader, num_epochs, ewc_lambda, task):
        self.model.train()

        for epoch in range(num_epochs):
            for inputs, targets in tqdm(dataloader):
                self.optimizer.zero_grad()
                outputs = self.model(inputs.to(self.device))

                if task == 'multi-label, binary-class':
                    targets = targets.to(torch.float32)
                    criterion = nn.BCEWithLogitsLoss()
                else:
                    targets = targets.squeeze().long()
                    criterion = nn.CrossEntropyLoss()

                loss = criterion(outputs, targets.to(self.device))
                ewc_loss = self.calculate_ewc_loss(ewc_lambda)
                loss += ewc_loss
                loss.backward()
                self.optimizer.step()

        # Update regularization
        for name, param in self.model.named_parameters():
            self.optpar_dict[name] = param.data.clone()
            self.fisher_dict[name] = param.grad.data.clone().pow(2)

    def calculate_ewc_loss(self, ewc_lambda):
        ewc_loss = 0

        for name, param in self.model.named_parameters():
            if name in self.fisher_dict:
                fisher = self.fisher_dict[name]
                optpar = self.optpar_dict[name]
                ewc_loss += (fisher * (optpar - param).pow(2)).sum() * ewc_lambda

        return ewc_loss

    def train_on_datasets(self, datasets, num_epochs, ewc_lambda, eval_dataset,
                          rehearsal=0):
        # Evaluate model before training
        eval_dataloader = DataLoader(eval_dataset.test_dataset, batch_size=32,
                                     shuffle=True)
        print(f"Before training (rehearsal={rehearsal}):")
        self.evaluate_model(eval_dataset.task, eval_dataloader)
        last_dataset = None

        for i, dataset in enumerate(datasets):
            print(f'\n[{i}] train on {dataset.data_flag}')

            # If rehearsal>0, carry over part of the previous dataset
            if rehearsal>0 and last_dataset is not None:
                imgs = last_dataset.train_dataset.imgs
                labels = last_dataset.train_dataset.labels
                n_samples = len(imgs)
                random_indices = np.random.choice(
                    n_samples, int(n_samples*rehearsal), replace=False)
                imgs_subset = imgs[random_indices]
                labels_subset = labels[random_indices]
                dataset.train_dataset.imgs = np.concatenate(
                    (dataset.train_dataset.imgs, imgs_subset), axis=0)
                dataset.train_dataset.labels = np.concatenate(
                    (dataset.train_dataset.labels, labels_subset), axis=0)

            dataloader = DataLoader(dataset.train_dataset, batch_size=32, shuffle=True)
            self.train_with_ewc(dataloader, num_epochs, ewc_lambda, dataset.task)
            self.evaluate_model(eval_dataset.task, eval_dataloader)
            last_dataset = dataset

    def evaluate_model(self, task, dataloader):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in tqdm(dataloader):

                if task == 'multi-label, binary-class':
                    labels = labels.to(torch.float32)
                else:
                    labels = labels.squeeze().long()

                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = correct / total
        print(f"Accuracy on {dataloader.dataset.flag}: {correct}/{total} = {accuracy}")
        return accuracy

    def save_model(self, filename):
        torch.save(self.model.state_dict(), filename)

    def get_resnet_model(self):
        model = models.resnet50(weights=False)
        # Remove the final classification layer
        # (usually with 1000 classes for ImageNet)
        num_features = model.fc.in_features
        model = nn.Sequential(
            *list(model.children())[:-2])  # Remove the final two layers

        # Add a new custom classification layer for your task
        custom_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Global average pooling
            nn.Flatten(),
            nn.Linear(num_features, self.num_classes)
            # Output layer with num_classes units
        )

        # Combine the pretrained ResNet-50 base with the custom classifier
        model = nn.Sequential(model, custom_classifier)
        model.to(self.device)
        return model


class Net(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(Net, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU())

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer3 = nn.Sequential(
            nn.Conv2d(16, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU())

        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU())

        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.fc = nn.Sequential(
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
