import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.models import ResNet50_Weights
from tqdm import tqdm
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


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
        self.trained_tasks = []

    def initialize_model(self, model_name='resnet50'):
        if model_name == 'resnet50':
            return self.get_resnet_model()
        elif model_name == 'customNet':
            return Net(3, self.num_classes)

    def train_with_ewc(self, dataloader, num_epochs, ewc_lambda):
        self.model.train()
        task_type = dataloader.dataset.info['task']

        for epoch in range(num_epochs):
            for inputs, targets in tqdm(dataloader):
                self.optimizer.zero_grad()
                outputs = self.model(inputs.to(self.device))

                if task_type == 'multi-label, binary-class':
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

        task_id = dataloader.dataset.flag
        self.optpar_dict[task_id] = {}
        self.fisher_dict[task_id] = {}
        self.trained_tasks.append(task_id)

        # Update regularization params
        for name, param in self.model.named_parameters():
            self.optpar_dict[task_id][name] = param.data.clone()
            self.fisher_dict[task_id][name] = param.grad.data.clone().pow(2)

    def calculate_ewc_loss(self, ewc_lambda):
        ewc_loss = 0

        for id in self.trained_tasks:
            for name, param in self.model.named_parameters():
                fisher = self.fisher_dict[id][name]
                optpar = self.optpar_dict[id][name]
                ewc_loss += \
                    (fisher * (optpar - param).pow(2)).sum() * ewc_lambda

        return ewc_loss

    def train_on_datasets(self, datasets, num_epochs, ewc_lambda, eval_dataset,
                          rehearsal=0):
        # Evaluate model before training
        eval_dataloader = DataLoader(eval_dataset.test_dataset, batch_size=32,
                                     shuffle=True)
        print(f"Before training (rehearsal={rehearsal}):")
        self.evaluate_model(eval_dataloader)
        last_dataset = None

        for i, dataset in enumerate(datasets):
            print(f'\n[{i}] train on {dataset.data_flag}')

            # If rehearsal>0, carry over part of the previous dataset
            if rehearsal>0 and last_dataset is not None:
                self.add_data_from_previous_task(
                    dataset, last_dataset, rehearsal)

            dataloader = DataLoader(
                dataset.train_dataset, batch_size=32, shuffle=True)
            self.train_with_ewc(dataloader, num_epochs, ewc_lambda)
            self.evaluate_model(eval_dataloader)
            last_dataset = dataset

    def add_data_from_previous_task(self, dataset, last_dataset, rehearsal):
        imgs = last_dataset.train_dataset.imgs
        labels = last_dataset.train_dataset.labels
        n_samples = len(imgs)
        # Randomly sample a subset of the data
        random_idx = np.random.choice(
            n_samples, int(n_samples * rehearsal), replace=False)
        print(f"imgs before {dataset.train_dataset.imgs.shape}, "
              f"labels before {dataset.train_dataset.labels.shape}")
        # Concatenate the subset to your dataset
        dataset.train_dataset.imgs = np.concatenate(
            (dataset.train_dataset.imgs, imgs[random_idx]), axis=0)
        dataset.train_dataset.labels = np.concatenate(
            (dataset.train_dataset.labels, labels[random_idx]), axis=0)
        print(
            f"size prev imgs {imgs.shape}, labels: {labels.shape}, n_samples {n_samples}")
        print(
            f"img subset {imgs[random_idx].shape}, labels subset {labels[random_idx].shape}")
        print(f"imgs {dataset.train_dataset.imgs.shape}, "
              f"labels {dataset.train_dataset.labels.shape}")

    def evaluate_model(self, dataloader):
        self.model.eval()
        correct = 0
        total = 0
        predicted_all = []
        task_type = dataloader.dataset.info['task']

        with torch.no_grad():
            for inputs, labels in tqdm(dataloader):

                if task_type == 'multi-label, binary-class':
                    labels = labels.to(torch.float32)
                else:
                    labels = labels.squeeze().long()

                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)

                total += labels.size(0)
                predicted_all.append(predicted)
                correct += (predicted == labels).sum().item()

        if task_type == 'multi-label, binary-class':
            labels = dataloader.dataset.labels
        else:
            labels = dataloader.dataset.labels.squeeze()

        accuracy = correct / total
        predicted_all = np.concatenate(predicted_all)
        ari = adjusted_rand_score(labels, predicted_all)
        nmi = normalized_mutual_info_score(labels, predicted_all)

        print(f"Accuracy on {dataloader.dataset.flag}: {correct}/{total} = {accuracy}")
        print(f"ARI on {dataloader.dataset.flag}: {ari}")
        print(f"NMI on {dataloader.dataset.flag}: {nmi}")
        return accuracy

    def save_model(self, filename):
        torch.save(self.model.state_dict(), filename)

    def get_resnet_model(self):
        model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
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
