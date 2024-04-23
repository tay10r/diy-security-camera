#!venv/bin/python3

import torch
import torch.utils.data
import torchvision.transforms.v2 as transforms
import os
from loguru import logger
import json
from model import PersonDetector
from dataset import Dataset
from dataclasses import dataclass, asdict

@dataclass
class Config:
    label_path: str = 'data/labels.json'
    model_path: str = 'data/model.pt'
    train_data_path: str = 'data/train'
    validation_data_path: str = 'data/validation'
    batch_size: int = 16
    device: str = 'cuda:0'
    learning_rate: float = 0.001
    model_path: str = 'data/model.pt'
    shuffle_data: bool = True
    num_epochs = 1000

def open_labels(path: str) -> dict:
    label_dict = {}
    with open(path, 'r') as labels_file:
        label_dict = json.load(labels_file)
    return label_dict

class Program:
    def __init__(self, config_path: str = 'train/train_config.json'):
        self.__config = Config()
        if os.path.exists(config_path):
            logger.info('Loading configuration file.')
            with open(config_path, 'r') as config_file:
                self.__config = Config(**json.load(config_file))
        else:
            logger.info('Creating default configuration file.')
            with open(config_path, 'w') as config_file:
                json.dump(asdict(self.__config), config_file, indent='  ')

        self.__labels = open_labels(self.__config.label_path)
        self.__transform = transforms.Compose([
            transforms.Resize((240, 320)),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.ColorJitter(brightness=(0.2, 2.0), saturation=(0.2, 2.0), contrast=(0.2, 2.0)),
            transforms.RandomHorizontalFlip(),
            transforms.GaussianBlur(kernel_size=3)
        ])
        self.__train_data = Dataset(self.__config.train_data_path, self.__labels, self.__transform)
        self.__validation_data = Dataset(self.__config.validation_data_path, self.__labels, self.__transform)
        self.__device = torch.device(self.__config.device)

    def open_model(self) -> PersonDetector:
        model = PersonDetector()
        if os.path.exists(self.__config.model_path):
            logger.info('Opening existing model.')
            model.load_state_dict(torch.load(self.__config.model_path))
        return model.to(self.__device)

    def save_model(self, model: PersonDetector):
        model.cpu().eval()
        torch.save(model.state_dict(), self.__config.model_path)

    def compute_validation_loss(self, model) -> float:
        loader = torch.utils.data.DataLoader(self.__validation_data,
                                             self.__config.batch_size,
                                             shuffle=True)
        model.eval()
        criterion = torch.nn.BCELoss()
        val_loss = 0.0
        for data in loader:
            frames, labels = data
            frames = frames.to(self.__device)
            labels = labels.to(self.__device)
            pred = model(frames)
            loss = criterion(pred, labels)
            val_loss += loss.item()
        avg_val_loss = val_loss / len(loader)
        return avg_val_loss

    def run(self):
        loader = torch.utils.data.DataLoader(self.__train_data,
                                             self.__config.batch_size,
                                             shuffle=self.__config.shuffle_data)
        model = self.open_model()
        criterion = torch.nn.BCELoss()
        optimizer = torch.optim.SGD(params=model.parameters(), lr=self.__config.learning_rate)
        for _ in range(self.__config.num_epochs):
            model = model.to(self.__device)
            model.train()
            training_loss = 0.0
            for data in loader:
                frames, labels = data
                frames = frames.to(self.__device)
                labels = labels.to(self.__device)
                optimizer.zero_grad()
                pred = model(frames)
                loss = criterion(pred, labels)
                training_loss += loss.item()
                loss.backward()
                optimizer.step()
            val_loss = self.compute_validation_loss(model)
            #self.save_model(model)
            training_loss /= len(loader)
            logger.info(f'Epoch done, training_loss = {training_loss}, validation loss = {val_loss}')

def main():
    program = Program()
    program.run()

main()
