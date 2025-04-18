import numpy as np
import pandas as pd
from sklearn.metrics import (roc_auc_score, accuracy_score,
                             average_precision_score)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image, make_grid
from tqdm import tqdm

from augmentations import set_seeds
from meters import AverageMeter

class AugmentOnTest:
    def __init__(self, dataset, n):
        self.dataset = dataset
        self.n = n

    def __len__(self):
        return self.n * len(self.dataset)

    def __getitem__(self, i):
        return self.dataset[i // self.n]

def load_data(dataset, n, num_workers):
    if n != 1:
        dataset = AugmentOnTest(dataset, n)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=n, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        worker_init_fn=set_seeds)

    return dataloader

def get_predictions_and_loss(model, dataloader, device, criterion, save_images):
    losses = AverageMeter()
    predictions = pd.DataFrame(columns=['isic_id', 'true_label', 'prediction'])

    for i, data in enumerate(tqdm(dataloader)):
        (inputs, labels), name, age, gender= data
        if inputs.dim() == 5:
            inputs = inputs.squeeze(0)
            labels = labels.repeat(inputs.shape[0])

        inputs = inputs.to(device)
        labels = labels.to(device)

        if save_images:
            if i <= 10:
                save_image(make_grid(inputs, padding=0),
                           'grid_{}.jpg'.format(i))

        with torch.no_grad():
            outputs = model(inputs)
            scores = F.softmax(outputs, dim=1)[:, 1].cpu().data.numpy()
            loss = criterion(outputs, labels)

        losses.update(loss.item(), inputs.size(0))

        predictions = predictions.append(
            {'isic_id': name[0],
             'prediction': scores.mean(),
             'true_label': labels.data[0].item()},
            ignore_index=True)

    return predictions, losses

def calculate_metrics(predictions, losses):
    labels_array = predictions['true_label'].values.astype(int)
    scores_array = predictions['prediction'].values.astype(float)
    auc = roc_auc_score(labels_array, scores_array)
    acc = accuracy_score(labels_array, np.where(scores_array >= 0.5, 1, 0))
    avp = average_precision_score(labels_array, scores_array)
    return {'loss': losses.avg, 'auc': auc, 'acc': acc, 'avp': avp}

def evaluate(model, dataset, device, num_workers, n, save_images=False):
    assert n >= 1, "n must be larger than 1"

    model.eval()

    criterion = nn.CrossEntropyLoss()
    dataloader = load_data(dataset, n, num_workers)
    predictions, losses = get_predictions_and_loss(model, dataloader, device, criterion, save_images)
    results = calculate_metrics(predictions, losses)
    return (results,predictions)
