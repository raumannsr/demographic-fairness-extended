from itertools import islice
import os
import pandas as pd
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.utils import save_image
from tqdm import tqdm
from networks import EnsembleNetwork
from augmentations import Augmentations
from augmentations import set_seeds
from dataset_loader import CSVDataset, CSVDatasetWithMeta, CSVDatasetWithAllLabels
from meters import AverageMeter
from test import evaluate
from torch.autograd import Variable
import dcor
from environment_variables import EnvVars

# region Configuration
env = EnvVars.getInstance()
train_val_test_path = env.get_env_var('train_val_test_path')
train_root = env.get_env_var('image_path')
train_csv = train_val_test_path + '/' + env.get_env_var('train_file')
val_root = env.get_env_var('image_path')
val_csv = train_val_test_path + '/' + env.get_env_var('validation_file') 
test_root = env.get_env_var('image_path')
test_csv = train_val_test_path + '/' + env.get_env_var('test_file')
epochs = int(env.get_env_var('num_epochs'))
aug = {
    'hflip': True,  # Random Horizontal Flip
    'vflip': True,  # Random Vertical Flip
    'rotation': 90,  # Rotation (in degrees)
    'shear': 20,  # Shear (in degrees)
    'scale': 1.0,  # Scale (tuple (min, max))
    'color_contrast': 0.3,  # Color Jitter: Contrast
    'color_saturation': 0.3,  # Color Jitter: Saturation
    'color_brightness': 0.3,  # Color Jitter: Brightness
    'color_hue': 0,  # Color Jitter: Hue
    'random_crop': False,  # Random Crops
    'random_erasing': False,  # Random Erasing
    'piecewise_affine': False,  # Piecewise Affine
    'tps': False,  # TPS Affine
    'autoaugment': False # AutoAugmentation
}
batch_size=int(env.get_env_var('batch_size'))
num_workers = 8 
val_samples = int(env.get_env_var('val_steps')) 
test_samples=int(env.get_env_var('pred_steps'))
early_stopping_patience = 10 
limit_data = False
images_per_epoch = 2000
lamda2 = int(env.get_env_var('lamda')) 
BEST_MODEL_PATH = '../2_pipeline/best-model'
LAST_MODEL_PATH = '../2_pipeline/last-model'
METRICS_PATH = '../2_pipeline/metrics'
RESULTS_PATH = '../2_pipeline/results'
# endregion

# region Helper Functions
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def m_correlation_coefficient_loss(x, y):
    if y.dim() == 1:
        y=y.unsqueeze(dim=1)
    if x.dim() == 1:
        x=x.unsqueeze(dim=1)
    mean_x = torch.mean(x,0)
    mean_y = torch.mean(y,0)
    xm = x.sub(mean_x)
    ym = y.sub(mean_y)
    num_bias = xm.shape[1]
    r_val_sq = torch.zeros([num_bias],dtype=torch.float32)
    for k in range(num_bias):
        r_num = xm[:,k].dot(ym[:,k])
        r_den = torch.norm(xm[:,k], 2) * torch.norm(ym[:,k], 2)
        r_val = r_num / (r_den+1e-5)
        r_val = torch.clamp(r_val, min=-1, max=1)
        r_val_sq[k] = r_val**2
    return -r_val_sq.sum()
# endregion

# region Training Functions
def train_epoch(
                device,
                dataloaders,
                encoder,
                classifier,
                distiller_age,
                criterion_classification_task,
                criterion_age_bias_predictor,
                optimizer_classification_task, 
                optimizer_age_bias_predictor, 
                optimizer_Enc,
                lamda2,
                batches_per_epoch=None):
    
    losses_cl = AverageMeter()
    losses_bp_g = AverageMeter()
    losses_br = AverageMeter()
    accuracies = AverageMeter()
    dcs = AverageMeter() # to measure the distance correlation sqr
    all_preds = []
    all_labels = []
    classifier.train()
    distiller_age.train()
    #============ Prepare the data loaders ==========#
    if batches_per_epoch:
        tqdm_loader = tqdm(
            islice(dataloaders['train'], 0, batches_per_epoch),
            total=batches_per_epoch)
    else:
        tqdm_loader = tqdm(dataloaders['train'])

    ctrl_dataloader_iterator = iter(dataloaders['train_ctrl'])
    # ============ Prepare the ctrl data loaders ==========#
    for data in tqdm_loader:
        # ---------------------
        # Select a subset of normal data
        # ---------------------
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        # ---------------------
        #  Select a batch of ctrl data
        # ---------------------
        try:
            (inputs_ctrl, labels_ctrl) , age_ctrl, _ = next(ctrl_dataloader_iterator)
        except:
            ctrl_dataloader_iterator = iter(dataloaders['train_ctrl'])
            (inputs_ctrl, labels_ctrl) , age_ctrl, _ = next(ctrl_dataloader_iterator)
        
        age_ctrl = age_ctrl / 85.0 # normalize the age
        bias_labels = age_ctrl.unsqueeze(-1)
        
        inputs_ctrl = inputs_ctrl.to(device)
        bias_labels = bias_labels.to(device)


        # ---------------------
        #  Train Encoder & Classifier (actual classification task) with normal data
        # ---------------------
        # update theta_c
        optimizer_classification_task.zero_grad()
        for param in classifier.decoder.parameters():
            param.requires_grad = True
        for param in encoder.parameters():
            param.requires_grad = False
        outputs = classifier(inputs)
        _, preds = torch.max(outputs.data, 1)
        loss_cl = criterion_classification_task(outputs, labels)

        loss_cl.backward()
        optimizer_classification_task.step() # it only updates the classifier head

        # ---------------------
        #  Train regressor_age (bias predictor) with ctrl data
        # ---------------------
        optimizer_age_bias_predictor.zero_grad()
        # update theta_bp (encoder fixed)
        outputs_age = distiller_age(inputs_ctrl)
        loss_bp_age = criterion_age_bias_predictor(outputs_age, bias_labels)
        loss_bp_age = Variable(loss_bp_age, requires_grad=True)

        loss_bp_age.backward()
        optimizer_age_bias_predictor.step()

        # ---------------------
        #  Train Disstiller (bias removal)
        # ---------------------
        optimizer_Enc.zero_grad()
        for param in distiller_age.decoder.parameters():
            param.requires_grad = False

        # For the distillation model (removing bias) we will only train the encoder adversarially
        # update theta_fe (theta_bp fixed)
        for param in encoder.parameters():
            param.requires_grad = True

        lamda=lamda2
        outputs_bias = distiller_age(inputs_ctrl)
        loss_br = criterion_bias_removal(outputs_age, bias_labels)
        outputs_age = distiller_age(inputs_ctrl)
        loss_age_bias_predictor = criterion_classification_task(outputs_age, bias_labels)
        loss_br = Variable(loss_cl - lamda * loss_br - lamda2 * loss_age_bias_predictor, requires_grad=True)

        loss_br.backward()
        optimizer_Enc.step()
    
        # set trainable parameters again
        for param in distiller_age.parameters():
            param.requires_grad = True

        losses_cl.update(loss_cl.item(), inputs.size(0))
        losses_bp_g.update(loss_bp_age.item(), inputs_ctrl.size(0))
        losses_br.update(loss_br.item(), inputs_ctrl.size(0))

        acc = torch.sum(preds == labels.data).item() / preds.shape[0]
        accuracies.update(acc)
        all_preds += list(F.softmax(outputs, dim=1)[:, 1].cpu().data.numpy())
        all_labels += list(labels.cpu().data.numpy())

        feature = classifier.enc_feat(inputs_ctrl)
        feature = feature.cpu().data.numpy()
        cf_labels = bias_labels.cpu().data.numpy()
        dc = dcor.u_distance_correlation_sqr(feature, cf_labels)
        dcs.update(dc, inputs_ctrl.size(0))

        tqdm_loader.set_postfix(loss=losses_cl.avg, acc=accuracies.avg, loss_bp_age=losses_bp_g.avg, loss_br=losses_br.avg)#,dc = dcs.avg)

    auc = roc_auc_score(all_labels, all_preds)

    return {'loss': losses_cl.avg, 'auc': auc, 'acc': accuracies.avg, 'loss_bp_age':losses_bp_g.avg , 'loss_br':losses_br.avg, 'dcs':dc}
# endregion

# region Model Setup
def save_images(dataset, to, n=32):
    for i in range(n):
        img_path = os.path.join(to, 'img_{}.png'.format(i))
        save_image(dataset[i][0], img_path)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet50(pretrained=True)
encoder = nn.Sequential(
    model.conv1,
    model.bn1,
    model.relu,
    model.maxpool,
    model.layer1,
    model.layer2,
    model.layer3,
    model.layer4)
in_features = model.fc.in_features

decoder_age=nn.Sequential(
    nn.Linear(in_features, 8),
    nn.Tanh(),
    nn.Linear(8, 1))
decoder_cl = nn.Sequential(
    nn.Linear(in_features, 8),
    nn.Tanh(),
    nn.Linear(8, 2))
avg_pool = nn.AdaptiveAvgPool2d((1, 1))

classifier = EnsembleNetwork(encoder=encoder,decoder=decoder_cl,avg_pool=avg_pool)
distiller_age = EnsembleNetwork(encoder=encoder,decoder=decoder_age,avg_pool=avg_pool)

aug['size'] = 224
aug['mean'] = [0.485, 0.456, 0.406]
aug['std'] = [0.229, 0.224, 0.225]

print("classifier parameters: ", count_parameters(classifier))
print("encoder parameters: ", count_parameters(encoder))
print("decoder gender parameters: ", count_parameters(decoder_age))
print("decoder classifier parameters: ", count_parameters(decoder_cl))

classifier.to(device)
distiller_age.to(device)
# endregion

# region Data Loading
augs = Augmentations(**aug)
classifier.aug_params = aug

datasets = {
    'train': CSVDataset(train_root, train_csv, 'isic_id', 'target',
                        transform=augs.tf_transform, add_extension='.JPG',
                        random_subset_size=limit_data,remove_unknowns=True, ctrl=False),
    'train_ctrl': CSVDatasetWithAllLabels(train_root, train_csv, 'isic_id', 'target',
                                        transform=augs.tf_transform, add_extension='.JPG',
                                        random_subset_size=limit_data, remove_unknowns=True,ctrl=False),
    'val': CSVDatasetWithMeta(
        val_root, val_csv, 'isic_id', 'target',
        transform=augs.no_augmentation, add_extension='.JPG'),
    'test': CSVDatasetWithMeta(
        test_root, test_csv, 'isic_id', 'target',
        transform=augs.no_augmentation, add_extension='.JPG',remove_unknowns=True),
}

dataloaders = {
    'train': DataLoader(datasets['train'], batch_size=batch_size,
                        shuffle=True, num_workers=num_workers,
                        worker_init_fn=set_seeds,drop_last=True),
    'train_ctrl': DataLoader(datasets['train_ctrl'], batch_size=batch_size,
                        shuffle=True, num_workers=num_workers,
                        worker_init_fn=set_seeds,drop_last=True)
}
# endregion

# region Loss Functions and Optimizers
criterion_classification_task = nn.CrossEntropyLoss()
criterion_bias_removal = m_correlation_coefficient_loss
criterion_age_bias_predictor = m_correlation_coefficient_loss
optimizer_classification_task = optim.Adam(classifier.decoder.parameters(),lr=0.001)
optimizer_age_bias_predictor = optim.Adam(distiller_age.decoder.parameters(), lr=0.0001)
optimizer_Enc = optim.Adam(encoder.parameters(),lr=0.001)

scheduler = optim.lr_scheduler.MultiStepLR(optimizer_classification_task,
                                            milestones=[10],
                                            gamma=0.1)
scheduler_BP_age = optim.lr_scheduler.MultiStepLR(optimizer_age_bias_predictor,
                                                milestones=[10],
                                                gamma=0.1)
scheduler_Enc = optim.lr_scheduler.MultiStepLR(optimizer_Enc,
                                                milestones=[10],
                                                gamma=0.1)
# endregion

# region Training Loop
metrics = {
    'train': pd.DataFrame(columns=['epoch', 'loss', 'acc', 'auc']),
    'val': pd.DataFrame(columns=['epoch', 'loss', 'acc', 'auc'])
}

best_val_auc = 0.0
best_epoch = 0
epochs_without_improvement = 0
if images_per_epoch:
    batches_per_epoch = images_per_epoch // batch_size
else:
    batches_per_epoch = None

for epoch in range(epochs):
    print('train epoch {}/{}'.format(epoch+1, epochs))
    # train the Classifier & update theta_fe & theta_c
    epoch_train_result= train_epoch(
        device, dataloaders,
        encoder, classifier,distiller_age,
        criterion_classification_task, criterion_age_bias_predictor,
        optimizer_classification_task,optimizer_age_bias_predictor,optimizer_Enc,lamda2,
        batches_per_epoch)

    metrics['train'] = metrics['train'].append(
        {**epoch_train_result, 'epoch': epoch}, ignore_index=True)
    print('train', epoch_train_result)


    epoch_val_result, _ = evaluate(
        classifier, datasets['val'], device, num_workers, val_samples)

    metrics['val'] = metrics['val'].append(
        {**epoch_val_result, 'epoch': epoch}, ignore_index=True)
    print('val', epoch_val_result)
    print('-' * 40)

    scheduler.step()
    scheduler_BP_age.step()
    scheduler_Enc.step()

    if epoch_val_result['auc'] > best_val_auc:
        best_val_auc = epoch_val_result['auc']
        best_epoch = epoch
        epochs_without_improvement = 0
        torch.save(classifier, BEST_MODEL_PATH+'/model_best.pth')
        path = env.get_env_var('experiment_path') + '/' + env.get_env_var('experiment_id') + '/'
        weights_filename = path + env.get_env_var('model_weights_file')
        torch.save(classifier, weights_filename)
    else:
        epochs_without_improvement += 1

    if epochs_without_improvement > early_stopping_patience:
        torch.save(classifier, LAST_MODEL_PATH+'/model_last.pth')
        break

    if epoch == (epochs-1):
        torch.save(classifier, LAST_MODEL_PATH+'/model_last.pth')
# endregion

# region Save the metrics to disk
for phase in ['train', 'val']:
    metrics[phase].epoch = metrics[phase].epoch.astype(int)
    metrics[phase].to_csv(os.path.join(METRICS_PATH, phase + '.csv'),
                            index=False)
    
# Run testing
test_result, test_preds = evaluate(
    torch.load(BEST_MODEL_PATH+'/model_best.pth'), datasets['test'], device,
    num_workers, test_samples)
print('test', test_result)

path = env.get_env_var('experiment_path') + '/' + env.get_env_var('experiment_id') + '/'

""" Save all predictions to csv files """
test_preds.to_csv(path + env.get_env_var('prediction_file'), index=False)

""" Save all results in a csv file """
#{'loss': losses.avg, 'auc': auc, 'acc': acc, 'avp': avp}

df = pd.DataFrame(test_result, index=[0])
#df = pd.DataFrame.from_dict(test_result, orient="index")
df.to_csv(path + env.get_env_var('model_performance_file'), index=False)
# endregion