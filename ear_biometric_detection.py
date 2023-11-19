## Load useful packages
import py7zr
from zipfile import ZipFile
from random import sample
import PIL.Image as Image
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, ShuffleSplit, StratifiedKFold, KFold 
# KFold is added by Atif
from sklearn.utils import shuffle
import os
import h5py
import numpy as np
import wget
from zipfile import ZipFile
# for noteable.io only
import shutil

#for pytorch and DL models
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn
import torch.nn.functional
import torch.optim
from torchvision import models # import models just for debugging
# from torchvision.transforms import v2 as transforms_v2
from torchvision import transforms as transforms_v2
from spc import SupervisedContrastiveLoss

# from data_augmentation.auto_augment import AutoAugment
# from data_augmentation.duplicate_sample_transform import DuplicateSampleTransform

import sys
import argparse

def parse_options():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--dataset', type=str, default='AMI_dataset',
                        choices=['AMI', 'IITD'], help='dataset')
    parser.add_argument('--target_size', type=str, default='(246, 351)', help='target size in the form of str tuple')
    parser.add_argument('--num_filters', type=int, default=8,
                        help='number of filters')
    parser.add_argument('--model_type', type=str, default='Encoder+Classifier',
                        choices=['Encoder+Classifier', 'DeepLSE', 'Classifier', 'AutoEncoder'], 						help='model_type')
    parser.add_argument('--train_device', type=str, default='cuda:0',
                        choices=['cpu', 'cuda:0', 'cuda:1', 'cuda:2', 'cuda:3'], 
                        help='train device')
    parser.add_argument('--conv_type', type=str, default='conventional',
                        choices=['conventional', 'deformable'], help='convolution type')
    parser.add_argument('--loss_fn_type', type=str, default='contrastive',
                        choices=['contrastive', 'conventional'], help='loss function type')
    parser.add_argument("--transformation", default=True, type=bool)
    parser.add_argument("--auto-augmentation", default=True, type=bool)
    parser.add_argument('--lambda1', type=float, default=0.5,
                        help='lambda1')
    parser.add_argument('--lambda2', type=float, default=0.5,
                        help='lambda2')                  
    parser.add_argument('--trails', type=int, default=5,
                        help='number of trails')                  
    parser.add_argument('--folds', type=int, default=7,
                        help='number of trails')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='number of training epochs')

    

    options = parser.parse_args()
    return options


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

  # Downloading the required files from GitHub repository 5y3datif/Ear_Biometrics
  #filename = 'utilities.py'
  #data_path = 'https://raw.githubusercontent.com/5y3datif/Ear_Biometrics/main/utilities.py'
  #filename = 'utilities.py'
  
  #if(os.path.exists(filename)):
  #  os.remove(filename)
  #  print('existing file:', filename, ' has been deleted')
    
  #print('downloading latest version of file:', filename)
  #wget.download(data_path, filename)
  #print('sucessfully downloaded')

  #data_path ='https://raw.githubusercontent.com/5y3datif/Ear_Biometrics/main/custom_models.py'
  #filename = 'custom_models.py'
  
  #if(os.path.exists(filename)):
  #	os.remove(filename)
  #	print('existing file:', filename, ' has been deleted')
    
  #print('downloading latest version of file:', filename)
  #wget.download(data_path, filename)
  #print('sucessfully downloaded')
  
  #data_path = 'https://raw.githubusercontent.com/5y3datif/Ear_Biometrics/main/training_helpers_v3.py'
  #filename = 'training_helpers_v3.py'
  
  # for noteable.io only
  #if(os.path.exists(filename)):
  #	os.remove(filename)
  #	print('existing file:', filename, ' has been deleted')
	
  #print('downloading latest version of file:', filename)
  #wget.download(data_path, filename)
  #print('sucessfully downloaded')
  
  from custom_models import LSE_model, Feature_Extraction_Module, Feature_Decoder_Module, 	AutoEncoder_model, Simple_Classification_model
  from training_helpers_v3 import train_epochs, train_one_epoch, reset_weights, train_folds, train_trails
  from spc import SupervisedContrastiveLoss
  
  options = parse_options()
    
  print(f'dataset: {options.dataset}')
  print(f'target_size: {eval(options.target_size)}')
  print(f'num_filters: {options.num_filters}')
  print(f'model_type: {options.model_type}')
  print(f'train_device: {options.train_device}')
  print(f'conv_type: {options.conv_type}')
  print(f'loss_fn_type: {options.loss_fn_type}')
  print(f'transformation: {options.transformation}')
  print(f'auto_augmentation: {options.auto_augmentation}')
  print(f'lambda1: {options.lambda1}')
  print(f'lambda2: {options.lambda2}')
  print(f'n_trails: {options.trails}')
  print(f'k_folds: {options.folds}')
  print(f'epochs_per_fold: {options.epochs}')
  
  if(os.path.exists('AMI_dataset')):
    shutil.rmtree('AMI_dataset')
  from utilities import load_dataset

  # LOADING Dataset
  dataset = options.dataset
  ear_images, sub_labels = load_dataset(dataset=dataset, target_size =eval(options.target_size))
  # ear_images, sub_labels = load_dataset(dataset='IITD_dataset', target_size = (50, 180))
      
  # finding input shape and num classes
  #data
  X_train, X_test, y_train, y_test = train_test_split(ear_images, sub_labels, test_size=0.142, random_state=42, stratify=sub_labels) # for AMI dataset
  
  print('Training dataset:\n',X_train.shape)
  print(y_train.shape)
  
  print('Test dataset:\n',X_test.shape)
  print(y_test.shape) 
  
  mean = np.mean(255*np.array(ear_images),axis=(0,2,3))
  std = np.std(255*np.array(ear_images),axis=(0,2,3))
  
  transform_train = []
  if options.transformation:
  	transform_train = [
      transforms_v2.RandomCrop(128, padding=4),
      transforms_v2.RandomHorizontalFlip(),
    ]
  if options.auto_augmentation:
  	transform_train.append(transforms_v2.AutoAugment())
  
  transform_train = transforms_v2.Compose(
    transform_train
  )
  
  transform_normalized = transforms_v2.Compose(
    [
      transforms_v2.Normalize(mean, std),
    ]
  )
  
  temp = transform_train(torch.tensor(255*X_train, dtype=torch.uint8))
  print(f'X_train temp shape: {temp.shape}')
  transformed_X_train = transform_normalized(temp.to(dtype=torch.float32))
    
  training_loader = DataLoader(TensorDataset(transformed_X_train, torch.tensor(y_train)), batch_size=128, shuffle=True)
  
  temp = torch.tensor(255*X_test, dtype=torch.float32)
  transformed_X_test = transform_normalized(temp)
  validation_loader = DataLoader(TensorDataset(transformed_X_test, torch.tensor(y_test)), batch_size=1)  
  
  # added by Atif
  num_training_samples = len(training_loader.dataset)
  num_validation_samples = len(validation_loader.dataset)
  
  training_samples = training_loader.dataset.tensors[0]
  training_targets = training_loader.dataset.tensors[1]
  
  input_shape=(training_samples.shape[2], training_samples.shape[3], training_samples.shape[1])
  num_classes=np.unique(training_targets).shape[0]
  
  print(f"Input Shape: {input_shape}, Number of Classes: {num_classes}")
    
  # Initialize the model and parameter.
  num_filters = options.num_filters
  model_type = options.model_type
  
  train_device = options.train_device
  conv_type = options.conv_type
  transformation = options.transformation
  auto_augmentation = options.auto_augmentation
  
  model = LSE_model(num_classes=num_classes, num_filters=num_filters,input_shape=input_shape, conv_type=conv_type).to(torch.device(train_device))
  
  # for model_type='Classifier' only.
  #if model_type == 'Classifier':
  #  resume_checkpoint = torch.load("auto_encoder_best_checkpoint_epochs_100.pth")
    # load model and optimizer
  #  model.load_state_dict(resume_checkpoint['model'])
  
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
  model_checkpoint = { 
    'model': model.state_dict(), 
  	'optimizer': optimizer.state_dict()
  }
  torch.save(model_checkpoint, "model_checkpoint.pth")
  
  loss_fn_type = options.loss_fn_type
  loss_fn = torch.nn.CrossEntropyLoss()
  if loss_fn_type == 'contrastive':
  	loss_fn2 = SupervisedContrastiveLoss()
  else:
  	loss_fn2 = torch.nn.MSELoss()  
  
  lambda1 = options.lambda1
  lambda2 = options.lambda2
  
  model_parameters = {'model_type': model_type, 'model': model, 
  					  'num_filters': num_filters, 'optimizer': optimizer,
                      'loss_fn_type': loss_fn_type, 'loss_fn': loss_fn, 
                      'loss_fn2': loss_fn2, 'lambda1': lambda1, 
                      'lambda2': lambda2}
  
  n_trails = options.trails
  k_folds = options.folds
  epochs_per_fold = options.epochs
  max_state = {'ntrails': n_trails, 'kfolds': k_folds, 'epochs': epochs_per_fold}
  current_state = {'trail': 1, 'fold': 1, 'epoch': 1}
  best_state = {'training_loss': 0, 'training_accuracy': 0,
              'validation_loss': 0,'validation_accuracy': 0,
              'trail': 0, 'fold': 0, 'epoch': 0}
  
  early_stop_thresh = 0.2*epochs_per_fold
  checkpoint_save_step=0
  results = [{'training_loss': 0, 'training_accuracy': 0,
            'validation_loss': 0,'validation_accuracy': 0,
            'trail': 0, 'fold': 0, 'epoch': 1}]*(n_trails*k_folds*epochs_per_fold)
  
  temp1 = 'augmentation' if transformation else ''
  temp2 = 'auto_augmentation' if auto_augmentation else ''
  auto_augmentation = options.auto_augmentation
  temp = f'{dataset}_{model_type}_{conv_type}_{loss_fn_type}_{temp1}_{temp2}_{num_filters}'
  print(temp)
  print(os.path.isdir(temp))
  if not os.path.isdir(temp):
  	os.mkdir(temp)
    #print('I am here')
    
    
  # run N trails 
  train_trails(ear_images, sub_labels, model_parameters = model_parameters,
               max_state = max_state, current_state = current_state, 
               best_state = best_state, transformation = transformation, 
               auto_augmentation = auto_augmentation, 
               early_stop_thresh = early_stop_thresh, train_device=train_device,
               checkpoint_save_step=checkpoint_save_step)
  
  if model_type == 'AutoEncoder':
    model_type = 'Classifier'
    temp = f'{dataset}_{model_type}_{conv_type}_{loss_fn_type}_{temp1}_{temp2}_{num_filters}'
    os.mkdir(temp)
    # run N trails
    train_trails(ear_images, sub_labels, model_parameters = model_parameters,
               max_state = max_state, current_state = current_state, 
               best_state = best_state, transformation = transformation, 
               auto_augmentation = auto_augmentation, 
               early_stop_thresh = early_stop_thresh, train_device=train_device,
               checkpoint_save_step=checkpoint_save_step)