import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn
import torch.nn.functional
import torch.optim
from torchvision import models #just for debugging
from  sklearn.model_selection import train_test_split, StratifiedShuffleSplit, ShuffleSplit, StratifiedKFold, KFold # KFold is added by Atif
from sklearn.utils import shuffle

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y]

# manaul training
def train_one_epoch(training_loader, validation_loader,
                    num_training_samples, num_validation_samples,
                    input_shape=(351, 246, 3), num_classes=100, num_filters=8,
                    model_type='Encoder+Classifier', model=None,
                    optimizer=None, loss_fn = torch.nn.CrossEntropyLoss(), 
                    loss_fn2 = torch.nn.MSELoss(), lambda1=0.5, lambda2=0.5, 
                    train_device='cuda'):

    # training metrics
    train_loss = 0
    train_correct = 0

    # validation metrics
    valid_loss = 0
    valid_correct = 0


    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    model.train(True)
    if(model_type=='Classifier'):
      lambda1=1;
      lambda2=0;
      ct = 0
      for child in model.children():
        ct += 1
        if ct == 2: # turn off weigth update for Encoder and Decoder modules
            for param in child.parameters():
                param.requires_grad = True
        else:
            for param in child.parameters():
                param.requires_grad = False
    elif(model_type=='Encoder+Classifier'):
      lambda1=1;
      lambda2=0;
      ct = 0
      for child in model.children():
        ct += 1
        if ct == 3: # turn off weigth update for Decoder module only
            for param in child.parameters():
                param.requires_grad = False
        else:
            for param in child.parameters():
                param.requires_grad = True
    elif(model_type=='AutoEncoder'):
      lambda1=0;
      lambda2=1;
      ct = 0
      for child in model.children():
        ct += 1
        if ct == 2: # turn off weigth update for Classifier module
            for param in child.parameters():
                param.requires_grad = False
        else:
            for param in child.parameters():
                param.requires_grad = True
    elif(model_type=='DeepLSE'): # train full-network
      for child in model.children():
        for param in child.parameters():
            param.requires_grad = True
    else:
      print('Incorrect choice for model configuration')

    for i, data in enumerate(training_loader,0):
        # Every data instance is an input + label pair
        train_input, train_label = data
        # train_input = train_input.unsqueeze(dim=1).float()
        train_label= torch.tensor(to_categorical(y=train_label, num_classes=num_classes)).float()
        # train_label = train_label[:,None]
        if len(train_label.shape)==1:
          train_label = train_label.unsqueeze(dim=0)

        train_input = train_input.to(torch.device(train_device))
        train_label = train_label.to(torch.device(train_device))

        # print('train_input:',train_input.shape, 'train_label:',train_label.shape)

        # Zero your gradients for every batch!
        optimizer.zero_grad()
        # optimizer_classifier.zero_grad()

        # Make predictions for this batch
        # train_features_output = feature_extraction_module(train_input)
        # train_output = classifier_module(train_features_output)
        # train_output = pytorch_model_c1(train_input)
        train_output, decoded_input = model(train_input)

        # print('train_input:',train_input.shape, 'train_label:',train_label.shape, 'train_output:',train_output.shape)
        # print('train_label:',train_label)
        # print('train_output:',train_output)

        # Compute the loss and its gradients
        loss = 2*(lambda1)*loss_fn(train_output, train_label)
        loss = loss + 2*(lambda2)*loss_fn2(train_input, decoded_input)

        loss.backward()

        # Adjust learning weights
        optimizer.step()
        # optimizer_classifier.step()

        # Gather data and report
        train_loss += loss.item()
        for batch_count in range(train_output.shape[0]):
          if(torch.argmax(train_output[batch_count,:]) == torch.argmax(train_label[batch_count,:])):
            train_correct += 1

    # print('training epoch complete')
    # Here, we use enumerate(validation_loader) instead of
    # iter(validation_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    model.train(False)
    for i, data in enumerate(validation_loader,0):
        # Every data instance is an input + label pair
        valid_input, valid_label = data

        # valid_input = valid_input.unsqueeze(dim=1).float()
        valid_label= torch.tensor(to_categorical(y=valid_label, num_classes=num_classes)).float()
        if len(valid_label.shape)==1:
          valid_label = valid_label.unsqueeze(dim=0)

        valid_input = valid_input.to(torch.device(train_device))
        valid_label = valid_label.to(torch.device(train_device))

        # Make predictions for this batch
        valid_output, temp = model(valid_input)

        # print('valid_input:',valid_input.shape, 'valid_label:',valid_label.shape, 'valid_output:',valid_output.shape)

        # Gather data and report
        valid_loss += loss_fn(valid_output, valid_label).item() + loss_fn2(valid_input, temp).item()
        for batch_count in range(valid_output.shape[0]):
          if(torch.argmax(valid_output[batch_count,:]) == torch.argmax(valid_label[batch_count,:])):
            valid_correct += 1

    training_accuracy = 100*train_correct/num_training_samples
    validation_accuracy = 100*valid_correct/num_validation_samples

    return train_loss, training_accuracy, valid_loss, validation_accuracy

# added by Atif
def checkpoint(current_checkpoint, filename):
  torch.save(current_checkpoint, filename)

#def resume(model, optimizer, filename):
#  checkpoint = torch.load(filename)
#  model.load_state_dict(checkpoint['model'])
#  optimizer.load_state_dict(checkpoint['optimizer'])
#  best_validation_accuracy = checkpoint['best_validation_accuracy']
#  best_validation_trail = checkpoint['best_validation_trail']
#  best_validation_fold = checkpoint['best_validation_fold']
#  best_validation_epoch = checkpoint['best_validation_epoch']
#  return best_validation_accuracy, best_validation_trail, best_validation_fold, best_validation_epoch


def train_epochs(X_train, y_train, X_test, y_test, input_shape=(351, 246, 3), 
                 num_classes=100, num_filters=8, model_type='Encoder+Classifier', 
                 model=None, optimizer=None, loss_fn = torch.nn.CrossEntropyLoss(), 
                 loss_fn2 = torch.nn.MSELoss(), lambda1=0.5, lambda2=0.5, 
                 epochs = 50, early_stop_thresh = 5, train_device='cuda', 
                 resume_from=None, results=np.empty([0]), 
                 best_validation_accuracy=0, trail=0, fold=0, epoch = 1):
                     
  #resume
  if not resume_from == None:
      resume_checkpoint = torch.load(resume_from)
      trail = resume_checkpoint['trail']
      fold = resume_checkpoint['fold']
      epoch = resume_checkpoint['epoch']
      best_validation_accuracy = resume_checkpoint['best_validation_accuracy']
      # load model and optimizer 
      model.load_state_dict(checkpoint['model'])
      optimizer.load_state_dict(checkpoint['optimizer'])
  
  #data
  training_loader = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(y_train)), batch_size=100, shuffle=True)
  validation_loader = DataLoader(TensorDataset(torch.tensor(X_test), torch.tensor(y_test)), batch_size=1)
  # added by Atif
  num_training_samples = len(training_loader.dataset)
  num_validation_samples = len(validation_loader.dataset)

  thresh_epoch = 1
  for epoch in range(epoch, epochs+1):
    print('EPOCH {}/{}:'.format(epoch,epochs))
    train_loss, training_accuracy, valid_loss, validation_accuracy = \
      train_one_epoch(training_loader, validation_loader,
                      num_training_samples, num_validation_samples,
                      input_shape=input_shape, num_classes=num_classes,
                      num_filters=num_filters, model_type=model_type,
                      model=model, optimizer=optimizer,
                      loss_fn=loss_fn, loss_fn2=loss_fn2,
                      lambda1=lambda1, lambda2=lambda2, train_device=train_device)

    print(f"Training: \n Training Accuracy: {training_accuracy}%, Average Training Loss: {train_loss/len(training_loader)}")

    print(f"Validation: \n Validation Accuracy: {validation_accuracy}%, Average Validation Loss: {valid_loss/len(validation_loader)}")

    if validation_accuracy > best_validation_accuracy: 
        best_validation_accuracy = validation_accuracy 
        thresh_epoch = 1
        #best_validation_epoch = epoch 
        # creating the best checkpoint
        best_checkpoint = { 
            'model': model.state_dict(), 
            'optimizer': optimizer.state_dict(),
            'training_accuracy': training_accuracy,
            'validation_accuracy': validation_accuracy, 
            'trail': trail, 
            'fold': fold,
            'epoch': epoch,
            'best_validation_accuracy': best_validation_accuracy,
            'results': results,
        }
        checkpoint(best_checkpoint, "best_checkpoint.pth")
    
    # creating the latest checkpoint
    latest_checkpoint = { 
        'model': model.state_dict(), 
        'optimizer': optimizer.state_dict(),
        'training_accuracy': training_accuracy,
        'validation_accuracy': validation_accuracy, 
        'trail': trail, 
        'fold': fold,
        'epoch': epoch,
        'best_validation_accuracy': best_validation_accuracy,
        'results': results,
    }
    checkpoint(latest_checkpoint, "latest_checkpoint.pth")
      
    if thresh_epoch >= early_stop_thresh:
        print(f"Early stopped training at epoch {epoch}. \nThe best vaidation accuarcy was {best_validation_accuracy}")
        break  # terminate the training loop
    thresh_epoch+=1
    
  return best_validation_accuracy

def reset_weights(m):
  '''
    Try resetting model weights to avoid
    weight leakage.
  '''
  for layer in m.children():
   if hasattr(layer, 'reset_parameters'):
    print(f'Reset trainable parameters of layer = {layer}')
    layer.reset_parameters()

def train_folds(ear_images, sub_labels, k_folds, input_shape=(351, 246, 3),
                num_classes=100, num_filters=8, model_type='Encoder+Classifier', 
                model=None, optimizer=None, 
                loss_fn = torch.nn.CrossEntropyLoss(), 
                loss_fn2 = torch.nn.MSELoss(), lambda1=0.5, lambda2=0.5,
                epochs_per_fold = 50, early_stop_thresh = 5, train_device='cuda', 
                resume_from=None, results=np.empty([0]), best_validation_accuracy=0, trail=0, fold=1, epoch = 1):

  
  #resume
  if not resume_from == None:
      resume_checkpoint = torch.load(resume_from)
      trail = resume_checkpoint['trail']
      fold = resume_checkpoint['fold']
      epoch = resume_checkpoint['epoch']
      best_validation_accuracy = resume_checkpoint['best_validation_accuracy']
      # load model and optimizer 
      model.load_state_dict(checkpoint['model'])
      optimizer.load_state_dict(checkpoint['optimizer'])
  
  # Set fixed random number seed
  kfold = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

  # For k fold results
  if trail== 0:
      results = np.zeros(k_folds)

  # Print k-fold results
  print(f'K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS')
  print('--------------------------------')
  sum = 0.0
  # K-fold Cross Validation model evaluation
  for fold, (train_ids, test_ids) in enumerate(kfold.split(ear_images, sub_labels)):

    # Print
    print(f'FOLD {fold}')
    print('--------------------------------')

    X_train = ear_images[train_ids, :, :, :]
    y_train = sub_labels[train_ids]
    X_test = ear_images[test_ids, :, :, : ]
    y_test = sub_labels[test_ids]

    print('Training dataset:\n',X_train.shape)
    print(y_train.shape)
    print('Test dataset:\n',X_test.shape)
    print(y_test.shape)

    # Reset model weights before each fold
    model.apply(reset_weights)
    best_validation_accuracy = train_epochs(X_train, y_train, X_test, y_test, 
                                            input_shape=input_shape, 
                                            num_classes=num_classes, 
                                            num_filters=num_filters, 
                                            model_type=model_type, 
                                            model=model, optimizer=optimizer, 
                                            loss_fn=loss_fn, loss_fn2=loss_fn2, 
                                            lambda1=lambda1, lambda2=lambda2, 
                                            epochs=epochs_per_fold, 
                                            early_stop_thresh=early_stop_thresh, 
                                            train_device=train_device, 
                                            resume_from=resume_from, results=results, 
                                            best_validation_accuracy=best_validation_accuracy, 
                                            trail=trail, fold=fold, epoch=epoch)

    print(f'Fold {fold}: {best_validation_accuracy} %')
    sum += best_validation_accuracy
    if trail== 0:
        results[fold] = best_validation_accuracy
    else:
        results[trail,fold] = best_validation_accuracy
  
  print(f'Average: {sum/k_folds} %')
  
  return best_validation_accuracy

def train_trails(n_trails, ear_images, sub_labels, k_folds, input_shape=(351, 246, 3),
                num_classes=100, num_filters=8, model_type='Encoder+Classifier',
                model=None, optimizer=None, loss_fn = torch.nn.CrossEntropyLoss(),
                loss_fn2 = torch.nn.MSELoss(), lambda1=0.5, lambda2=0.5,
                epochs_per_fold = 50, early_stop_thresh = 5,
                resume_from=None, train_device='cuda'):

  trail = 1
  #resume
  if not resume_from == None:
      resume_checkpoint = torch.load(resume_from)
      trail = resume_checkpoint['trail']
      fold = resume_checkpoint['fold']
      epoch = resume_checkpoint['epoch']
      best_validation_accuracy = resume_checkpoint['best_validation_accuracy']
      # load model and optimizer 
      model.load_state_dict(checkpoint['model'])
      optimizer.load_state_dict(checkpoint['optimizer'])
  else:
      trail = 1
      fold = 1
      epoch = 1
      best_validation_accuracy = 0
      
  # For N trail results
  results = np.zeros((n_trails, k_folds))

  # Print N trail results
  print(f'N-TRAILS CROSS VALIDATION RESULTS FOR {k_folds} FOLDS')
  print('--------------------------------')
  sum = 0.0
  # N-trail Cross Validation model evaluation
  for trail in range(trail, n_trails+1):
    print(f"Trail: {trail}")
    X, y = shuffle(ear_images, sub_labels, random_state=42)
    best_val_acc = train_folds(X, y, k_folds, input_shape=input_shape, 
                               num_classes=num_classes, num_filters=num_filters, 
                               model_type=model_type, model=model, 
                               optimizer=optimizer, loss_fn=loss_fn, 
                               loss_fn2=loss_fn2, lambda1=lambda1, 
                               lambda2=lambda2, epochs_per_fold = epochs_per_fold, 
                               early_stop_thresh = early_stop_thresh, 
                               train_device=train_device, resume_from=resume_from, 
                               results=results, 
                               best_validation_accuracy=best_validation_accuracy, 
                               trail=trail, fold=fold, epoch=epoch)

    print(f'Trail {trail}: {np.sum(results[trail])} %')
    sum += np.sum(results[trail])

  print(f'Average: {sum/n_trails} %')
  
