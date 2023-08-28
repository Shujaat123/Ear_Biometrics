import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn
import torch.nn.functional
import torch.optim
from torchvision import models #just for debugging

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y]

# added by Atif
def checkpoint(model, optimizer, filename):
  torch.save({
    'optimizer': optimizer.state_dict(),
    'model': model.state_dict(),
  }, filename)

def resume(model, optimizer, filename):
  checkpoint = torch.load(filename)
  model.load_state_dict(checkpoint['model'])
  optimizer.load_state_dict(checkpoint['optimizer'])

# manaul training
def train_one_epoch(training_loader, validation_loader,
                    num_training_samples, num_validation_samples,
                    input_shape=(351, 246, 3), num_classes=100, num_filters=8,
                    model_type='Encoder+Classifier', model=None,
                    optimizer=None, loss_fn = torch.nn.CrossEntropyLoss(), 
                    loss_fn2 = torch.nn.MSELoss(), lambda1=0.5, lambda2=0.5, 
                    device='cuda'):

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

        train_input = train_input.to(torch.device(device))
        train_label = train_label.to(torch.device(device))

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

        valid_input = valid_input.to(torch.device('cuda'))
        valid_label = valid_label.to(torch.device('cuda'))

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

def train_epoches(training_loader, validation_loader,
                  num_training_samples, num_validation_samples,
                  input_shape=(351, 246, 3), num_classes=100, num_filters=8,
                  model_type='Encoder+Classifier', model=None,
                  optimizer=None, loss_fn = torch.nn.CrossEntropyLoss(),
                  loss_fn2 = torch.nn.MSELoss(), lambda1=0.5, lambda2=0.5,
                  epochs = 50, resume=False, early_stop_thresh = 5,
                  device='cuda'):


  epochs = 50
  #optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

  best_validation_accuracy = -1
  best_validation_epoch = -1

  for epoch in range(epochs):
    print('EPOCH {}:'.format(epoch))
    # if (epoch<3):
    #   lambda1=0
    # elif ((epoch>=3)&(epoch<10)):
    #   lambda1=0.2
    # elif ((epoch>=10)&(epoch<20)):
    #lambda1=0.5
    #lambda2=0.5
    # model_type='DeepLSE'
    # model_type='AutoEncoder'
    # model_type='Classifier'
    #model_type='Encoder+Classifier'
    # else:
    #   lambda1=0.6
    # lambda1=(1+epoch)/EPOCHS
    # lambda1 = np.remainder(epoch,2)

    train_loss, training_accuracy, valid_loss, validation_accuracy = \
      train_one_epoch(training_loader, validation_loader,
                      num_training_samples, num_validation_samples,
                      input_shape=input_shape, num_classes=num_classes,
                      num_filters=num_filters, model_type=model_type,
                      model=model, optimizer=optimizer,
                      loss_fn=loss_fn, loss_fn2=loss_fn2,
                      lambda1=lambda1, lambda2=lambda2, device=train_device)

    print(f"Training: \n Training Accuracy: {training_accuracy}%, Average Training Loss: {train_loss/len(training_loader)}")

    print(f"Validation: \n Validation Accuracy: {validation_accuracy}%, Average Validation Loss: {valid_loss/len(validation_loader)}")

    if validation_accuracy > best_validation_accuracy:
      best_validation_accuracy = validation_accuracy
      best_validation_epoch = epoch
      checkpoint(model, optimizer, "best_model.pth")

    elif epoch - best_validation_epoch > early_stop_thresh:
        print(f"Early stopped training at epoch {epoch}. \nThe epoch of best vaidation accuarcy was {best_validation_epoch} with vaidation accuarcy of {best_validation_accuracy}")
        break  # terminate the training loop
