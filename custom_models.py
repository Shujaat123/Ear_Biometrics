import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn
import torch.nn.functional
import torch.optim
from torchvision import models #just for debugging


class Feature_Extraction_Module(torch.nn.Module):
  #  Determine what layers and their order in CNN object
  def __init__(self, num_classes=221, num_filters=8, input_shape=(180,50,3)):
    super(Feature_Extraction_Module,self).__init__()
    #self.encoder_input = input_shape[-1]
    kernel_size = 3
    # Encoder Layer1
    self.encoder_layer1_name = 'encoder_layer1'
    self.encoder_layer1_conv = torch.nn.Conv2d(input_shape[2],
                                               num_filters,
                                               kernel_size,
                                               padding='same')

    self.encoder_layer1_activation = torch.nn.ReLU()
    self.encoder_layer1_pooling = torch.nn.MaxPool2d(kernel_size=(2, 2))

    # Encoder Layer2
    self.encoder_layer2_name = 'encoder_layer2'
    self.encoder_layer2_conv = torch.nn.Conv2d(num_filters,
                                               2*num_filters,
                                               kernel_size,
                                               padding='same')
    self.encoder_layer2_activation = torch.nn.ReLU()
    self.encoder_layer2_batch_norm = torch.nn.BatchNorm2d(2*num_filters,
                                                          eps = 1e-3,
                                                          momentum = 0.99)
    self.encoder_layer2_pooling = torch.nn.MaxPool2d(kernel_size=(2, 2))

    # Encoder Layer3
    self.encoder_layer3_name = 'encoder_layer3'
    self.encoder_layer3_conv = torch.nn.Conv2d(2*num_filters,
                                               4*num_filters,
                                               kernel_size,
                                               padding='same')
    self.encoder_layer3_activation = torch.nn.ReLU()
    self.encoder_layer3_pooling = torch.nn.MaxPool2d(kernel_size=(2, 2))

    # Encoder Layer4
    self.encoder_layer4_name = 'encoder_layer4'
    self.encoder_layer4_conv = torch.nn.Conv2d(4*num_filters,
                                               8*num_filters,
                                               kernel_size,
                                               padding='same')
    self.encoder_layer4_activation = torch.nn.ReLU()
    self.encoder_layer4_batch_norm = torch.nn.BatchNorm2d(8*num_filters,
                                                          eps = 1e-3,
                                                          momentum = 0.99)
    self.encoder_layer4_pooling = torch.nn.MaxPool2d(kernel_size=(2, 2))

    # Encoder Layer5
    self.encoder_layer5_name = 'encoder_layer5'
    self.encoder_layer5_conv = torch.nn.Conv2d(8*num_filters,
                                               16*num_filters,
                                               kernel_size,
                                               padding='same')

    self.encoder_layer5_activation = torch.nn.ReLU()
    self.encoder_layer5_pooling = torch.nn.MaxPool2d(kernel_size=(2, 2))

   # Encoder Layer6
    self.encoder_layer6_name = 'encoder_layer2'
    self.encoder_layer6_conv = torch.nn.Conv2d(16*num_filters,
                                               32*num_filters,
                                               kernel_size,
                                               padding='same')
    self.encoder_layer6_activation = torch.nn.ReLU()
    self.encoder_layer6_batch_norm = torch.nn.BatchNorm2d(32*num_filters,
                                                          eps = 1e-3,
                                                          momentum = 0.99)

  def forward(self,x):
    # Encoder Layer1
    out = self.encoder_layer1_conv(x)
    out = self.encoder_layer1_activation(out)
    out = self.encoder_layer1_pooling(out)

    # Encoder Layer2
    out = self.encoder_layer2_conv(out)
    out = self.encoder_layer2_activation(out)
    out = self.encoder_layer2_batch_norm(out)
    out = self.encoder_layer2_pooling(out)

    # Encoder Layer3
    out = self.encoder_layer3_conv(out)
    out = self.encoder_layer3_activation(out)
    out = self.encoder_layer3_pooling(out)

    # Encoder Layer4
    out = self.encoder_layer4_conv(out)
    out = self.encoder_layer4_activation(out)
    out = self.encoder_layer4_batch_norm(out)
    out = self.encoder_layer4_pooling(out)

    # Encoder Layer5
    out = self.encoder_layer5_conv(out)
    out = self.encoder_layer5_activation(out)
    out = self.encoder_layer5_pooling(out)

    # Encoder Layer6
    out = self.encoder_layer6_conv(out)
    out = self.encoder_layer6_activation(out)
    out = self.encoder_layer6_batch_norm(out)

    return out
  

class Feature_Decoder_Module(torch.nn.Module):
  #  Determine what layers and their order in CNN object
  def __init__(self, num_classes=221, num_filters=8, input_shape=(180,50,3)):
    super(Feature_Decoder_Module,self).__init__()
    #self.encoder_input = input_shape[-1]
    kernel_size = 3
    # Encoder Layer1
    self.encoder_layer1_name = 'encoder_layer1'
    self.encoder_layer1_conv = torch.nn.Conv2d(num_filters,
                                               input_shape[2],
                                               kernel_size,
                                               padding='same')

    self.encoder_layer1_activation = torch.nn.ReLU()
    self.encoder_layer1_pooling = torch.nn.MaxPool2d(kernel_size=(2, 2))

    # Encoder Layer2
    self.encoder_layer2_name = 'encoder_layer2'
    self.encoder_layer2_conv = torch.nn.ConvTranspose2d(2*num_filters,
                                               num_filters,
                                               kernel_size, stride=2, padding=(0,0), output_padding=(0,0))
    self.encoder_layer2_activation = torch.nn.ReLU()
    self.encoder_layer2_batch_norm = torch.nn.BatchNorm2d(num_filters,
                                                          eps = 1e-3,
                                                          momentum = 0.99)
    self.encoder_layer2_pooling = torch.nn.MaxPool2d(kernel_size=(2, 2))

    # Encoder Layer3
    self.encoder_layer3_name = 'encoder_layer3'
    self.encoder_layer3_conv = torch.nn.ConvTranspose2d(4*num_filters,
                                               2*num_filters,
                                               kernel_size, stride=2, padding=(0,0), output_padding=(0,0))
    self.encoder_layer3_activation = torch.nn.ReLU()
    # self.encoder_layer3_pooling = torch.nn.MaxPool2d(kernel_size=(2, 2))

    # Encoder Layer4
    self.encoder_layer4_name = 'encoder_layer4'
    self.encoder_layer4_conv = torch.nn.ConvTranspose2d(8*num_filters,
                                               4*num_filters,
                                               kernel_size, stride=2, padding=(0,0), output_padding=(0,0))
    self.encoder_layer4_activation = torch.nn.ReLU()
    self.encoder_layer4_batch_norm = torch.nn.BatchNorm2d(4*num_filters,
                                                          eps = 1e-3,
                                                          momentum = 0.99)

    # Encoder Layer5
    self.encoder_layer5_name = 'encoder_layer5'
    self.encoder_layer5_conv = torch.nn.ConvTranspose2d(16*num_filters,
                                               8*num_filters,
                                               kernel_size, stride=2, padding=(0,0), output_padding=(0,0))

    self.encoder_layer5_activation = torch.nn.ReLU()

   # Encoder Layer6
    self.encoder_layer6_name = 'decoder_layer6'
    self.encoder_layer6_conv = torch.nn.ConvTranspose2d(32*num_filters,
                                               16*num_filters,
                                               kernel_size, stride=2, padding=(0,0), output_padding=(0,0))
    self.encoder_layer6_activation = torch.nn.ReLU()
    self.encoder_layer6_batch_norm = torch.nn.BatchNorm2d(16*num_filters,
                                                          eps = 1e-3,
                                                          momentum = 0.99)

    # Dense layer
    # self.fc1_flatten = torch.nn.Flatten()
    # self.fc1_linear = torch.nn.Linear(32*num_filters*(input_shape[0]//(2**5))*(input_shape[1]//(2**5)), num_classes)
    # self.fc1_activation = torch.nn.Softmax()

  def forward(self,x):

    # Encoder Layer6
    out = self.encoder_layer6_conv(x)
    out = self.encoder_layer6_activation(out)
    out = self.encoder_layer6_batch_norm(out)

    # Encoder Layer5
    out = self.encoder_layer5_conv(out)
    out = self.encoder_layer5_activation(out)
    # out = self.encoder_layer5_pooling(out)

    # Encoder Layer4
    out = self.encoder_layer4_conv(out)
    out = self.encoder_layer4_activation(out)
    out = self.encoder_layer4_batch_norm(out)
    # out = self.encoder_layer4_pooling(out)

    # Encoder Layer3
    out = self.encoder_layer3_conv(out)
    out = self.encoder_layer3_activation(out)
    # out = self.encoder_layer3_pooling(out)

    # Encoder Layer2
    out = self.encoder_layer2_conv(out)
    out = self.encoder_layer2_activation(out)
    out = self.encoder_layer2_batch_norm(out)
    # out = self.encoder_layer2_pooling(out)

    # Encoder Layer1
    out = self.encoder_layer1_conv(out)
    out = self.encoder_layer1_activation(out)
    # out = self.encoder_layer1_pooling(out)

    return out


class Classifier_Module(torch.nn.Module):
  #  Determine what layers and their order in CNN object
  def __init__(self, num_classes=221, num_filters=8, input_shape=(180,50,3)):
    super(Classifier_Module,self).__init__()

    # Dense layer
    self.fc1_flatten = torch.nn.Flatten()
    self.fc1_linear = torch.nn.Linear(32*num_filters*(input_shape[0]//(2**5))*(input_shape[1]//(2**5)), num_classes)
    self.fc1_activation = torch.nn.Softmax()

  def forward(self,x):
    # Dense Layer
    out = self.fc1_flatten(x)
    out = self.fc1_linear(out)
    out = self.fc1_activation(out)

    return out
  
class LSE_model(torch.nn.Module):
  #  Determine what layers and their order in CNN object
  def __init__(self, num_classes=100, num_filters=8, input_shape=(180,50,3)):
    super(LSE_model,self).__init__()

    self.feature_extraction_module = Feature_Extraction_Module(num_classes=num_classes, num_filters=num_filters, input_shape=input_shape)
    self.classification_module = Classifier_Module(num_classes=num_classes, num_filters=num_filters, input_shape=input_shape)
    self.feature_decoder_module = Feature_Decoder_Module(num_classes=num_classes, num_filters=num_filters, input_shape=input_shape)
    self.input_shape = input_shape

  def forward(self,x):
    # Encoder Layer1
    encoded_output = self.feature_extraction_module(x)
    out = self.classification_module(encoded_output)
    decoded_output = self.feature_decoder_module(encoded_output)
    return out, decoded_output[:,:,0:self.input_shape[0],0:self.input_shape[1]]


class Simple_Classification_model(torch.nn.Module):
  #  Determine what layers and their order in CNN object
  def __init__(self, num_classes=100, num_filters=8, input_shape=(180,50,3)):
    super(Simple_Classification_model,self).__init__()

    self.feature_extraction_module = Feature_Extraction_Module(num_classes=num_classes, num_filters=num_filters, input_shape=input_shape)
    self.classification_module = Classifier_Module(num_classes=num_classes, num_filters=num_filters, input_shape=input_shape)
    self.input_shape = input_shape

  def forward(self,x):
    # Encoder Layer1
    encoded_output = self.feature_extraction_module(x)
    out = self.classification_module(encoded_output)
    return out


class AutoEncoder_model(torch.nn.Module):
  #  Determine what layers and their order in CNN object
  def __init__(self, num_classes=100, num_filters=8, input_shape=(180,50,3)):
    super(AutoEncoder_model,self).__init__()

    self.feature_extraction_module = Feature_Extraction_Module(num_classes=num_classes, num_filters=num_filters, input_shape=input_shape)
    self.feature_decoder_module = Feature_Decoder_Module(num_classes=num_classes, num_filters=num_filters, input_shape=input_shape)
    self.input_shape = input_shape

  def forward(self,x):
    # Encoder Layer1
    encoded_output = self.feature_extraction_module(x)
    decoded_output = self.feature_decoder_module(encoded_output)
    return out, decoded_output[:,:,0:self.input_shape[0],0:self.input_shape[1]]

  

  
