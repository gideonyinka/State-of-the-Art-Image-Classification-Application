import os
from torch import nn, optim
from torchvision import datasets, transforms
import torchvision.models as model
import torch
import torchvision
from collections import OrderedDict
from workspace_utils import keep_awake
import argparse


data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

#Define the transformation for the inputs
train_transforms = transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), 
                                      transforms.RandomRotation(30), transforms.ToTensor(), 
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), 
                                      transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


valid_transforms = transforms.Compose([transforms.Resize(224), transforms.ToTensor(), 
                                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

#Load the Datasets with ImageFolder
train_datasets = datasets.ImageFolder(train_dir, transform = train_transforms)
test_datasets = datasets.ImageFolder(test_dir, transform = test_transforms)
valid_datasets = datasets.ImageFolder(valid_dir, transform = valid_transforms)

# Load the Images with the dataloaders
train_dataloaders = torch.utils.data.DataLoader(train_datasets, batch_size = 64, shuffle = True)
test_dataloaders = torch.utils.data.DataLoader(test_datasets, batch_size = 64)
valid_dataloaders = torch.utils.data.DataLoader(valid_datasets, batch_size = 64)

image_datasets = [train_datasets, test_datasets, valid_datasets]
image_loaders = [train_dataloaders, test_dataloaders, valid_dataloaders]

architectures = {"vgg19":25088, "vgg13":25088, "densenet161": 2208}
     
parser = argparse.ArgumentParser(description = "training my model")
parser.add_argument("-a", "--arch", type = str, metavar = "architecture",
                    choices = ["vgg19", "vgg13", "densenet161"],
                    default = "vgg13", help = "input the type of architecture for your model")
parser.add_argument("-e", "--epochs", type = int, metavar = "epochs", 
                    default = 1, help = "input the number of epoch in integer")
parser.add_argument("-d", "--dropout", type = float, metavar = "dropout", 
                    default = 0.25, help = "input dropout number in float")
parser.add_argument("-lr", "--learning_rate", type = float, metavar = "learning_rate", 
                    default = 0.001, help = "set the learning rate in float")
parser.add_argument("-hu", "--hidden_units", type = int, metavar = "hidden_units", 
                    default = 1000, help = "sets the hidden units in integer")
parser.add_argument("-sd", "--save_dir", metavar = "save_directory", default = "checkpoint.pth", help = "input save directory")
    
args =parser.parse_args()

if args.arch=="vgg19":
    model = model.vgg19(pretrained=True)
elif args.arch =="vgg13":
    model = model.vgg13(pretrained=True)
elif args.arch == "densenet161":
    model = model.densenet161(pretrained=True)
else:
    print("This is wrong model; Please try vgg19, vgg13, densenet161")
    
#Set options for either gpu or cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    
#Freeze the model parameters
for param in model.parameters():
    param.requires_grad = False
    
#Build the network

Classifier = nn.Sequential(OrderedDict([("ly1", nn.Linear(architectures[args.arch], args.hidden_units)), ("relu", nn.ReLU()), ("dropout", nn.Dropout(p=args.dropout)), ("ly2", nn.Linear(args.hidden_units, 102)), ("output", nn.LogSoftmax(dim=1))]))
    
#update the classifier
model.classifier = Classifier

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr = args.learning_rate)
        
#move the model to device
model = model.to(device)
        
#set the parameters
epoch = args.epochs
steps = 0
training_loss = 0
print_every = 5
    
for idx in range(epoch):
    #load the images and labels into device
    for images, labels in train_dataloaders:
        steps += 1
        images, labels = images.to(device), labels.to(device)
        
        #Zero the gradient
        optimizer.zero_grad()
        logps = model(images)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()
        training_loss += loss.item()
        
        #Do validation of the test set
        if steps % print_every == 0:
            
            #turn off the dropout
            model.eval()
            validation_loss = 0
            accuracy = 0
            
            #load the test images and labels
            for images, labels in test_dataloaders:
                #move the images and labels into device
                images, labels = images.to(device), labels.to(device)
                logps = model(images)
                loss = criterion(logps, labels)
                validation_loss += loss.item()
                
                #calculate the accuracy
                ps = torch.exp(logps)
                top_ps, top_class = ps.topk(1, dim=1)
                equality = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equality.type(torch.FloatTensor))
                
            print(f'Epoch {idx+1}/{epoch}..', f'Train loss: {training_loss/print_every:.3f}..', f'Validation loss: {validation_loss/len(test_dataloaders):.3f}..', f'Test accuracy: {(accuracy/len(test_dataloaders))*100:.3f}')
            
            training_loss = 0
            model.train()
                          
#save the checkpoint
model.class_to_idx = image_datasets[0].class_to_idx
checkpoint = {"input_size":architectures[args.arch], "output_size":102, "epochs":args.epochs, "batch_size":64, "learning_rate":args.learning_rate, "classifier":Classifier, "class_to_idx": model.class_to_idx,"hidden_units": args.hidden_units, "state_dict":model.state_dict(), "optimizer":optimizer.state_dict()}
if args.save_dir == "checkpoint.pth":
    torch.save(checkpoint, "checkpoint.pth")
else:
    print("Wrong Directory! Please enter checkpoint.pth")                         
                        
                    
                        
                        
           
            
            
            
    