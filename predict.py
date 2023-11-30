import numpy as np
import torch
import matplotlib.pyplot as plt
import json
from PIL import Image
import argparse
from torch.autograd import Variable
import os, random
from workspace_utils import keep_awake
import torchvision.models as model
import torchvision
from torch import nn
from collections import OrderedDict

#Get image input from the user
parser = argparse.ArgumentParser(description="get image input from the user")
parser.add_argument("-im", "--image", metavar = "image", choices = ['image_07211', 'image_07215', 'image_07218', 'image_07219', 'image_08099', 'image_08104'], type = str, help = "enter the full name of the flower in this directory")
parser.add_argument("-a", "--arch", type = str, metavar = "architecture", choices = ["vgg19", "vgg13", "densenet161"], default = "vgg13", help = "input the type of architecture for your model")

args = parser.parse_args()

if args.arch=="vgg19":
    model = model.vgg19(pretrained=True)
elif args.arch =="vgg13":
    model = model.vgg13(pretrained=True)
elif args.arch == "densenet161":
    model = model.densenet161(pretrained=True)
else:
    print("This is wrong model; Please try vgg19, vgg13, densenet161")
    

def load_checkpoint(filename):
    checkpoint = torch.load(filename)
    learning_rate = checkpoint['learning_rate']
    model.classifier = checkpoint['classifier']
    model.epochs = checkpoint['epochs']
    model.output_size = checkpoint['output_size']
    model.hidden_units = checkpoint['hidden_units']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    model.optimizer = checkpoint['optimizer']
    model.load_state_dict(checkpoint["state_dict"])
        
    return model
    

#Process the images
def process_image(image):
    pil_im =Image.open(image)
    pil_im.thumbnail((256, 256))
    
    #New dimension
    new_width = 224
    new_height = 224
    
    #get the original dimensions
    original_width, original_height = pil_im.size
    
    #get the dimensions to crop
    left = (original_width - new_width)/2
    top = (original_height - new_height)/2
    right = (original_width + new_width)/2
    bottom = (original_height + new_height)/2
    
    #crop the center of the image
    pil_im = pil_im.crop((left, top, right, bottom))
    
    #convert to numpy array between 0 and 1
    np_image = np.array(pil_im)/255
    
    #Normalize the color channel width mean and standard deviation
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    pil_image = (np_image-mean)/std
    
    #reirder the color channel to first dimension from third dimension
    pil_image = pil_image.transpose((2, 0, 1))
    
    return pil_image


#predict the image with the probabilities
def predict(image_path, model, topk=3):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #move the parameters to gpu
    model.to(device)
    
    #turn off dropout
    model.eval()
    
    #process the image
    image = process_image(image_path)
    
    #transfer to tensor
    image = torch.from_numpy(np.array([image])).float()
    
    image = Variable(image)
    
    #move the image into gpu
    image = image.to (device)
    
    #turn off the gradients
    with torch.no_grad():
        logps = model(image)
        ps = torch.exp(logps)
        
        #top 3 probabilities 
        top_prob = torch.topk(ps, topk)[0].tolist()[0]
        
        #top 3 indices
        top_index = torch.topk(ps, topk)[1].tolist()[0]
        
        
        #transfer index to label
        
        indices = []
        for i in range(len(list(model.class_to_idx.items()))):
            indices.append(list(model.class_to_idx.items())[i][0])
            
        label = []
        for i in range(3):
            label.append(indices[top_index[i]])
            
        return top_prob, label

    
with open("cat_to_name.json", "r") as f:
    cat_to_name = json.load(f)
    #print(cat_to_name)


choice = ['image_07211', 'image_07215', 'image_07218', 'image_07219', 'image_08099', 'image_08104']

img = ''
for i in range(6):
    if args.image == choice[i]:
        img_list = os.listdir('./flowers/test/7/')
        img += img_list[i]
       
img_path = './flowers/test/7/' + img


#plot the probabilities and predict the image

new_model = load_checkpoint('checkpoint.pth')
print(new_model)
prob,classes = predict(img_path, new_model)
max_index = np.argmax(prob)
max_probability = prob[max_index]
label = classes[max_index]

labels = []
for ibx in classes:
    labels.append(cat_to_name[ibx])
print(f'\nThe names of top three images with high probabilitites: {labels}')
print(f'\nThe probabilities of top three images: {prob}')
      
print(f'\nThe probability of top image: {max_probability}')
print(f'\nThe index and the name of the top image with high probability: {label, labels[0]}')
      

 