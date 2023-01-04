# Define Imports
import torch
from torchvision import models
import numpy as np
import torch.utils.data
from collections import OrderedDict
from PIL import Image
import argparse
import json

# Define Arguments using argparse (https://docs.python.org/3/library/argparse.html)
parser = argparse.ArgumentParser (description = "Parser for predict.py")
parser.add_argument ('image_dir', help = 'Provide path to image. Required argument', type = str)
parser.add_argument ('load_dir', help = 'Provide path to checkpoint. Required argument', type = str)
parser.add_argument ('--top_k', help = 'Top K most likely classes (example: 5  will provide top 3 classes). Optional input', type = int)
parser.add_argument ('--category_names', help = 'Mapping of categories to real names. Provide JSON file name. Optional input', type = str)
parser.add_argument ('--GPU', help = "Option to use GPU. Optional input", type = str)

def loading_model (file_path):
    ''' Function to loads the checkpoint.pth file 
        defined in train.py and rebuilds the model
    '''

    checkpoint = torch.load (file_path) #loading checkpoint from a file
    if checkpoint ['arch'] == 'alexnet':
        model = models.alexnet (pretrained = True)
    else: #vgg13 as only 2 options available
        model = models.vgg13 (pretrained = True)
    model.classifier = checkpoint ['classifier']
    model.load_state_dict (checkpoint ['state_dict'])
    model.class_to_idx = checkpoint ['mapping']

    for param in model.parameters():
        param.requires_grad = False #turning off tuning of the model

    return model

def process_image(image):
    ''' Function to process a PIL image.
        Scales, crops, and normalizes a PIL 
        image for a PyTorch model,
        returns an Numpy array
    '''
    im = Image.open (image) #loading image
    width, height = im.size #original size

    # smallest part: width or height should be kept not more than 256
    if width > height:
        height = 256
        im.thumbnail ((50000, height), Image.ANTIALIAS)
    else:
        width = 256
        im.thumbnail ((width,50000), Image.ANTIALIAS)

    width, height = im.size #new size of im
    #crop 224x224 in the center
    reduce = 224
    left = (width - reduce)/2
    top = (height - reduce)/2
    right = left + 224
    bottom = top + 224
    im = im.crop ((left, top, right, bottom))

    #preparing numpy array
    np_image = np.array (im)/255 #to make values from 0 to 1
    np_image -= np.array ([0.485, 0.456, 0.406])
    np_image /= np.array ([0.229, 0.224, 0.225])

    #PyTorch expects the color channel to be the first dimension but it's the third dimension in the PIL image and Numpy array.
    #The color channel needs to be first and retain the order of the other two dimensions.
    np_image= np_image.transpose ((2,0,1))
    return np_image


def predict(image_path, model, topkl, device):
    ''' Function to Predict the classes of an image 
        using a trained deep learning model.
    '''
    # Load image path to process_image function
    image = process_image (image_path)

    if device == 'cuda':
        im = torch.from_numpy (image).type (torch.cuda.FloatTensor)
    else:
        im = torch.from_numpy (image).type (torch.FloatTensor)

    # Code used to make size of torch image as expected. 
    # Doing that we will have batch size = 1
    im = im.unsqueeze (dim = 0) 
   

    # Send device to model
    model.to (device)
    im.to (device)

    with torch.no_grad ():
        output = model.forward (im)
    output_prob = torch.exp (output) # Probabilities

    probs, indeces = output_prob.topk (topkl)
    probs = probs.cpu ()
    indeces = indeces.cpu ()
    
    # Convert tensors array to numpy array
    probs = probs.numpy () 
    indeces = indeces.numpy ()

    # Convert numpy array to list
    probs = probs.tolist () [0] 
    indeces = indeces.tolist () [0]

    mapping = {val: key for key, val in
                model.class_to_idx.items()
                }
    
    # Match up keys:values defined in mapping (JSON File)
    classes = [mapping [item] for item in indeces]

    # Convert to numpy arrays
    classes = np.array (classes) 

    return probs, classes

# Set file path
args = parser.parse_args ()
file_path = args.image_dir

# Setup device to be used with tensors (cuda or cpu)
if args.GPU == 'GPU':
    device = 'cuda'
else:
    device = 'cpu'

# Loading mappiing JSON file if provided, else load default file name
if args.category_names:
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)
else:
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
        pass

# Load model from checkpoint
model = loading_model (args.load_dir)

# If statement to define number of predicted classes. Default = 1
if args.top_k:
    nm_cl = args.top_k
else:
    nm_cl = 1

# Use predict function to calculate probabilies, classes
# number of predicted classes and device. 
probs, classes = predict (file_path, model, nm_cl, device)

# Map names using JSON File (dict) cat_to_name.json (dafault)
class_names = [cat_to_name [item] for item in classes]

for l in range (nm_cl):
     print("Number: {}/{}.. ".format(l+1, nm_cl),
            "Class name: {}.. ".format(class_names [l]),
            "Probability: {:.3f}..% ".format(probs [l]*100),
            )