# Define imports 
import torch
from torch import nn
from torch import optim
from torchvision import datasets, models, transforms
import torch.utils.data
from collections import OrderedDict
import argparse
import json
import urllib

# Define Arguments using argparse (https://docs.python.org/3/library/argparse.html)
parser = argparse.ArgumentParser (description = "Parser for train.py")
parser.add_argument ('data_dir', help = 'Input data directory path (example: /flowers/dir)', type = str)
parser.add_argument ('--save_dir', help = 'Input saving directory (optional)', type = str)
parser.add_argument ('--arch', help = 'Input arch (example: vgg13) otherwise Alexnet = default', type = str)
parser.add_argument ('--lrn', help = 'Learning rate (default value 0.001)', type = float)
parser.add_argument ('--hidden_units', help = 'Hidden units in Classifier (example: 512) otherwise default = 2048', type = int)
parser.add_argument ('--epochs', help = 'Number of epochs (example: 20)', type = int)
parser.add_argument ('--GPU', help = "Option to use GPU", type = str)

# Set directory locations
args = parser.parse_args()
data_dir = args.data_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# Setup device to be used with tensors (cuda or cpu)
if args.GPU == 'GPU':
    device = 'cuda'
else:
    device = 'cpu'

# Load Data 
if data_dir:
    # Define your transforms for the training, validation, and testing sets
    train_data_transforms = transforms.Compose ([transforms.RandomRotation (30),
                                                transforms.RandomResizedCrop (224),
                                                transforms.RandomHorizontalFlip (),
                                                transforms.ToTensor (),
                                                transforms.Normalize ([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                                                ])

    valid_data_transforms = transforms.Compose ([transforms.Resize (255),
                                                transforms.CenterCrop (224),
                                                transforms.ToTensor (),
                                                transforms.Normalize ([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                                                ])

    test_data_transforms = transforms.Compose ([transforms.Resize (255),
                                                transforms.CenterCrop (224),
                                                transforms.ToTensor (),
                                                transforms.Normalize ([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                                                ])
    # Load the datasets with ImageFolder as done in image classifier project
    train_image_datasets = datasets.ImageFolder (train_dir, transform = train_data_transforms)
    valid_image_datasets = datasets.ImageFolder (valid_dir, transform = valid_data_transforms)
    test_image_datasets = datasets.ImageFolder (test_dir, transform = test_data_transforms)

    # Using the image datasets and the trainforms, define the dataloaders
    train_loader = torch.utils.data.DataLoader(train_image_datasets, batch_size = 64, shuffle = True)
    valid_loader = torch.utils.data.DataLoader(valid_image_datasets, batch_size = 64, shuffle = True)
    test_loader = torch.utils.data.DataLoader(test_image_datasets, batch_size = 64, shuffle = True)
    

# Mapping of label to category name (image classifier project for reference)
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

def load_model (arch, hidden_units):
    ''' Statments that set up model architectures based on input args 
    --arch (example: vgg13), --hidden (example: 512), --epoch (example 20)'''

    if arch == 'vgg13': # vgg13 model arch
        model = models.vgg13 (pretrained = True)
        for param in model.parameters():
            param.requires_grad = False #Set gradients to false
        if hidden_units: # If statement if hidden unit args was provided
            classifier = nn.Sequential  (OrderedDict ([
                            ('fc1', nn.Linear (25088, 4096)),
                            ('relu1', nn.ReLU ()),
                            ('dropout1', nn.Dropout (p = 0.3)),
                            ('fc2', nn.Linear (4096, hidden_units)),
                            ('relu2', nn.ReLU ()),
                            ('dropout2', nn.Dropout (p = 0.3)),
                            ('fc3', nn.Linear (hidden_units, 102)),
                            ('output', nn.LogSoftmax (dim =1))
                            ]))
        else: # Set to default architecture
            classifier = nn.Sequential  (OrderedDict ([
                        ('fc1', nn.Linear (25088, 4096)),
                        ('relu1', nn.ReLU ()),
                        ('dropout1', nn.Dropout (p = 0.3)),
                        ('fc2', nn.Linear (4096, 2048)),
                        ('relu2', nn.ReLU ()),
                        ('dropout2', nn.Dropout (p = 0.3)),
                        ('fc3', nn.Linear (2048, 102)),
                        ('output', nn.LogSoftmax (dim =1))
                        ]))
    else: # Setup default model Alexnet (Refer to project 1 for details on use of Alexnet)
        arch = 'alexnet' 
        model = models.alexnet (pretrained = True)
        for param in model.parameters():
            param.requires_grad = False # Set gradients to false
        if hidden_units: # If statement if hidden_units arg given
            classifier = nn.Sequential  (OrderedDict ([
                            ('fc1', nn.Linear (9216, 4096)),
                            ('relu1', nn.ReLU ()),
                            ('dropout1', nn.Dropout (p = 0.3)),
                            ('fc2', nn.Linear (4096, hidden_units)),
                            ('relu2', nn.ReLU ()),
                            ('dropout2', nn.Dropout (p = 0.3)),
                            ('fc3', nn.Linear (hidden_units, 102)),
                            ('output', nn.LogSoftmax (dim =1))
                            ]))
        else: # If hidden_units arg not given
            classifier = nn.Sequential  (OrderedDict ([
                        ('fc1', nn.Linear (9216, 4096)),
                        ('relu1', nn.ReLU ()),
                        ('dropout1', nn.Dropout (p = 0.3)),
                        ('fc2', nn.Linear (4096, 2048)),
                        ('relu2', nn.ReLU ()),
                        ('dropout2', nn.Dropout (p = 0.3)),
                        ('fc3', nn.Linear (2048, 102)),
                        ('output', nn.LogSoftmax (dim =1))
                        ]))
    # Set model to classifier used
    model.classifier = classifier
    return model, arch

# Validation Function
def validation(model, valid_loader, criterion):
    model.to (device)

    valid_loss = 0
    accuracy = 0
    for inputs, labels in valid_loader:

        inputs, labels = inputs.to(device), labels.to(device)
        output = model.forward(inputs)
        valid_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()

    return valid_loss, accuracy

# Load Model
model, arch = load_model (args.arch, args.hidden_units)

# Train Model
# Setup Criterion Loss Function
criterion = nn.NLLLoss ()

# If statement to setup optimizer based on default or user input arg --learnrate
if args.lrn: #if learning rate was provided
    optimizer = optim.Adam (model.classifier.parameters (), lr = args.lrn)
else:
    optimizer = optim.Adam (model.classifier.parameters (), lr = 0.001)

# Send device to model (cuda or cpu)
model.to (device)

# If statement to setup epochs based on default or user input arg --epochs
if args.epochs:
    epochs = args.epochs
else:
    epochs = 7

print_every = 40
steps = 0

# For statement to run throught epochs
for e in range (epochs):
    running_loss = 0
    for ii, (inputs, labels) in enumerate (train_loader):
        steps += 1
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad ()

        # Forward and backward passes
        outputs = model.forward (inputs) 
        loss = criterion (outputs, labels) 
        loss.backward ()
        optimizer.step () 
        running_loss += loss.item ()

        if steps % print_every == 0:
            
            # Set model to eval mode to turn dopouts
            model.eval ()

            with torch.no_grad():
                valid_loss, accuracy = validation(model, valid_loader, criterion)

            print("Epoch: {}/{}.. ".format(e+1, epochs),
                  "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                  "Valid Loss: {:.3f}.. ".format(valid_loss/len(valid_loader)),
                  "Valid Accuracy: {:.3f}%".format(accuracy/len(valid_loader)*100))

            running_loss = 0
            
            # Set model back to train mode
            model.train()

# Send model to CPU in effort to save/load models and manipulate tensors
model.to ('cpu')

# Saving mapping between predicted class and class name note class names will be ints of subarrays
model.class_to_idx = train_image_datasets.class_to_idx #


# Dictionary to save model key and values (refer to image classifier project for reference)
checkpoint = {'classifier': model.classifier,
              'state_dict': model.state_dict (),
              'arch': arch,
              'mapping':    model.class_to_idx
             }

# If statement to setup checkpoint path
# To be used with predict.py file. 
if args.save_dir:
    torch.save (checkpoint, args.save_dir + '/checkpoint.pth')
else:
    torch.save (checkpoint, 'checkpoint.pth')