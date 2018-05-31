import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torchvision import datasets, transforms, models
import numpy as np
import os
import os.path
import argparse
from collections import OrderedDict
import sys
import torch.nn.functional as F


#Command arguments 
#python train.py --data_directory flowers  --arch densenet  --gpu yes  --lr 0.001  --epochs 2  --hidden_units 500 --checkpoint checkpoint.pth



def load_the_model(hidden_units=500, arch="densenet"):

    if arch=='densenet':
        model = models.densenet121(pretrained=True)
        inputs = 1024
    elif arch=='vgg':
        model = models.vgg19(pretrained=True)
        inputs = 25088
    else:
        print(str(arch) + " is not supported. Please try again")
        sys.exit()
        
    for param in model.parameters():
        param.requires_grad = False
    
    output = 102
    classifier = nn.Sequential(OrderedDict([ 
        ('fc1', nn.Linear(inputs, hidden_units)),
        ('relu1', nn.ReLU()), 
        ('fc2', nn.Linear(hidden_units, hidden_units)),
        ('relu2', nn.ReLU()), 
        ('fc3', nn.Linear(hidden_units, output)), 
        ('output', nn.LogSoftmax(dim=1))]))
  
    model.classifier = classifier
    return model


def train_model(data_sets_array, checkpoint="", arch="densenet", hidden_units=500, lr=0.001, gpu=False, epochs=3):
    
    if args.arch:
        arch = args.arch     
        
    if args.hidden_units:
        hidden_units = args.hidden_units 

    if args.epochs:
        epochs = args.epochs
            
    if args.lr:
        lr = args.lr

    if args.gpu:
        gpu = args.gpu

    if args.checkpoint:
        checkpoint = args.checkpoint   
        
        
    trainloader = data_sets_array[0]
    validateloader = data_sets_array[1]
    
    #print the user values!!!
    print('Network architecture :', arch)
    print('Learning rate :', lr)
    print('Number of epochs :', epochs)
    print('Number of hidden units :', hidden_units)
    print("The number of classes :", len(trainloader))
    print("Does the User want GPU :", gpu)
    print("Is GPU Available :", torch.cuda.is_available())
    
   
    model = load_the_model(hidden_units, arch)
  
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=lr)
    
   
    if gpu and torch.cuda.is_available():
        model.cuda()
        print("Running on GPU!")
    else :
        model.cpu()
        print("Running on CPU!")
        
    steps = 0 
    running_loss = 0 
    print_every = 40
    
    for e in range(epochs):

        model.train()        

        for ii, (inputs, labels) in enumerate(trainloader):
            inputs, labels = Variable(inputs), Variable(labels)
            steps = steps + 1

            optimizer.zero_grad()

            
            if gpu and torch.cuda.is_available():
                inputs, labels = inputs.cuda(), labels.cuda()
            else:
                inputs, labels = inputs.cpu(), labels.cpu()

        
            output = model.forward(inputs)
            
            loss = criterion(output, labels)
            loss.backward()
        
            optimizer.step()
            running_loss = running_loss + loss.data[0]

            if steps % print_every == 0:
                model.eval()
                accuracy = 0
                test_loss = 0
                for ii, (inputs, labels) in enumerate(validateloader):

                    if gpu and torch.cuda.is_available():
                        inputs, labels = inputs.cuda(), labels.cuda()
                    else:
                        inputs, labels = inputs.cpu(), labels.cpu()
                    inputs, labels = Variable(inputs), Variable(labels)
                    output = model.forward(inputs)
                    test_loss += criterion(output, labels).data[0]

                    ps = torch.exp(output).data
                    equality = (labels.data == ps.max(1)[1])
                
                    accuracy += equality.type_as(torch.FloatTensor()).mean()

                print("Epoch: {}/{}.. ".format(e+1, epochs),
                    "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                    "Validation Loss: {:.3f}.. ".format(test_loss/len(validateloader)),
                    "Validation Accuracy: {:.3f}".format(accuracy/len(validateloader)))
                running_loss = 0
                model.train()
                
    if checkpoint:
        
        def find_classes(dir):
            classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
            classes.sort()
            class_to_idx = {classes[i]: i for i in range(len(classes))}
            return class_to_idx

        state = {
            'epochs': epochs,
            'arch': arch,
            'lr': lr,
            'hidden_units': hidden_units,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'class_to_idx': find_classes(args.data_directory + '/train')
        }
        print ('Saving checkpoint to:', checkpoint) 
        torch.save(state, checkpoint)

                
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        
# Define command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str2bool, help='Use GPU (if available) ')
parser.add_argument('--epochs', type=int, help='Number of epochs')
parser.add_argument('--lr', type=float, help='Learning rate')
parser.add_argument('--arch', type=str, help='Model architecture to use')
parser.add_argument('--hidden_units', type=int, help='Number of hidden units in the model')
parser.add_argument('--data_directory', type=str, help='Location of dataset ')
parser.add_argument('--checkpoint', type=str, help='Checkpoint file path name')


args, _ = parser.parse_known_args()

if args.data_directory:
    data_dir = args.data_directory
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    
    data_transforms = transforms.Compose([transforms.RandomRotation(30),
                                          transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(224),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])])


    validation_transforms = transforms.Compose([transforms.Resize(224),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406], 
                                                                     [0.229, 0.224, 0.225])])

    train_data = datasets.ImageFolder(train_dir, transform=data_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
    validate_data = datasets.ImageFolder(valid_dir, transform=validation_transforms)
    

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64)
    validateloader = torch.utils.data.DataLoader(validate_data, batch_size=32)
    
    #[0] = Train Data | [1] = Validate Data
    data_sets = [trainloader, validateloader]
    
    
    train_model(data_sets)

    


