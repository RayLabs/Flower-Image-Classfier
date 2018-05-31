import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from torch import optim
from train import load_the_model
import json
import argparse

def load(checkpoint):
    arch = checkpoint['arch'] 
    lr = checkpoint['lr']
    class_to_idx = checkpoint['class_to_idx']
    hidden_units = checkpoint['hidden_units'] 
    state_dict = checkpoint['state_dict']
    optimizer_dict = checkpoint['optimizer']
    epochs = checkpoint['epochs']
    
    model = load_the_model(hidden_units, arch)
    optimizer = optim.Adam(model.classifier.parameters(), lr=lr)
    model.load_state_dict(state_dict)
    model.class_to_idx = class_to_idx
    optimizer.load_state_dict(optimizer_dict) 
    return model
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    h = 112
    img = Image.open(image)
    
    if img.size[0]>=img.size[1]:
        img.resize((10000,256)) 
    else: 
        img.resize((256,10000))

    hw = img.size[0] / 2
    hh = img.size[1] / 2
    img = img.crop( (hw - h, hh - h, hw + h, hh + h) )

    img = np.array(img)/255
    img = ( img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225] 

    img = img.transpose((2,0,1))
    return torch.from_numpy(img)
def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
   
    output = model.forward(Variable(image_path, volatile=True))
    ps = torch.exp(output)
    
    result = torch.topk(ps, topk)
    
    return result

#Command
#python predict.py --checkpoint checkpoint.pth --image flowers/test/8/image_03299.jpg --topk 5  --labels cat_to_name.json --gpu yes

def predict_image(image, checkpoint_path, topk=1, labels='', gpu=False):
    if args.image:
        image = args.image     
        
    if args.checkpoint:
        checkpoint_path = args.checkpoint

    if args.topk:
        topk = args.topk
            
    if args.labels:
        labels = args.labels

    if args.gpu:
        gpu = args.gpu
 
 
    checkpoint = torch.load(checkpoint_path)

    model = load(checkpoint)
    
    model.eval()
    
    img = process_image(image)
    img2 = img.float()
    img3 = img2.clone()
    img3 = img3.resize_(1, 3, 224, 224)
    if gpu and torch.cuda.is_available():
        model.cuda()
        img3 = img3.cuda()
        print("Running on GPU!")
    else:
        model.cpu()
        img3.cpu()
        print("Running on CPU!")

    result = predict(img3, model, topk)
   
    classes = result[1].cpu().data.numpy()[0]
    probs = torch.nn.functional.softmax(Variable(result[0].data), dim=1)
    class_to_idx = model.class_to_idx
    inv_map = {v: k for k, v in class_to_idx.items()}

    if args.labels:
        with open(labels, 'r') as f:
            cat_to_name = json.load(f)
        top_class_names = [cat_to_name[inv_map[classes[x]]] for x, y in enumerate(classes) ]
        print(top_class_names)
        print(probs)
    else:
        top_class =[ inv_map[classes[x]] for x, y in enumerate(classes)]
        print(top_class)
        print(probs)
    
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
# Define command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str2bool, help='Use GPU if available')
parser.add_argument('--checkpoint', type=str, help='Model checkpoint to use when predicting the image')
parser.add_argument('--topk', type=int, help='Return top K predictions')
parser.add_argument('--image', type=str, help='Image to predict')
parser.add_argument('--labels', type=str, help='JSON file path that contains label names')


args, _ = parser.parse_known_args()

if args.checkpoint and args.image:
    checkpoint_path = args.checkpoint
    predict_image(args.image, checkpoint_path)
