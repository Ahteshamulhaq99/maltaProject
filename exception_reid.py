from efficientnet_pytorch import EfficientNet
import numpy as np
from PIL import Image
from torchvision import transforms
from torch import nn
import cv2
#------------------------------EfficientNetB3---------------
# Load the model
modeleff = EfficientNet.from_pretrained('efficientnet-b3')
modeleff.eval()

##########################personException################
# person2 = Image.open("exception/man.jpg")
# person3 = Image.open("exception/man_second.jpg")
# person4 = Image.open("exception/man_third.jpg")
# # Note: images have to be of equal size

# person2 = person2.resize((300,300))
# person3 = person3.resize((300,300))
# person4 = person4.resize((300,300))
            
# # Convert the images to tensors

# person2_tensor = transforms.ToTensor()(person2)
# person3_tensor = transforms.ToTensor()(person3)
# person4_tensor = transforms.ToTensor()(person4)
        
# # Add a fourth dimension for the batch and extract the features 

# personfeatures2 = modeleff.extract_features(person2_tensor.unsqueeze(0))
# personfeatures3 = modeleff.extract_features(person3_tensor.unsqueeze(0))
# personfeatures4 = modeleff.extract_features(person4_tensor.unsqueeze(0))



def person_exception(crop_obj):
    image1 = cv2.cvtColor(crop_obj, cv2.COLOR_BGR2RGB)
    image1 = Image.fromarray(image1)
    image1 = image1.resize((300,300))
    image1_tensor = transforms.ToTensor()(image1)
    features1 = modeleff.extract_features(image1_tensor.unsqueeze(0))
    cos = nn.CosineSimilarity(dim=0)
    value = round(float(cos(features1.reshape(1, -1).squeeze(), \
        personfeatures2.reshape(1, -1).squeeze())),4)
    value3 = round(float(cos(features1.reshape(1, -1).squeeze(), \
        personfeatures3.reshape(1, -1).squeeze())),4)
    value4 = round(float(cos(features1.reshape(1, -1).squeeze(), \
        personfeatures4.reshape(1, -1).squeeze())),4)

    if value>0.55:
        return True
    elif value3>0.75:
        return True
    elif value4>0.87:
        return True
        #..................................................................
    print("Person Not Matched!!!!")
    return False