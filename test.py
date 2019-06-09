import nnModel
import torchvision
from torchvision import datasets, transforms, models
import helpers.JsonLoader

# The purpose of this file is to load the saved model and test it on different images

# Load the model
model = models.vgg19(pretrained=True)
model = nnModel.load_model('./myModelCheckpoint.pth', model)

# Load the content of the json file
cat_to_name = helpers.JsonLoader.load_json('cat_to_name.json')

# Sanity check
nnModel.sanity_check(cat_to_name, 'flowers/valid/100/image_07917.jpg', model, 100)