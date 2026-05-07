import torch

IMAGE_WIDTH = 512
IMAGE_HEIGHT = 256
BATCH_SIZE = 8
LEARNING_RATE = 5e-4
DEVICE = "cuda"
ROOT_DIR = "/content/drive/MyDrive/Lane_Detection_Project/data/tusimple" #inside tusimple -> train_set , test_set , val.txt , train.txt 


#The class weights are calculated based on the paper.
#Paper's formula
#c = 1.02
#w_bg   = 1.0 / np.log(c + p_bg)
#w_lane = 1.0 / np.log(c + p_lane) by using all train set and saved into config.py as CLASS_WEIGHTS.
#We use this special weigthing instead of just giving some numbers like 1 and 10 because these numbers are calculated 
#based on the actual distribution of the classes in the training data, which can help the model learn better 
#by giving more importance to the minority class (lane markings) and less importance to the majority class 
#(background)
CLASS_WEIGHTS = torch.tensor([1.4506, 21.5162])

EPOCHS = 75