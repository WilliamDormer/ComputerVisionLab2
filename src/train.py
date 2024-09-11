#this file is for the purpose of training the weights for the decoder of the AdaIN network.

#it takes the input format specified in my_train.txt, specifically reading from my_train.txt

# following this tutorial: https://github.com/naoto0804/pytorch-AdaIN/blob/master/train.py

import torch
import matplotlib.pyplot as plt
import argparse
import datetime
from torchvision import transforms
from torch.utils import data
import numpy as np
from pathlib import Path
from PIL import Image, ImageFile
from tqdm import tqdm #progress bar
from tensorboardX import SummaryWriter
import os
import math
import AdaIN_net
import sys

#Dataset class from the github implementation of AdaIN
class FlatFolderDataset(data.Dataset):
    def __init__(self, root, transform):
        super(FlatFolderDataset, self).__init__()
        self.root = root
        self.paths = list(Path(self.root).glob('*'))
        self.transform = transform

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(str(path)).convert('RGB')
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.paths)

    def name(self):
        return 'FlatFolderDataset'

#important check to enable multithreading with python on windows.
if __name__ == '__main__':

    #use argparse to get the inputs from the command line

    #Checks if the input directories are valid, code I found at: https://stackoverflow.com/questions/38834378/path-to-a-directory-as-argparse-argument
    def dir_path(string):
        if os.path.isdir(string) or os.path.isfile(string):
            return string
        else:
            if os.path.isdir(string) == False:
                raise NotADirectoryError(string)
            else:
                raise FileNotFoundError(string)


    # format for the train.py file:
    # -content_dir first argument is the location of the content images to train on
    # -style_dir second argument is the location of the style images to train on
    # -gamma is the gamma value
    # -e is the epochs
    # -b is the batch size
    # -l is the path to the encoder weights
    # -s is the path we are going to save the decoder weights to.
    # -p is the output image destination
    # -cuda tells whether to use CUDA (Y) or not (N)

    parser = argparse.ArgumentParser()
    parser.add_argument("-content_dir", "--content_dir", default="../images/content", type=dir_path, help="")
    parser.add_argument("-style_dir","--style_dir",default= "../images/style", type=dir_path, help="")
    parser.add_argument("-gamma","--gamma", type=float,default=1.0, help="")
    parser.add_argument("-e","--epochs", type=int,default=3, help="")
    parser.add_argument("-b","--batch_size", type=int, default=8, help="")
    parser.add_argument("-l","--encoder_path", default="../encoder.pth", type=dir_path, help="")
    # parser.add_argument("-s","--decoder_path", type=str, default="decoder.pth", help="") #need to figure out when to use this
    # parser.add_argument("-p","--output_image_path", type=str,default="decoder.png", help="") #not in use, using tensorboard
    parser.add_argument("-cuda","--cuda",default="Y", type=str, help="")

    parser.add_argument('--save_dir', default='./experiments', help='Directory to save the model') #versus when to use this.
    parser.add_argument("--lr", type=float, default=1e-4, help="")
    parser.add_argument('--lr_decay', type=float, default=3e-4) #5e-5 was recommended by the github.
    parser.add_argument('--n_threads', type=int, default=1) #not in use
    parser.add_argument('--save_model_interval', type=int, default=10)
    parser.add_argument('--log_dir', default='./logs', help='Directory to save the log')

    parser.add_argument("--experiment_name", default="decoder",type=str)

    parser.add_argument('-v', "--verbose", default="N", help="For verbose output (print statements).")
    args = parser.parse_args()

    #just some input checking
    if args.cuda != "Y" and args.cuda != "N":
        raise Exception("The input to -cuda must be either Y or N")

    print(f"running training with the following parameters:")
    print("gamma: ", args.gamma)
    print("epochs: ", args.epochs)
    print("batch size: ", args.batch_size)
    print('learning rate: ', args.lr)
    print("learing rate decay: ", args.lr_decay)

    #other hyperparameters

    #get the model structure from the other module
    decoder = AdaIN_net.encoder_decoder.decoder
    encoder = AdaIN_net.encoder_decoder.encoder

    if args.verbose == "Y":
        print("encoder path from argument list: ", args.encoder_path)

    #get the encoder from the saved file.
    encoder.load_state_dict(torch.load(args.encoder_path)) #load the encoder weights.
    #construct the full model with the full encoder, and the empty decoder.
    network = AdaIN_net.AdaIN_net(encoder, decoder)

    if args.verbose == "Y":
        print("type of 'network' : ", type(network))

    #set up the summary writer
    log_dir = Path(args.log_dir)
    writer = SummaryWriter(log_dir=str(log_dir)+"/"+args.experiment_name)

    #set it to train mode.
    network.train()

    #select the device based on input:
    device = None
    if torch.cuda.is_available() and args.cuda == "Y":
        device = torch.device("cuda")
        torch.cuda.set_per_process_memory_fraction(0.9)
    else:
        device = torch.device("cpu")

    print("device: ", device)
    network.to(device)






    #loading data for training, and creating data loaders

    #performs transformation of the images
    def train_transform(): #need to verify what this function does.
        transform_list = [
            transforms.Resize(size = (512,512)), #resize the images the the expected size.
            transforms.RandomCrop(256), #this gives us a random crop of the image, so that we can make a variety of different images from the same image.
            transforms.ToTensor() #convert the output to a tensor.
        ]
        return transforms.Compose(transform_list)

    content_tf = train_transform()
    style_tf = train_transform()

    #helper sampling functions provided by the github AdaIN implementation
    def InfiniteSampler(n):
        i = n - 1
        order = np.random.permutation(n)
        while True:
            yield order[i]
            i += 1
            if i >= n:
                np.random.seed()
                order = np.random.permutation(n)
                i = 0

    class InfiniteSamplerWrapper(data.sampler.Sampler):
        def __init__(self, data_source):
            self.num_samples = len(data_source)

        def __iter__(self):
            return iter(InfiniteSampler(self.num_samples))

        def __len__(self):
            return 2 ** 31

    #construct the dataset from the directories pointed by the arguments.
    content_dataset = FlatFolderDataset(args.content_dir, content_tf)
    style_dataset = FlatFolderDataset(args.style_dir, style_tf)

    #define the iterators for the datasets for both style and content to be able to feed both at a time.
    content_iter = iter(data.DataLoader(
        content_dataset,
        batch_size = args.batch_size,
        sampler = InfiniteSamplerWrapper(content_dataset),
        num_workers=args.n_threads
    ))
    style_iter = iter(data.DataLoader(
        style_dataset,
        batch_size=args.batch_size,
        sampler=InfiniteSamplerWrapper(style_dataset),
        num_workers=args.n_threads
    ))

    #manual scheduler, as shown in the github.
    def adjust_learning_rate(optimizer, iteration_count):
        lr = args.lr / (1.0 + args.lr_decay * iteration_count)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    #define optimizer
    optimizer = torch.optim.Adam(network.decoder.parameters(), lr=args.lr)

    #define training behaviour:
    def train():
        #start timer of when the training starts:
        start_time = datetime.datetime.now()
        # print("in train function")

        #the number of batches per epoch.
        n_batches = int(math.floor(len(content_dataset) / args.batch_size))

        for i in tqdm(range(args.epochs)):
            #adjust the learning rate of the optimizer.
            adjust_learning_rate(optimizer, iteration_count=i)
            for batch in range(n_batches):

                #send the images to the GPU.
                content_images = next(content_iter).to(device) #this will crash if I use too large a batch size.
                style_images = next(style_iter).to(device)

                #disable gradient update
                optimizer.zero_grad()
                #calculate the losses
                loss_c, loss_s = network.forward(content_images, style_images)

                #apply the gamma to the style loss.
                loss_s_altered = args.gamma * loss_s
                #so then we get our adjusted total loss.
                loss = loss_c + loss_s_altered

                #backprop.
                loss.backward()
                optimizer.step()

                #save the original style loss, the content loss and the total loss to the loss plots with tensorboardx
                writer.add_scalar("loss_content", loss_c.item(), i+1)
                writer.add_scalar("loss_style", loss_s.item(), i+1)
                writer.add_scalar("loss_total", loss.item(), i+1)

            #save the model logic.
            if (i + 1) % args.save_model_interval == 0 or (i+1) == args.epochs:
                state_dict = network.decoder.state_dict()
                for key in state_dict.keys():
                    state_dict[key] = state_dict[key].to(torch.device('cpu'))
                torch.save(state_dict, args.save_dir + '/decoder_' + args.experiment_name + '_iter_{:d}.pth.tar'.format(i+1))

        writer.close()


        #simply print out the time elapsed. TQDM also gives a good output for this, so this part isn't really needed
        end_time = datetime.datetime.now()
        print("time elapsed: ", end_time - start_time)
        pass

    #finally, run the training
    train()






