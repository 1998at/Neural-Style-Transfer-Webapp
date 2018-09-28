# -*- coding: utf-8 -*-
"""
Created on  June 17 10:51:17 2018

@author: Ayush
"""

from flask import Flask, render_template, request

import imageio
import requests


import os
import sys
import time
import numpy as np
from tqdm import tqdm, trange

import torch
from torch.optim import Adam
from torch.autograd import Variable
from torch.utils.data import DataLoader

from torchvision import datasets
from torchvision import transforms

import utils
from net import Net, Vgg16

from option import Options

import json
import os

app =Flask(__name__)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))


def find_pos(image):
    pass

def evaluate(content):
    print("enter a style number between 1 and 21")
    i=4##Choosing from a style number from available 21 styles
    content_image = utils.tensor_load_rgbimage(content, size=512, keep_asp=True)
    content_image = content_image.unsqueeze(0)
    list_of_styles=['candy.jpg',
                'composition_vii.jpg',
                'escher_sphere.jpg',
                'feathers.jpg',
                'frida_kahlo.jpg',
                'la_muse.jpg',
                'mosaic.jpg',
                'mosaic_ducks_massimo.jpg',
                'pencil.jpg',
                'picasso_selfport1907.jpg',
                'rain_princess.jpg',
                'Robert_Delaunay,_1906,_Portrait.jpg',
                'seated-nude.jpg',
                'shipwreck.jpg',
                'starry_night.jpg',
                'stars2.jpg',
                'strip.jpg','the_scream.jpg','udnie.jpg','wave.jpg','woman-with-hat-matisse.jpg']


    style = utils.tensor_load_rgbimage("images/21styles/"+list_of_styles[i-1], size=512)
    style = style.unsqueeze(0)    
    style = utils.preprocess_batch(style)
#    print("content_image={},   size={},    ".format(args.content_image,args.content_size))
#    print("style_imagee={},   style_size={},    ".format(args.style_image,args.style_size))
#    
    

    style_model = Net(ngf=128)
    #print("ngf is {}".format(args.ngf))
    style_model.load_state_dict(torch.load("21styles.model"), False)
   # print(args.model)
    gpu_support=0
    if (torch.cuda.is_available()):
        gpu_support=1

        print("Gpu Support Detected")
        style_model.cuda()
        content_image = content_image.cuda()
        style = style.cuda()

    style_v = Variable(style)

    content_image = Variable(utils.preprocess_batch(content_image))
    style_model.setTarget(style_v)

    output = style_model(content_image)
    #output = utils.color_match(output, style_v)
    a=content.split("//")[1]
    utils.tensor_save_bgrimage(output.data[0], "static/"+a, gpu_support)
    return a


@app.route('/')
def hello_world():
    return render_template('index.html')

@app.route("/predict", methods=["POST"])
def predict():    
    target = os.path.join(APP_ROOT, 'static/')
    print(target)
    if not os.path.isdir(target):
            os.mkdir(target)
    else:
        print("Couldn't create upload directory: {}".format(target))
    print(request.files.getlist("file"))
    for upload in request.files.getlist("file"):
        print(upload)
        print("{} is the file name".format(upload.filename))
        filename = upload.filename
        destination = "/".join([target, filename])
        print ("Accept incoming file:", filename)
        print ("Save it to:", destination)
        upload.save(destination)
    ##the above code is to take the input image from the html form

    filename=evaluate(destination)
    print(destination)
    return render_template("results.html",image_name=filename)
    
'''main function to run'''    
if __name__ == "__main__":
    print(("Loading"))
    
    app.run()
