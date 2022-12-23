import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import json
import torch
import imutils
import numpy as np
import pandas as pd 
from PIL import Image
from matplotlib import pyplot as plt
from deuces.Card import Card
from deuces.Evaluator import Evaluator
import string, random

import telebot

DIR_TEMP = "templates"

API_TOKEN = '5890609645:AAE_7F-Qvjxe2h6U4ePgOHjp-BaryMx8qYY'
bot = telebot.TeleBot(API_TOKEN)

def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3

def preprocess(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),2 )
    thresh = cv2.adaptiveThreshold(blur,255,1,1,11,1)
    blur_thresh = cv2.GaussianBlur(thresh,(5,5),5)
    return blur_thresh

def detect_cards(im, num = None):
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(1,1),1000)
    flag, thresh = cv2.threshold(blur, 200, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea,reverse=True)#[:numcards] 
    prev_list = []
    for i, elem in enumerate(contours):
        ar = cv2.contourArea(elem)
        if i == 0:
            pass
        else:
            prev_list.append( abs((ar - prev)/(prev+0.01)))
        prev = ar
        if i == 20:
            break
    for i, elem in enumerate(prev_list):
        if elem > 0.25:
            numcards = i+1
            break
    #print(prev_list)
    if num != None:
        numcards = num   
    #print(str(numcards) + " cards detected")
    r_all = []
    for i in range(numcards):
        card = contours[i]
        peri = cv2.arcLength(card,True)
        approx = cv2.approxPolyDP(card,0.02*peri,True)
        rect = cv2.minAreaRect(contours[i])
        r = cv2.boxPoints(rect)
        r_all.append(r)
    r_sort = []
    for r in r_all:
        approx = np.array(r, dtype = np.float32)
        i = approx
        x_s = sorted(i, key=lambda x: x[1])
        y_s = sorted(i, key=lambda x: x[0])
        x_s = [list(i) for i in x_s]
        y_s = [list(i) for i in y_s]
        ll = intersection(x_s[:2], y_s[:2])
        lu = intersection(x_s[:2], y_s[2:])
        rl = intersection(x_s[2:], y_s[:2])
        ru = intersection(x_s[2:], y_s[2:])
        ordered = [ll[0], lu[0], rl[0], ru[0]]
        ordered = np.array(ordered)
        r_sort.append(ordered)
    detected = []
    for locs in r_sort:
        h = np.array([[0,0],[199, 0], [0,299],[199,299]],np.float32)
        transform = cv2.getPerspectiveTransform(locs,h)
        warp = cv2.warpPerspective(im,transform,(200,300))
        corr = []
        names = []
        for i in os.listdir(DIR_TEMP):
            if i[0] == ".":
                continue
            template = cv2.imread(DIR_TEMP + "/" + i)
            diff = cv2.absdiff(preprocess(warp),preprocess(template))  
            diff = cv2.GaussianBlur(diff,(5,5),5)    
            flag, diff = cv2.threshold(diff, 200, 255, cv2.THRESH_BINARY) 
            corr.append((np.sum(diff), i))
        warp = cv2.rotate(warp, cv2.ROTATE_180)
        for i in os.listdir(DIR_TEMP):
            if i[0] == ".":
                continue
            template = cv2.imread(DIR_TEMP + "/" + i)
            diff = cv2.absdiff(preprocess(warp),preprocess(template))  
            diff = cv2.GaussianBlur(diff,(5,5),5)    
            flag, diff = cv2.threshold(diff, 200, 255, cv2.THRESH_BINARY) 
            corr.append((np.sum(diff), i))
        warp = cv2.rotate(warp, cv2.ROTATE_180)
        ans = sorted(corr,key = lambda x: x[0])
        detected.append(ans[0][1][:-4])
        
    arr_sort = []
    for i in range(len(detected)):
        arr_sort.append((detected[i], r_sort[i]))
        
    arr_sort = sorted(arr_sort, key = lambda x: x[1][0][1])
        
    detected = [i[0] for i in arr_sort]
    r_sort = [i[1] for i in arr_sort]
    #print(detected, r_sort)
    return detected, r_sort

def save_img(im, detected, r_sort, SAVE_DIR):
    plt.ioff();
    im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis
    ax.imshow(im_rgb)

    for i, elem in enumerate(r_sort):
        elem = elem[0]
        plt.text(elem[0] + 20, elem[1] - 15, detected[i], fontsize = 25, color = 'b')
        
    len_dir = len(os.listdir(SAVE_DIR))
    print(SAVE_DIR + "/" + str(len_dir) +".png")
    plt.savefig(SAVE_DIR + "/" + str(len_dir) +".png")  
    plt.ion()
    plt.axis('off')

# Handle '/start' and '/help'
@bot.message_handler(commands=['help', 'start'])
def send_welcome(message):
    bot.reply_to(message, """\
Hi there, Poker pro player bot. Use it on your own risk and for sure don`t use it in Las Vegas.
Have fun!\
""")

# Handle all other messages with content_type 'text' (content_types defaults to ['text'])
@bot.message_handler(func=lambda message: True)
def echo_message(message):
    bot.reply_to(message, message.text)

@bot.message_handler(content_types=['photo'])
def handle_photo(message):
    bot.reply_to(message, 'Hmhmhm let me see.... 🧐')
    file_info = bot.get_file(message.photo[len(message.photo) - 1].file_id)
    downloaded_file = bot.download_file(file_info.file_path)

    len_dir_inp = len(os.listdir('D:/poker/'))
    len_dir_out = len(os.listdir('D:/poker/res'))
    src = 'D:/poker/image_{}.jpg'.format(len_dir_inp)
    
    with open(src, 'wb') as new_file:
        new_file.write(downloaded_file)
    # bot.reply_to(message, "Photo is saved on D:/ Alex`s computer")

    image_path = 'D:/poker/image_{}.jpg'.format(len_dir_inp)
    save_path = 'D:/poker/res'

    im = cv2.imread(image_path)
    detected,r_sort = detect_cards(im)
    eval = Evaluator()
    
    bot.send_message(message.chat.id,'Detected cards: ' + ', '.join(detected))
    
    save_img(im, detected, r_sort,save_path)
    res_img = open(save_path+"/" + str(len_dir_out) +".png", 'rb')

    bot.send_photo(message.chat.id, res_img, reply_to_message_id=message.message_id)

    closed_cards = False
    if 'XX' in detected:
        closed_cards= True
    
    for i in detected: 
        if(i=='XX'): 
            detected.remove('XX')
    
    try:
        detected.remove('XX')
    except:
        print('Tried to remove xx')

    try:
        detected.remove('XX')
    except:
        print('Tried to remove xx second time')

    if len(detected)==4 :
        bot.send_message(message.chat.id,'1st player hand: '+', '.join([detected[0],detected[1]])+'\n' + '2nd player hand: '+', '.join([detected[-1],detected[-2]]))
    elif len(detected)>4 and not closed_cards:

        board = [Card.new(i) for i in detected[2:-2]]

        pl1_hand = [Card.new(detected[0]),Card.new(detected[1])]
        pl2_hand = [Card.new(detected[len(detected)-1]),Card.new(detected[len(detected)-2])]

        score_pl1 = eval.evaluate(board, pl1_hand)
        score_pl2 = eval.evaluate(board, pl2_hand)

        bot.send_message(message.chat.id,'1st player hand: '+', '.join([detected[0],detected[1]])+'\n' + '2nd player hand: '+', '.join([detected[-1],detected[-2]]))
        
        bot.send_message(message.chat.id,'Cards on board: '+', '.join(detected[2:-2]))
        bot.send_message(message.chat.id,'1st player hand rank: '+ str(score_pl1) +'\n' + '2nd player hand: '+ str(score_pl2))
        res_img.close()


bot.infinity_polling()