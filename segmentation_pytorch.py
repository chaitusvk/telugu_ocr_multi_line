import numpy as np
import cv2 as cv

from glob import glob
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize, thin
from PIL import Image
from demo_functioncall import predict_word
from page import Page










def extract_words(img_path, visual=0):
    img_path = img_path.strip("\n")
    txt_para = str()
    page = Page(img_path)
    page.process()
    num_lines = page.num_lines
    text_num_lines = str(num_lines)
    print("NumLines:",num_lines)

    
    for line_id in range(0,num_lines):
        line = page.lines[line_id].arr
        if(line.shape[0]>40):
            line = line[20:,:]
        h_p = 64-line.shape[0]
        w_p = 800-line.shape[1]
        h_p_l = h_p // 2
        h_p_r = h_p - h_p_l
        w_p_l = w_p //2
        w_p_r = w_p - w_p_l
        img_paded = np.pad(line,((h_p_l,h_p_r),(w_p_l,w_p_r)), mode='constant', constant_values=0)
        img_p = Image.fromarray(np.uint8(img_paded * 255) , 'L')

        txt_line = predict_word(img_p)

        txt_para = txt_para + txt_line + str('<br> ')



    
    
            

    
    return txt_para


if __name__ == "__main__":
    
    img = cv.imread('../Dataset/tel_img.png')
    words = extract_words(img, 1)
    for word in words:
        print(word[0].shape)
        # print(word[0].type)