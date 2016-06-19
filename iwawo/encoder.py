#!/usr/bin/env python
#-*- coding: utf-8 -*-

import os
from PIL import Image
import re

direct = "image/"
files = os.listdir(direct)

files.remove(".DS_Store")

a = 1
for i, file in enumerate(files):
    print file
    img = Image.open(direct+file)

    img.save("image2/"+str(a)+".bmp", "bmp")
    a += 1
             
             
    
    
