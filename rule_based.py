import gzip
import json
from gym import spaces
from importlib import import_module
import torch
import torchvision
import cv2
import time
import math
import os
import datetime
import sys
import numpy as np
from numpy import asarray
from PIL import Image as PILIMAGE
import torchvision.transforms as T
from PIL import Image
from detect import objectdetect
from instructextract import *
from floordetection import findfloor, isfloor
import time
sys.path.append('/home/elle/interbotix_ws/src/interbotix_ros_toolboxes/interbotix_xs_toolbox/interbotix_xs_modules/src/interbotix_xs_modules/')
from interbotix_xs_modules.locobot import InterbotixLocobotCreate3XS

import gc

locobot = InterbotixLocobotCreate3XS(robot_model="locobot_base")
locobot.camera.pan_tilt_go_home()
print("robot setup")

def findclearpath():
    print("finding clear path...")
    pathclear = False
    #points camera to floor so it can better analyze floor
    locobot.camera.tilt(math.pi / 12.0)
    while not pathclear:
        img, depth = getimage()
        # find empty floor
        floordepth = findfloor(img, depth)
        # floordepth = np.sum(floordepth, axis=2)
        if isfloor(floordepth):
            #finds farthest clear floor path
            maxindex = np.unravel_index(np.argmax(floordepth), floordepth.shape)
            #rotates locobot to face the clear floor path
            angle = (math.atan((maxindex[1]-floordepth.shape[1])/(maxindex[0])))
            locobot.base.move(0.1, angle, 1.3)
            locobot.base.move(3, 0, 2.00)
            pathclear = True
            print("found clear path!")
            locobot.camera.tilt(-math.pi / 12.0)
        else:
            #rotates locobot if it doesn't see a clear path
            locobot.base.move(0.1, math.pi/4, 1.3)

def run(instruction):
    nounverbs = extract(instruction)
    i=0
    while i <= len(nounverbs)-1:
        if nounverbs[i][1] == "N":
            findobject(nounverbs[i][0])
            i += 1
        else:
            print({nounverbs[i][0]})
            i = i + action(nounverbs[i:])


def search(word):
    locobot.camera.pan_tilt_go_home()
    found = False
    time.sleep(0.5) #so that image is not blurry
    img, depth = getimage()
    f, boxes = objectdetect(img, [word])
    print(f)
    if f:
        print(f"{word} detected!")
        print(boxes)
        #make tuples of boxes and corresponding labels
        boxes = [(temp1, temp2) for (temp1, temp2) in zip(boxes[0], boxes[1])]
        print(boxes)
        middlex = 320
        print(boxes)
        #rotate locobot to middle of the object's bounding box
        for i, j in enumerate(boxes):
            label, box = j
            if label == word:
                x0, y0, x1, y1 = box
                print(f"x1-x0 = {x1-x0}")
                if x1-x0 < 360:
                    angle = math.atan(((middlex-((x0+x1)/2)) / 460))
                    print(f"rotating {angle} to face object!")
                    locobot.base.move(0.1, angle, 1.3)
                    break

        locobot.camera.tilt(math.pi / 13.0)
        time.sleep(0.5)
        img, depth = getimage()
        # find empty floor
        floordepth = findfloor(img, depth)
        floormax = np.max(floordepth[:, middlex])
        while floormax >= 0.3:
            locobot.base.move(1, math.pi/7, 1.30)
            img, depth = getimage()
            floordepth = findfloor(img, depth)
            floormax = np.max(floordepth[:, middlex])
            gc.collect()
        found = True
        print(f"{word} found!, moving on")
    del img, depth, boxes
    return found


def findobject(word):
    found = False
    print(f"looking for {word}")
    while not found:
        # for i in range(8):
        found = search(word)
        if found:
            return
        locobot.base.move(0.1, (2*math.pi)/8, 1.0)
        # findclearpath()

def action(nounverbs):
    nextobject = ""
    #given that instructions always finish on a noun
    for thing in nounverbs:
        if thing[1] == "N":
            nextobject = thing[0]
            index = nounverbs.index(thing)
            break
    if nounverbs[0][0] == "straight":
        found = False
        while not found:
            locobot.camera.tilt(math.pi / 12.0)
            img, depth = getimage()
            # find empty floor
            floordepth = findfloor(img, depth)
            del img
            del depth
            # floordepth = np.sum(floordepth, axis=2)
            findclearpath()
            if isfloor(floordepth):
                locobot.base.move(0.1, (2*math.pi)/5, 1.3)
                found = search(nextobject)
                if found:
                    locobot.camera.tilt(math.pi / 12.0)
                    return index
                locobot.base.move(0.1, (2*math.pi)/5, 1.3)
                found = search(nextobject)
                if found:
                    locobot.camera.tilt(math.pi/ 12.0)
                    return index
                locobot.base.move(0.1, (2*math.pi)/5, 1.3)
                found = search(nextobject)
                if found:
                    locobot.camera.tilt(math.pi / 12.0)
                    return index
                locobot.base.move(0.1, (2*math.pi)/5, 1.3)
                locobot.base.move(1, 0, 2.00)

            else:
                findclearpath()
                return 0
    elif nounverbs[0][0] == "right" or nounverbs[0][0] == "left":
        if nounverbs[0][0] == "right":
            print("turning right!")
            locobot.base.move(0.1, -math.pi / 3, 1.2)
        else:
            print("turning left")
            locobot.base.move(0.1, math.pi / 3, 1.2)
        found = False
        found = search(nextobject)
        if found:
            return index + 1
        else:
            return 1
            # while not found:
            #     print ("THIS THING!!")
            #     locobot.camera.tilt(math.pi / 12.0)
            #     img, depth = getimage()
            #     # find empty floor
            #     floordepth = findfloor(img, depth)
            #     del img
            #     del depth
            #     # floordepth = np.sum(floordepth, axis=2)
            #     findclearpath()
            #     if isfloor(floordepth):
            #         locobot.base.move(0.1, (2*math.pi)/5, 1.3)
            #         found = search(nextobject)
            #         if found:
            #             locobot.camera.tilt(-math.pi / 12.0)
            #             return index
            #         locobot.base.move(0.1, (2*math.pi)/5, 1.3)
            #         found = search(nextobject)
            #         if found:
            #             locobot.camera.tilt(-math.pi / 12.0)
            #             return index
            #         locobot.base.move(0.1, (2*math.pi)/5, 1.3)
            #         found = search(nextobject)
            #         if found:
            #             locobot.camera.tilt(-math.pi / 12.0)
            #             return index
            #         locobot.base.move(0.1, (2*math.pi)/5, 1.3)
            #         locobot.base.move(1, 0, 2.00)
            #     else:
            #         findclearpath()
            #         locobot.camera.tilt(-math.pi / 12.0)
            #         return 0

def getimage():
    color_image = None
    depth_image = None
    if locobot != None:
        color_image, depth_image = locobot.base.get_img()
        color_img = PILIMAGE.fromarray(color_image.astype(np.uint8))
        #converts depth image from millimeters to meters
        depth_img = depth_image / 1000.0
        # converts depth image to a torch tensor
        depth_img = torch.Tensor(depth_img)
        # clips depth image of outliers
        depth_img = torch.clip(depth_img,0.2,3)
        # converts depth image back to numpy array
        depth_img = depth_img.numpy()
        # depth_img = PILIMAGE.fromarray(depth_img.astype(np.uint16), mode='I;16')
        # color_img.save("color_img.jpg")
        # del color_img
        # finalcolor = Image.open("color_img.jpg")
        finalcolor = color_img
        finaldepth = depth_img
        return finalcolor, finaldepth
    return

# run("start by the bed and turn right and go to the table")
run(input("directions?"))
# start by the bed and turn right and go to the table

# locobot.camera.tilt(math.pi / 13.0)
# img, depth = getimage()
# img.save("test.jpg")
