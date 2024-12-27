from PIL import Image, ImageFile
from transformers import OwlViTProcessor, OwlViTForObjectDetection
import numpy as np
from detect import objectdetect, tensorToImage
import cv2
import time
import math
import matplotlib.pyplot as plt


def binaryfloor(boxes):
    size = (480, 640)
    binaryfloor = np.zeros(size)
    for box in boxes:
        x0, y0, x1, y1 = box
        x0 = int(x0.item())
        if x0 < 0:
            x0 = 0
        elif x0 > size[1]:
            x0 = size[1]
        y0 = int(y0.item())
        if y0 < 0:
            y0 = 0
        elif y0 > size[0]:
            y0 = size[0]
        x1 = int(x1.item())
        if x1 < 0:
            x1 = 0
        elif x1 > size[1]:
            x1 = size[1]
        y1 = int(y1.item())
        if y1 < 0:
            y1 = 0
        elif y1 > size[0]:
            y1 = size[0]
        for j in range(x0, x1):
            for i in range(y0, y1):
                binaryfloor[i][j] = 1
        binaryfloor = np.array(binaryfloor)
        break
    return binaryfloor

def subtractfurniture(floor, furniture):
    for box in furniture:
        size = (480, 640)
        x0, y0, x1, y1 = box
        x0 = int(x0.item())
        if x0 < 0:
            x0 = 0
        elif x0 > size[1]:
            x0 = size[1]
        y0 = int(y0.item())
        if y0 < 0:
            y0 = 0
        elif y0 > size[0]:
            y0 = size[0]
        x1 = int(x1.item())
        if x1 < 0:
            x1 = 0
        elif x1 > size[1]:
            x1 = size[1]
        y1 = int(y1.item())
        if y1 < 0:
            y1 = 0
        elif y1 > size[0]:
            y1 = size[0]
        for i in range(y0, y1):
            for j in range(x0, x1):
                floor[i][j] = 0
    return floor

def getFloor(data):
    floor_boxes = [box for label, box in data if label == "floor" or label == "carpet" or label == "ground"]
    floor = binaryfloor(floor_boxes)
    # Combine furniture bounding boxes
    furniture_boxes = [box for label, box in data if label == "furniture" or label == "wall"]
    floor = subtractfurniture(floor, furniture_boxes)

    return floor

def maskfloor(floormask, depth):
    floormask = np.array(floormask)
    floormask = floormask.astype(np.uint8)
    # for i in range(len(floormask)):
    #     for j in range(len(floormask[0])):
    #         floormask[i][j] = floormask[i][j]*255

    # floormask = np.repeat(floormask[:,:,np.newaxis], 3, axis=2)
    # depth = np.array(depth, dtype = np.uint8)
    # depth = cv2.cvtColor(depth,cv2.COLOR_GRAY2RGB)
    # result = cv2.bitwise_and(depth, floormask)
    # cv2.imwrite("maskedfloor.png", result)
    # mask = (floormask == 0)
    # print(mask)
    # depth[mask] = 0.0
    depth = depth * floormask
    return depth

def findfloor(image, depth):
    texts = ["floor", "carpet", "ground", "furniture", "wall"]
    print(type(image))
    found, labelsandboxes = objectdetect(image, texts)
    zippedlabelsandboxes = list(zip(labelsandboxes[0], labelsandboxes[1]))
    boxes = getFloor(zippedlabelsandboxes)
    return maskfloor(boxes, depth)

def isfloor(floordepth):
    size = floordepth.shape[0]*floordepth.shape[1]
    threshold = size/7
    flooramount = np.sum(floordepth > 0)
    if flooramount > threshold and np.max(floordepth) >= 2:
        print(np.max(floordepth))
        print("there is floor!")
        return True
    print("no floor found :(")
    return False

if __name__ == "__main__":
    image = Image.open("rgb0002.jpg")
    depth = Image.open("depth0002.jpg")
    floordepth = findfloor(image, depth)
    Image.fromarray(floordepth).show()

    print(np.unravel_index(np.argmax(floordepth), floordepth.shape), floordepth.shape)
    floordepth = np.sum(floordepth, axis = 2)
    if isfloor(floordepth):
        maxindex = np.unravel_index(np.argmax(floordepth), floordepth.shape)
        angle = (math.atan((maxindex[1]-floordepth.shape[1])/(maxindex[0])))
        print(angle)

