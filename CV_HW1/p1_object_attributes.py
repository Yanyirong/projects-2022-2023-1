#!/usr/bin/env python3
from re import X
import cv2
import numpy as np
import sys


def binarize(gray_image, thresh_val):
  # TODO: 255 if intensity >= thresh_val else 0
  binary_image = np.zeros_like(gray_image)
  for i in range(gray_image.shape[0]):
    for j in range(gray_image.shape[1]):
      if gray_image[i][j] >= thresh_val:
        binary_image[i][j] = 255
      else:
        binary_image[i][j] = 0
  return binary_image

def label(binary_image):
  # TODO
  marked_image = np.zeros_like(binary_image)
  # deal with the first line
  pre_stack = []
  pre_stack.append(-1)
  label = 1

  def union(label1,label2):
    if label1 == label2:
      return
    if pre_stack[label1] > pre_stack[label2]:
      pre_stack[label2] += pre_stack[label1]
      pre_stack[label1] = label2
    else:
      pre_stack[label1] += pre_stack[label2]
      pre_stack[label2] = label1
    return

  def find(x):
    if (pre_stack[x]<0):
      return x
    return_value = find(pre_stack[x])
    return return_value

  for i in range(binary_image.shape[0]):
    for j in range(binary_image.shape[1]):
      # deal with the first line
      if i == 0:
        # deal with the first pixel
        if j ==0:
          if binary_image[i][j]>0:
            marked_image[i][j] = label
            label += 1
            pre_stack.append(-1)
        if binary_image[i][j]>0:
          if marked_image[i][j-1] > 0:
            marked_image[i][j] = marked_image[i][j-1]
          else:
            marked_image[i][j] = label
            label += 1
            pre_stack.append(-1)
      #deal with the next lines
      else:
        if j ==0:
          if binary_image[i][j]>0:
            if marked_image[i-1][j]>0:
              marked_image[i][j] = marked_image[i-1][j]
            else:
              marked_image[i][j] = label
              label += 1
              pre_stack.append(-1)
        else:
          if binary_image[i][j] > 0:
            #case 1:
            if marked_image[i][j-1]+marked_image[i-1][j]+marked_image[i-1][j-1] == 0:
              marked_image[i][j] = label
              label += 1
              pre_stack.append(-1)
            if marked_image[i][j-1] > 0:
              marked_image[i][j] = marked_image[i][j-1]
            if marked_image[i-1][j] > 0:
              marked_image[i][j] = marked_image[i-1][j]
            if marked_image[i-1][j-1] > 0:
              marked_image[i][j] = marked_image[i-1][j-1]
  #Now we get a marked image
  labeled_image = np.zeros_like(marked_image)
  for i in range(marked_image.shape[0]):
    for j in range(marked_image.shape[1]):
      if i ==0:
        continue
      else:
        if j ==0:
          continue
        else:
          if marked_image[i][j] == 0:
            if marked_image[i-1][j] > 0 and marked_image[i][j-1] > 0:
              if marked_image[i-1][j] == marked_image[i][j-1]:
                continue
              else:
                union(marked_image[i-1][j],marked_image[i][j-1])
          else:
            if marked_image[i-1][j] > 0 and marked_image[i][j-1] > 0:
              if marked_image[i-1][j] == marked_image[i][j-1]:
                continue
              else:
                union(marked_image[i-1][j],marked_image[i][j-1])
  # union done
  for i in range(marked_image.shape[0]):
    for j in range(marked_image.shape[1]):
      if marked_image[i][j] > 0:
        labeled_image[i][j] = find(marked_image[i][j])
  return labeled_image

def get_attribute(labeled_image):
  # TODO
  attribute_list=[]
  return attribute_list

def main(argv):
  img_name = argv[0]
  thresh_val = int(argv[1])
  img = cv2.imread('data/' + img_name + '.png', cv2.IMREAD_COLOR)
  gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  binary_image = binarize(gray_image, thresh_val=thresh_val)
  labeled_image = label(binary_image)
  attribute_list = get_attribute(labeled_image)

  cv2.imwrite('output/' + img_name + "_gray.png", gray_image)
  cv2.imwrite('output/' + img_name + "_binary.png", binary_image)
  cv2.imwrite('output/' + img_name + "_labeled.png", labeled_image)
  print(attribute_list)


if __name__ == '__main__':
  main(sys.argv[1:])
