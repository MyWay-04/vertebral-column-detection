# segmentation.py
import cv2
import numpy as np
from preprocess import segment_and_threshold

def get_connected_component(img):
    visited = np.zeros_like(img)
    component = np.zeros_like(img)
    height, width = img.shape
    comp_num = 1
    for row in range(height):
        for col in range(width):
            if img[row, col] > 240 and visited[row, col] == 0:
                stack = [(row, col)]
                visited[row, col] = 1
                # comp_size += 1
                while stack:
                    r, c = stack.pop()
                    component[r, c] = comp_num
                    nb = get_neighbour(img, r, c)
                    new_nb = []
                    for coord in nb:
                        if visited[coord[0], coord[1]] == 0 and img[coord[0], coord[1]] > 240:
                            new_nb.append(coord)
                            visited[coord[0], coord[1]] = 1
                            # comp_size += 1
                    stack += new_nb
                comp_num += 1
    return component

def get_neighbour(img, row, col):
    neighbours = []
    height, width = img.shape
    for i in range(-1, 2):
        for j in range(-1, 2):
            if row + i >= 0 and row + i < height and col + j >= 0 and col + j < width and not (i == 0 and j == 0):
                neighbours.append((row + i, col + j))
    return neighbours

def height_segment_threshold(img, segments: int = 16, threshold: float = float(3)):
    height, width = img.shape
    segment_height = height // segments
    thresholded_segments = []
    for i in range(segments):
        segment = img[i * segment_height:(i + 1) * segment_height, :]
        threshold_value = np.percentile(segment, 100 - threshold)
        thresholded_segment = np.where(segment >= threshold_value, 255, 0).astype(np.uint8)
        thresholded_segments.append(thresholded_segment)

    #Handle the last segment if the height is not perfectly divisible
    if height % segments != 0:
      i = segments
      segment = img[i * segment_height:, :]
      threshold_value = np.percentile(segment, 100 - threshold)
      thresholded_segment = np.where(segment >= threshold_value, 255, 0).astype(np.uint8)
      thresholded_segments.append(thresholded_segment)

    modified = np.concatenate(thresholded_segments, axis=0)

    return modified

def find_box_start_to_end(image):
    weight, height = image.shape
    start = 0
    end = 0
    index_box = {} 
    for row in range(weight):
        for col in range(height):
            if image[row][col] != 0 :
                if image[row][col] in index_box:
                  if row < index_box[image[row][col]]["start"][0]: #check left most
                    index_box[image[row][col]]["start"][0] = row
                  if row > index_box[image[row][col]]["end"][0]: #check right most
                    index_box[image[row][col]]["end"][0] = row
                  index_box[image[row][col]]["end"][1] = col
                else:
                  index_box[image[row][col]] = {"start":[row,col],"end":[row,col]}
    return index_box #{index: {start: [row,col], end : [row,col]}}