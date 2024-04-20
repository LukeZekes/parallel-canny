import cv2
import numpy as np

PERCENT_OVERLAP = 0.1

class Chunk:
  def __init__(self, top_left, bottom_right, data):
    self.top_left = top_left
    self.bottom_right = bottom_right
    self.data = data

  def set_edges(self, edges):
    self.edges = edges
def edge_detection(chunk):
  # Perform edge detection using the OpenCV library
  edges = cv2.Canny(chunk.data, 150, 200)
  return edges

def halve_chunk(chunk:Chunk, n):
      if n <= 0: return [chunk]

      height =  chunk.bottom_right[0] - chunk.top_left[0]
      width = chunk.bottom_right[1] - chunk.top_left[1]
      # Split the chunk perpendicular to the longest axis
      if height > width:
        # Split chunk horizontally
        half_height = height // 2
        overlap_size = int(half_height * PERCENT_OVERLAP)
        chunk1 = Chunk(top_left=chunk.top_left, bottom_right=(chunk.top_left[0] + half_height + overlap_size - 1, chunk.bottom_right[1]), data=chunk.data[:half_height + overlap_size, :])
        chunk2 = Chunk(top_left=(chunk.top_left[0] + half_height - overlap_size, chunk.top_left[1]), bottom_right=chunk.bottom_right,data=chunk.data[half_height - overlap_size: ])
      else:
        # Split chunk vertically
        half_width = width // 2
        overlap_size = int(half_width * PERCENT_OVERLAP)
        chunk1 = Chunk(top_left=chunk.top_left, bottom_right=(chunk.bottom_right[0], chunk.top_left[1] + half_width + overlap_size - 1), data=chunk.data[:, :half_width + overlap_size])
        chunk2 = Chunk(top_left=(chunk.top_left[0], chunk.top_left[1] + half_width - overlap_size), bottom_right=chunk.bottom_right, data=chunk.data[:, half_width - overlap_size:])

      chunks = [chunk1, chunk2]

      return [item for chunk in chunks for item in halve_chunk(chunk, n-1)]

def process_chunk(chunk:Chunk):
  chunk_edges = edge_detection(chunk)
  chunk.set_edges(chunk_edges)
  return chunk
