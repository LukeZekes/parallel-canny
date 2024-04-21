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
def sobel_edge_detection(chunk):
  # Perform edge detection using the OpenCV library
  edges = cv2.Canny(chunk.data, 150, 200)
  return edges

def non_maximum_suppression(amplitude, direction):
    # Perform non maximum suppression on the amplitude and direction of the gradient
    suppressed_amplitude = amplitude.copy()

    for i in range(1, amplitude.shape[0] - 1):
      for j in range(1, amplitude.shape[1] - 1):
        angle = direction[i, j]

        # Determine the neighboring pixels based on the angle
        if (angle >= -np.pi/8 and angle < np.pi/8) or (angle >= 7*np.pi/8 and angle <= np.pi) or (angle >= -np.pi and angle < -7*np.pi/8):
          neighbor1 = amplitude[i, j + 1]
          neighbor2 = amplitude[i, j - 1]
        elif (angle >= np.pi/8 and angle < 3*np.pi/8) or (angle >= -7*np.pi/8 and angle < -5*np.pi/8):
          neighbor1 = amplitude[i + 1, j - 1]
          neighbor2 = amplitude[i - 1, j + 1]
        elif (angle >= 3*np.pi/8 and angle < 5*np.pi/8) or (angle >= -5*np.pi/8 and angle < -3*np.pi/8):
          neighbor1 = amplitude[i + 1, j]
          neighbor2 = amplitude[i - 1, j]
        else:
          neighbor1 = amplitude[i - 1, j - 1]
          neighbor2 = amplitude[i + 1, j + 1]

        # Suppress the amplitude if it is not the maximum among the neighbors
        if amplitude[i, j] < neighbor1 or amplitude[i, j] < neighbor2:
          suppressed_amplitude[i, j] = 0

    return suppressed_amplitude


def roberts_edge_detection(chunk):
  # Convert the chunk data to grayscale
  gray_data = cv2.cvtColor(chunk.data, cv2.COLOR_BGR2GRAY)
  # Perform Gaussian blur on the chunk data
  blurred_data = cv2.GaussianBlur(gray_data, (5, 5), 0)
  
  # Perform convolution on the blurred data
  Gx = np.array(([1, 0], [0, -1]))
  Gy = np.array([[0, 1], [-1, 0]])
  Ex = cv2.filter2D(blurred_data, -1, Gx)
  Ey = cv2.filter2D(blurred_data, -1, Gy)

  # Calculate the amplitude and direction of the gradient
  amplitude = np.sqrt(Ex**2 + Ey**2)
  direction = np.arctan2(Ey, Ex)

  suppressed_amplitude = non_maximum_suppression(amplitude, direction)

  # Apply double-threshold method to select edge candidates from suppressed_amplitude
  low_threshold = 0.1 * np.max(suppressed_amplitude)
  high_threshold = 0.3 * np.max(suppressed_amplitude)

  # Initialize an array to store the edge candidates
  edge_candidates = np.zeros_like(suppressed_amplitude)

  # Iterate over each pixel in the suppressed_amplitude
  for i in range(suppressed_amplitude.shape[0]):
    for j in range(suppressed_amplitude.shape[1]):
      if suppressed_amplitude[i, j] >= high_threshold:
        # Strong edge candidate
        edge_candidates[i, j] = 255
      elif suppressed_amplitude[i, j] >= low_threshold:
        # Check if the pixel is adjacent to another edge point
        is_adjacent = False
        for x in range(max(0, i-1), min(suppressed_amplitude.shape[0], i+2)):
          for y in range(max(0, j-1), min(suppressed_amplitude.shape[1], j+2)):
            if edge_candidates[x, y] == 255:
              is_adjacent = True
              break
          if is_adjacent:
            break
        if is_adjacent:
          # Weak edge candidate
          edge_candidates[i, j] = 255

  return edge_candidates

  
  
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
  chunk_edges = sobel_edge_detection(chunk)
  chunk.set_edges(chunk_edges)
  return chunk
