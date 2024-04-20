from PIL import Image
import os
import time
import cv2
import numpy as np
import multiprocessing
import sys

def edge_detection(chunk):
  # Perform edge detection using the OpenCV library
  print(chunk.data.shape)
  edges = cv2.Canny(chunk.data, 150, 200)
  print(edges.shape)
  return edges

class Chunk:
  def __init__(self, top_left, bottom_right, data):
    self.top_left = top_left
    self.bottom_right = bottom_right
    self.data = data

  def set_edges(self, edges):
    self.edges = edges


BSDS_IMAGES = "./bsds_images"  # Replace with the actual path to bsds_images directory
PERCENT_OVERLAP = 0.1

if __name__ == "__main__":
  NUM_CORES = multiprocessing.cpu_count()
  if len(sys.argv) < 3 or int(sys.argv[1]) <= 0 or int(sys.argv[2]) <= 0:
    print("Usage: python main.py <num_cores_used> <num_images_to_check>")
    sys.exit(1)

  num_cores_used = min(NUM_CORES, int(sys.argv[1]))
  if not (num_cores_used & (num_cores_used - 1) == 0):
    print("<num_cores_used> must be a power of 2")
    sys.exit(1)
  num_images_to_check = min(400, int(sys.argv[2]))


  total_time = 0
  # Get the first NUM_IMAGES files from bsds_images
  image_files = os.listdir(BSDS_IMAGES)[:num_images_to_check]
  start_time = time.time()
  for image_path in image_files:
    image = Image.open(f"./bsds_images/{image_path}")
    image_array = np.array(image)
    image_chunk = Chunk((0, 0), (image.height - 1, image.width - 1), image_array)
    # Split the image into num_cores_used equal sized chunks with an overlap of PERCENT_OVERLAP
    chunk_height = image.height // num_cores_used
    chunk_width = image.width // num_cores_used
    y_overlap = int(chunk_height * PERCENT_OVERLAP)
    x_overlap = int(chunk_width * PERCENT_OVERLAP)

    chunks = []
    combined_image = None
    num_halves = np.log2(num_cores_used)

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

    # Split the image into chunks
    chunks = halve_chunk(image_chunk, num_halves)


    combined_image = np.zeros((image_array.shape[0], image_array.shape[1]))
    # Rejoin the chunks together
    for chunk in chunks:
      chunk_edges = edge_detection(chunk)
      start_row = chunk.top_left[0]
      end_row = chunk.bottom_right[0]
      start_col = chunk.top_left[1]
      end_col = chunk.bottom_right[1]
      combined_image[start_row:(end_row+1), start_col:(end_col+1)] = np.maximum(combined_image[start_row:(end_row+1), start_col:(end_col+1)], chunk_edges)

    edges_image = Image.fromarray(combined_image).convert('L')
    image.save(f"./edge_results/original_{image_path}")
    edges_image.save(f"./edge_results/edges_{image_path}")

  # # Combine the chunks back together to form one image
  #   for x, chunk in enumerate(chunks):
  #     chunk_edges = edge_detection(chunk)
  #     if combined_image is None:
  #         combined_image = np.zeros((image.height, image.width))

  #     start_row = max(0, x * chunk_height - y_overlap)
  #     end_row = start_row + chunk_edges.shape[0]
  #     start_col = max(0, x * chunk_width - x_overlap)
  #     end_col = start_col + chunk_edges.shape[1]

  #     combined_image[start_row:end_row + 1, start_col:end_col + 1] = np.maximum(combined_image[start_row:end_row + 1, start_col:end_col + 1], chunk_edges)
  #   # Save the edges as a new image
  #   edges_image = Image.fromarray(combined_image).convert('L')
  #   edges_image.save(f"edges_{image_path}")


  end_time = time.time()
  total_time = end_time - start_time
  print(f"Total running time over all images: {total_time} seconds")

