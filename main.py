from PIL import Image
import os
import time
import cv2
import numpy as np
import multiprocessing
import sys

def edge_detection(img):
  # Perform edge detection using the OpenCV library
  img = np.array(img)
  print(img.shape)
  edges = cv2.Canny(img, 150, 200)
  return edges


BSDS_IMAGES = "./bsds_images"  # Replace with the actual path to bsds_images directory
PERCENT_OVERLAP = 0.1

if __name__ == "__main__":
  NUM_CORES = multiprocessing.cpu_count()
  if len(sys.argv) < 3 or int(sys.argv[1]) <= 0 or int(sys.argv[2]) <= 0:
    print("Usage: python main.py <num_cores_used> <num_images_to_check>")
    sys.exit(1)

  num_cores_used = min(NUM_CORES, int(sys.argv[1]))
  num_images_to_check = min(400, int(sys.argv[2]))


  total_time = 0
  # Get the first NUM_IMAGES files from bsds_images
  image_files = os.listdir(BSDS_IMAGES)[:num_images_to_check]
  start_time = time.time()
  for image_path in image_files:
    image = Image.open(f"./bsds_images/{image_path}")

    # Split the image into num_cores_used equal sized chunks with an overlap of PERCENT_OVERLAP
    chunk_height = image.height // num_cores_used
    chunk_width = image.width // num_cores_used
    y_overlap = int(chunk_height * PERCENT_OVERLAP)
    x_overlap = int(chunk_width * PERCENT_OVERLAP)

    chunks = []
    for i in range(num_cores_used):
      for j in range(num_cores_used):
        start_row = max(0, i * chunk_height - y_overlap)
        end_row = min(image.height, (i + 1) * chunk_height + y_overlap)
        start_col = max(0, j * chunk_width - x_overlap)
        end_col = min(image.width, (j + 1) * chunk_width + x_overlap)
        chunk = image.crop((start_col, start_row, end_col, end_row))
        chunks.append(chunk)

    # Combine the chunks back together to form one image
    
      combined_image = None
      for i, chunk in enumerate(chunks):
        chunk_edges = edge_detection(chunk)
        if combined_image is None:
            combined_image = np.zeros((image.height, image.width))

        start_row = max(0, i * chunk_height - y_overlap)
        end_row = start_row + chunk_edges.shape[0]
        start_col = max(0, j * chunk_width - x_overlap)
        end_col = start_col + chunk_edges.shape[1]

        combined_image[start_row:end_row, start_col:end_col] = np.maximum(combined_image[start_row:end_row, start_col:end_col], chunk_edges)
      # Save the edges as a new image
      edges_image = Image.fromarray(combined_image).convert('L')
      edges_image.save(f"edges_{image_path}")


  end_time = time.time()
  total_time = end_time - start_time
  print(f"Total running time over all images: {total_time} seconds")

