from PIL import Image
import os
import time
import numpy as np
import multiprocessing
import concurrent.futures
import sys
import process_chunk

BSDS_IMAGES = "./bsds_images"
def evaluate_performace(image_files, num_cores_used, num_images_to_check, create_images, verbose):

  start_time = time.time()
  for i, image_path in enumerate(image_files):
    image = Image.open(f"./bsds_images/{image_path}")
    image_array = np.array(image)
    image_chunk = process_chunk.Chunk((0, 0), (image.height - 1, image.width - 1), image_array)
    combined_image = np.zeros((image_array.shape[0], image_array.shape[1]))

    # Split the image into num_cores_used equal sized chunks with an overlap of PERCENT_OVERLAP
    num_halves = np.log2(num_cores_used)
    chunks = process_chunk.halve_chunk(image_chunk, num_halves) # len(chunks) = num_cores_used

    # Rejoin the chunks together after edge-detection
    with concurrent.futures.ThreadPoolExecutor() as executor:
      # Parallelize edge detection
      results = [executor.submit(process_chunk.process_chunk, chunk) for chunk in chunks]

      for f in concurrent.futures.as_completed(results):
        chunk = f.result()
        start_row = chunk.top_left[0]
        end_row = chunk.bottom_right[0]
        start_col = chunk.top_left[1]
        end_col = chunk.bottom_right[1]
        combined_image[start_row:(end_row+1), start_col:(end_col+1)] = np.maximum(
          combined_image[start_row:(end_row+1), start_col:(end_col+1)], chunk.edges
        )

    # For checking edge detection results
    if create_images:
      edges_image = Image.fromarray(combined_image).convert('L')
      image.save(f"./edge_results/original_{image_path}")
      edges_image.save(f"./edge_results/edges_{image_path}")

    if verbose: print(f"{i+1}/{num_images_to_check}")

  end_time = time.time()
  return end_time - start_time

if __name__ == "__main__":
  if len(sys.argv) < 3 or int(sys.argv[1]) <= 0 or int(sys.argv[2]) <= 0:
    print("Usage: python main.py <num_cores_used> <num_images_to_check> ")
    sys.exit(1)

  max_num_cores_used = min(multiprocessing.cpu_count(), int(sys.argv[1]))
  if not (max_num_cores_used & (max_num_cores_used - 1) == 0):
    print("<num_cores_used> must be a power of 2")
    sys.exit(1)
  num_images_to_check = min(400, int(sys.argv[2]))
  # Get the first NUM_IMAGES files from bsds_images
  image_files = os.listdir(BSDS_IMAGES)[:num_images_to_check]
  # Generate all the powers of 2 up to (and including) the number of cores provided on the command line
  num_cores_to_test = [2 ** p for p in range(int(np.log2(max_num_cores_used)) + 1)]
  for num_cores in num_cores_to_test:
    total_execution_time = 0
    for i in range(10):
      total_execution_time += evaluate_performace(image_files, num_cores, num_images_to_check, False, False)
    average_execution_time = total_execution_time / 10
    print(f"{num_cores}: {average_execution_time * 1000}ms")