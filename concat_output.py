import cv2
import os
import argparse

parser = argparse.ArgumentParser(
    description="Concatenate the images in the 'pred' and 'gt' folders and create a video."
)
parser.add_argument(
    "-f",
    "--folder",
    type=str,
    default="output",
    help="Path to the folder containing the predicted images.",
)
args = parser.parse_args()
source_folder = args.folder

# Input folders containing images
pred_folder = os.path.join(source_folder, "pred")
gt_folder = os.path.join(source_folder, "gt")

# Output video file name
output_video = os.path.join(source_folder, "output.mp4")

# List files in both folders
pred_images = sorted(os.listdir(pred_folder))
gt_images = sorted(os.listdir(gt_folder))

# Check if the number of images in both folders is the same
if len(pred_images) != len(gt_images):
    raise ValueError("Number of images in 'pred' and 'gt' folders must be the same.")

# Define the output video codec and frames per second
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
fps = 30  # You can adjust the frames per second as needed

# Get the dimensions of the first image (assuming all images have the same dimensions)
first_image = cv2.imread(os.path.join(pred_folder, pred_images[0]))
height, width, layers = first_image.shape

# Create a VideoWriter object to write the video
video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width * 2, height))

# Iterate through the images and concatenate them horizontally
for pred_image_name, gt_image_name in zip(pred_images, gt_images):
    pred_image = cv2.imread(os.path.join(pred_folder, pred_image_name))
    gt_image = cv2.imread(os.path.join(gt_folder, gt_image_name))
    # draw filename for pred_image and gt_image
    cv2.putText(
        pred_image,
        pred_image_name,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2,
    )
    cv2.putText(
        gt_image, gt_image_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
    )

    # Concatenate the images horizontally
    concatenated_image = cv2.hconcat([pred_image, gt_image])

    # Write the concatenated image to the video
    video_writer.write(concatenated_image)

# Release the VideoWriter
video_writer.release()

print("Video created successfully.")
