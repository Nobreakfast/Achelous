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
pred_folder = source_folder + "pred"
gt_folder = source_folder + "gt"

# Output video file name
output_video = source_folder + "output_video.mp4"

# Get the list of image files in the folders
pred_images = [
    os.path.join(pred_folder, img) for img in sorted(os.listdir(pred_folder))
]
gt_images = [os.path.join(gt_folder, img) for img in sorted(os.listdir(gt_folder))]

# Check if both folders have the same number of images
if len(pred_images) != len(gt_images):
    raise ValueError(
        "The number of images in 'pred' and 'gt' folders must be the same."
    )

# Get image dimensions
image = cv2.imread(pred_images[0])
height, width, layers = image.shape

# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*"mp4")  # You can change the codec as needed
video_writer = cv2.VideoWriter(output_video, fourcc, 30.0, (width, height))

# Iterate through the images and create the video
for pred_img, gt_img in zip(pred_images, gt_images):
    pred_frame = cv2.imread(pred_img)
    gt_frame = cv2.imread(gt_img)

    # Concatenate the two images horizontally
    concat_frame = cv2.hconcat([pred_frame, gt_frame])

    # Write the frame to the video
    video_writer.write(concat_frame)

# Release the VideoWriter
video_writer.release()

print("Video created successfully.")
