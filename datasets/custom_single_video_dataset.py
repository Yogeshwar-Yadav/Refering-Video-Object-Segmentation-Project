# import os
# import cv2
# from torch.utils.data import Dataset
# from torchvision import transforms
# from PIL import Image

# class CustomSingleVideoDataset(Dataset):
#     def __init__(self, video_path, query_path=None, transform=None, max_frames=300):
#         self.video_path = video_path
#         self.query_path = query_path  # Accept and store query_path to fix the TypeError
#         self.transform = transform
#         self.max_frames = max_frames

#         self.frames = sorted(os.listdir(video_path))  # If video_path is a folder of frames

#         # Load video using OpenCV
#         cap = cv2.VideoCapture(video_path)

#         if not cap.isOpened():
#             raise IOError(f"Cannot open video file {video_path}")

#         frame_count = 0
#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret or frame_count >= max_frames:
#                 break

#             # Convert BGR (OpenCV) to RGB (PIL)
#             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             pil_img = Image.fromarray(frame)
#             self.frames.append(pil_img)
#             frame_count += 1

#         cap.release()

#         if len(self.frames) == 0:
#             raise RuntimeError("No frames extracted from the video.")

#     def __len__(self):
#         return len(self.frames)

#     def __getitem__(self, idx):
#         frame = self.frames[idx]
#         if self.transform:
#             frame = self.transform(frame)
#         return frame
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class CustomSingleVideoDataset(Dataset):
    def __init__(self, video_path, query_path=None, transform=None, max_frames=300):
        """
        Dataset for processing video frames on the fly.

        Args:
            video_path (str): Path to the video file.
            query_path (str, optional): Path to query data (if applicable).
            transform (callable, optional): Transformations to apply to each frame.
            max_frames (int): Maximum number of frames to extract from the video.
        """
        self.video_path = video_path
        self.query_path = query_path  # Accept and store query_path
        self.transform = transform or transforms.ToTensor()  # Default to ToTensor if no transform is provided
        self.max_frames = max_frames

        # Open the video file using OpenCV
        self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            raise IOError(f"Cannot open video file {video_path}")

        # Get total number of frames in the video
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Limit frames to max_frames if specified
        self.num_frames = min(self.total_frames, max_frames)

    def __len__(self):
        """
        Returns the number of frames to be processed.
        """
        return self.num_frames

    def __getitem__(self, idx):
        if idx >= self.num_frames:
            raise IndexError("Index out of range.")

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError(f"Failed to read frame at index {idx} from {self.video_path}")

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame)
        tensor_frame = self.transform(pil_img)

        # Assuming 'samples' refers to some data related to the frame (like bounding boxes, features, etc.)
        # For now, let's create a dummy tensor for 'samples' (e.g., a tensor of zeros)
        samples = torch.zeros(1, 4)  # Dummy tensor (e.g., one bounding box with [x, y, w, h])

        return {
            'image': tensor_frame,
            'text': "",  # empty string or dummy text if none available
            'idx': idx,   # optional: helpful for debugging or tracking predictions
            'samples': samples  # Add the 'samples' key with dummy data or actual data
        }

    def __del__(self):
        """
        Ensures that OpenCV's VideoCapture is released when the dataset object is deleted.
        """
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
