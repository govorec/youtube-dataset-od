import yt_dlp
import cv2
from urllib.parse import urlparse
from urllib.parse import parse_qs
import glob
import os
import torch
import fiftyone as fo
from fiftyone import ViewField as F
import fiftyone.utils.random as four
import pandas as pd

os.makedirs('data/videos', exist_ok=True)
os.makedirs('data/frames', exist_ok=True)
os.makedirs('dataset', exist_ok=True)


def download_videos(video_urls, output_dir):
    """Downloads videos from the given URLs to the specified directory.

    Args:
        video_urls (list): List of YouTube video URLs.
        output_dir (str): Path to the output directory.
    """
    os.makedirs(output_dir, exist_ok=True)

    with yt_dlp.YoutubeDL({
        "outtmpl": os.path.join(output_dir, "%(autonumber)s [%(id)s] %(title)s.%(ext)s"),
        "format": "135",  # 480p
        #"format":"136", #720p #'best'
        "progress_hooks": [lambda d: print(f"{d['status']}%")],
        "quiet": True,
    }) as ydlp:
        ydlp.download(video_urls)


def extract_frames(videos_dir, output_dir, max_frames=5000):
    """Extracts frames each second from a videos and saves them to the specified directory.

    Args:
        videos_dir (str): Path to the directory with video files.
        output_dir (str): Path to the output directory.
        skip_frames (int, optional): The number of frames to skip between extractions. Defaults to 5000.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Splitting videos to frames
    for video_path in glob.glob(os.path.join(videos_dir,"*")):
        print(f"\nProcessing: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        skip_frames = max(fps, -(total_frames // -(max_frames))) # up to 5k frames

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % skip_frames == 0:
                frame_path = os.path.join(
                    output_dir,
                    f"{os.path.basename(video_path)[:19]} {frame_count:07}.jpg",
                )
                cv2.imwrite(frame_path, frame)
                print(f"Progress: {(frame_count + 1) / total_frames * 100:.2f}%", end="\r")

            frame_count += 1

        cap.release()


def create_dataset(frames_dir, class_mapping, conf_threshold, ds_name):
    """
    Creates a FiftyOne dataset from the given frames, applying object detection.

    Args:
        frames_dir (str): Path to the directory with frame files.
        class_mapping (dict): Mapping of YOLOv5 class IDs to desired labels.
        conf_threshold (float): Confidence threshold for object detection.

    Returns:
        fo.Dataset: The created FiftyOne dataset.
    """
    frame_paths = glob.glob(os.path.join(frames_dir,"*.jpg"))
    
    # Load pretrained model fo data labeling
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

    samples = []
    for filepath in frame_paths:
        # Preprocess the image
        results = model(filepath)
        boxesn = results.pandas().xyxyn[0]
        # Keep only necessary detections
        boxesn = boxesn[
            boxesn['class'].isin(class_mapping.keys()) & (boxesn['confidence']>=conf_threshold)
        ].replace({"class": class_mapping})
        boxesn = boxesn.assign(width = lambda r: r.xmax - r.xmin, height = lambda r: r.ymax - r.ymin)

        # Convert detections to FiftyOne format
        detections = []
        for i, obj in boxesn.iterrows():
            detections.append(
                fo.Detection(
                    label=obj["class"], 
                    bounding_box=obj[['xmin', 'ymin', 'width', 'height']].tolist(), 
                    confidence=obj['confidence'], 
                    index=i
                )
            )

        sample = fo.Sample(filepath=filepath)
        sample["ground_truth"] = fo.Detections(detections=detections)
        samples.append(sample)

    # Create dataset
    dataset = fo.Dataset(ds_name, overwrite=True)
    dataset.add_samples(samples)

    # Train/validation split
    four.random_split(dataset, {"train": 0.8, "val": 0.2})
    return dataset


def data_distribution_analysis(ds):
        """
        Analyzes data distribution and print as markdown.

        Args:
            dataset (fo.Dataset): The FiftyOne dataset.
        """
        samples_cnt = ds.count()
        labels_cnt = ds.count("ground_truth.detections.label")
        print("Samples count:", samples_cnt, '<br>')
        print("Labels count:", labels_cnt)

        dist_analysis = (
            pd.Series(ds.count_values("ground_truth.detections.label"), name="label_counts")
            .rename_axis("class")
            .to_frame()
        )
        dist_analysis["label_share"] = dist_analysis["label_counts"] / labels_cnt
        dist_analysis["samples_counts"] = {
            k: ds.filter_labels("ground_truth", F("label").is_in([k])).count()
            for k in dist_analysis.index
        }
        dist_analysis["samples_share"] = dist_analysis["samples_counts"] / samples_cnt

        print("\nDistribution labels and samples between classes:")
        print(dist_analysis.to_markdown())


def analyze_and_export_dataset(dataset, output_dir):
    """
    Analyzes and exports the dataset to COCO format.

    Args:
        dataset (fo.Dataset): The FiftyOne dataset.
        output_dir (str): Path to the output directory.
    """
    print("\n# Whole dataset")
    data_distribution_analysis(dataset)
    print("\n## Train split")
    data_distribution_analysis(dataset.match_tags("train"))
    print("\n## Val split")
    data_distribution_analysis(dataset.match_tags("val"))

    os.makedirs(output_dir, exist_ok=True)

    dataset.export(
        export_dir=output_dir,
        dataset_type=fo.types.COCODetectionDataset,
        label_field="ground_truth",
    )


if __name__ == "__main__":
    # List of video links to collect
    video_urls = [
        # cars
        "https://www.youtube.com/watch?v=MNn9qKG2UFI",
        
        # people
        "https://www.youtube.com/watch?v=8qfdGBNoqBc",
        "https://www.youtube.com/watch?v=NyLF8nHIquM",
        
        # pets
        "https://www.youtube.com/watch?v=w8WdogrgkAU",
        "https://www.youtube.com/watch?v=czoArYpSCMI",
        "https://www.youtube.com/watch?v=ZdwzExyUY9k"
    ]
    class_mapping = {
        0: 'person',
        #1: 'bicycle',
        2: 'car',
        3: 'car', #'motorcycle',
        #4: 'airplane',
        5: 'car', #'bus',
        #6: 'train',
        7: 'car', #'truck',
        
        14: 'pet', #'bird',
        15: 'pet', #'cat',
        16: 'pet', #'dog',
        17: 'pet', #'horse',
        18: 'pet', #'sheep',
        19: 'pet', #'cow'
    }
    # Detection confidence threshold
    conf_threshold = 0.4

    download_videos(video_urls, "data/videos")
    extract_frames("data/videos", "data/frames")
    dataset = create_dataset("data/frames", class_mapping, conf_threshold, "youtube-dataset-od")
    analyze_and_export_dataset(dataset, "dataset")