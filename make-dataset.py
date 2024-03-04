import yt_dlp
import cv2
from urllib.parse import urlparse
from urllib.parse import parse_qs
import glob

import torch

import fiftyone as fo
from fiftyone import ViewField as F
import fiftyone.utils.random as four

import pandas as pd


# Взяти за джерело даних youtube та зібрати збалансований датасет для тренування та валідації моделі для вирішення задачі object detection для трьох класів [person, car, pet] (person - люди, car - всі дорожні авто, pet - всі домашні тварини)
# Реалізація збору даних з ресурсу на вибір, плюсом буде реалізований пайплайн;
# Розбити відео на фрейми, в результаті має буде до 5к зображень;
# Розмітку зробити за допомогою загально доступної object detection моделі (автолейблінг);
# Датасет розподілити на train/val частини;
# Реалізувати код для аналізу створеного датасету, який відображає розподіл даних; 
# Опрацювання датасета реалізувати через fiftyone і зберегти у форматі COCO.

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

## Downloading of videos
with yt_dlp.YoutubeDL({
    "outtmpl": "data/videos/%(autonumber)s [%(id)s] %(title)s.%(ext)s",
    "format":"135", #480p
    #"format":"136", #720p #'best'
    "progress_hooks": [lambda d: print(f"{d['status']}%")],
    "quiet": True
    }) as ydlp:
    ydlp.download(video_urls)

# Splitting videos to frames
for i, url in enumerate(video_urls):
    video_id = parse_qs(urlparse(url).query)['v'][0]
    video_path = [item for item in glob.glob("data/videos/*") if f"[{video_id}]" in item][0]
    print(f"\n{i+1} {video_id}")
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    totalNoFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    skipNoFrames = max(fps, -(totalNoFrames // -(5000))) # up to 5k frames

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % skipNoFrames == 0:
            cv2.imwrite(f"data/frames/{i+1:05} [{video_id}] {frame_count:07}.jpg", frame)
            # Print progress
            progress = (frame_count + 1) / totalNoFrames * 100
            print(f"Progress: {progress:.2f}%", end="\r")

        frame_count += 1
    cap.release()

# Load pretrained model fo data labeling
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

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

# Set confidence threshold
conf_threshold = 0.4

# Create samples for your data
frame_paths = glob.glob("data/frames/*.jpg")
samples = []
for filepath in frame_paths:
    sample = fo.Sample(filepath = filepath)
    
    # Preprocess the image
    results = model(filepath)
    boxesn = results.pandas().xyxyn[0]
    # Keep only necessary detections
    boxesn = boxesn[boxesn['class'].isin(class_mapping.keys()) & (boxesn['confidence']>=conf_threshold)].replace({"class": class_mapping})
    boxesn = boxesn.assign(width = lambda r: r.xmax - r.xmin, height = lambda r: r.ymax - r.ymin)

    # Convert detections to FiftyOne format
    detections = []
    for i, obj in boxesn.iterrows():
        detections.append(
            fo.Detection(label = obj["class"], 
                         bounding_box = obj[['xmin', 'ymin', 'width', 'height']].tolist(), 
                         confidence = obj['confidence'], 
                         index = i)
        )
    
    sample["ground_truth"] = fo.Detections(detections=detections)
    samples.append(sample)

# Create dataset
dataset = fo.Dataset("youtube-dataset-od")
dataset.add_samples(samples)

# Train/validation split
four.random_split(dataset, {"train": 0.8, "val": 0.2})

# Exploratory analysis of data distribution in a dataset
def ds_dist_analysis(ds):
    samples_cnt = ds.count()
    labels_cnt = ds.count("ground_truth.detections.label")
    print('Samples count:', samples_cnt, '<br>')
    print('Labels count:', labels_cnt)

    dist_analysis = pd.Series(ds.count_values("ground_truth.detections.label"), name="label_counts").rename_axis("class").to_frame()
    dist_analysis["label_share"] = dist_analysis['label_counts'] / labels_cnt
    dist_analysis["samples_counts"] = pd.Series(
        {k: ds.filter_labels("ground_truth", fo.ViewField("label").is_in([k])).count()
        for k in dist_analysis.index}
        )
    dist_analysis["samples_share"] = dist_analysis["samples_counts"] / samples_cnt

    print('\nDistribution labels and samples between classes:')
    print(dist_analysis.to_markdown())

print("\n# Whole dataset")
ds_dist_analysis(dataset)
print("\n## Train split")
ds_dist_analysis(dataset.match_tags("train"))
print("\n## Val split")
ds_dist_analysis(dataset.match_tags("val"))

# Export as COCO format
dataset.export(
    export_dir = "dataset",
    dataset_type = fo.types.COCODetectionDataset,
    label_field = "ground_truth",
)