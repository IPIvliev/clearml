from ultralytics import YOLO
from clearml import Dataset


# Load the model.
model = YOLO('yolov8n.pt')

# dataset_path = Dataset.get(
#     dataset_name='drone_v2', 
#     dataset_project='Yolov8'
# ).get_mutable_local_copy()

# print(dataset_path)

if __name__ == '__main__':
   # Training.
   results = model.train(
      data='e:/python/gl_drone_detection/datasets/drone_dataset_v2/data.yaml',
    #   data=dataset_path+'/data.yaml',
      imgsz=640,
      epochs=2,
      batch=8,
      name='yolov8n_custom_10e',
      device=0,
      workers=8)