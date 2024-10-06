from ultralytics import YOLO

# Load a model
model = YOLO("../models/yolov8s.pt")  # load model from local file

# Use the model
results = model.train(data="C:\\Users\\jayyp\\Desktop\\research\\ANPR\\yolo\\code\\data.yaml", 
                      epochs=11, 
                      batch=16,  # Adjust the batch size based on your GPU capacity
                      device='cpu',  # Ensure to use GPU (0 is typically the first GPU)
                      lr0=0.01,  # Learning rate
                      imgsz=640)  # Image size


print("Training complete. Results saved to:", results)  # save results to 'runs/detect/train'