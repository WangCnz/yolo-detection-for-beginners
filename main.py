from ultralytics import YOLO

# # set YOLO_PROJECT_DIR= "D:/lane"
#
# load yolov8m model
model = YOLO("yolov8m.pt")

# # resume training
# results = model.train(
#     resume=True,
#     device="cpu",
#     workers=0
# )

# training
results = model.train(
    data="config.yaml",
    project="runs/detect",
    imgsz=640,
    epochs=250,
    batch=4,
    lr0=0.0001,
    momentum=0.9,
    weight_decay=0.05,
    warmup_epochs=10,
    augment=True,
    optimizer="AdamW",
    device="cpu",
    save=True,
    val=True,
    save_period=10,
    name="my_train",
    workers=0,
    dropout=0.01
                      )

# validating
metrics = model.val(
    data="config.yaml",
    project="runs/detect",
    imgsz=640,
    split="val",
    device="cpu",
    name="my_val"
)

# model = YOLO("../lane/runs/detect/my_train/weights/best.pt")

# detect and save images
image_results = model.predict(
    source="dataset/images/test/*png",
    imgsz=[1280, 375],
    conf=0.25,
    augment=True,
    save=True,
    save_txt=True,
    save_conf=True,
    project="runs/detect",
    name="image_predictions"
)

# detect and save a video
video_result = model.predict(
    source="dataset/video/cars_on_road.mp4",
    imgsz=[1280, 736],
    conf=0.25,
    save=True,
    project="runs/detect",
    name="video_predictions"
)
