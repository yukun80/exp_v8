import os
from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO(model="weights/best.pt")
    # Source directory containing images
    source_dir = "datastes/test"
    # Destination directory to save results
    dest_dir = "results/CAS_test"
    # Ensure the destination directory exists
    os.makedirs(dest_dir, exist_ok=True)
    # Get a list of image files in the source directory
    image_files = [os.path.join(source_dir, f) for f in os.listdir(source_dir) if f.endswith((".jpg", ".png", ".jpeg", ".bmp", ".tif", ".tiff"))]

    results = model.predict(
        # source="datastes/CAS_yolo_dataset/images/test",  # source file/folder, 0 for webcam/
        source=image_files,
        stream=True,
        show_labels=False,
        show_conf=False,
        show_boxes=False,
        save_conf=True,
    )

    # Process results generator
    for i, result in enumerate(results):
        # Get the base filename without extension
        base_filename = os.path.basename(image_files[i]).split(".")[0]
        # Construct the save path
        save_path = os.path.join(dest_dir, f"{base_filename}_result.jpg")
        # Save the result to the specified path
        result.save(filename=save_path)
