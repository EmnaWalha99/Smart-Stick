from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from transformers import YolosImageProcessor, YolosForObjectDetection
from PIL import Image
import torch
import io

# Initialize the FastAPI app
app = FastAPI()

# Load the YOLOS model and image processor
model = YolosForObjectDetection.from_pretrained('hustvl/yolos-tiny')
image_processor = YolosImageProcessor.from_pretrained("hustvl/yolos-tiny")

@app.post("/detect/")
async def detect_object(file: UploadFile = File(...)):
    # Read the uploaded image file
    image_data = await file.read()
    
    # Open the image from the byte data
    image = Image.open(io.BytesIO(image_data))
    
    # Preprocess the image and run object detection
    inputs = image_processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    
    # Get the predicted bounding boxes and labels
    target_sizes = torch.tensor([image.size[::-1]])  # Reverse width and height
    results = image_processor.post_process_object_detection(outputs, threshold=0.9, target_sizes=target_sizes)[0]
    
    detections = []
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        detections.append({
            "label": model.config.id2label[label.item()],
            "confidence": round(score.item(), 3),
            "bounding_box": box
        })
    
    # Return the detection results as a JSON response
    return JSONResponse(content={"detections": detections})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
