import io
import requests
import supervision as sv
from PIL import Image
from inference import get_model
from datetime import datetime


model = get_model("rfdetr-base")

url = "https://media.roboflow.com/notebooks/examples/dog-2.jpeg"
start_time = datetime.now()
image = Image.open(io.BytesIO(requests.get(url).content))
predictions = model.infer(image, confidence=0.5)[0]
detections = sv.Detections.from_inference(predictions)

labels = [
    f"{prediction.class_name} {prediction.confidence:.2f}"
    for prediction in predictions.predictions
]

annotated_image = image.copy()
annotated_image = sv.BoxAnnotator().annotate(annotated_image, detections)
annotated_image = sv.LabelAnnotator().annotate(annotated_image, detections, labels)
end_time = datetime.now()

time_difference = (end_time - start_time).total_seconds() * 10**3
print("Execution time of program is: ", time_difference, "ms")
sv.plot_image(annotated_image)
