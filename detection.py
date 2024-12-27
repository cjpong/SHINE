from PIL import Image, ImageDraw, ImageFont
import torch
from transformers import OwlViTProcessor, OwlViTForObjectDetection

processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
texts = [["white cup"]]


def tensorToImage(image, boxes, labels, scores):
    img = image
    draw = ImageDraw.Draw(img)
    filtered_scores = torch.tensor(scores)
    filtered_scores_list = filtered_scores.tolist()

    for box, label, score in zip(boxes, labels, filtered_scores_list):
        x0, y0, x1, y1 = box
        draw.rectangle([x0, y0, x1, y1], outline = "red", width = 3)
        draw.text((x0, y0), f"{label}: {str(round(score, 3))}", fill = "red", font = ImageFont.truetype("arial.ttf", size=20))

    img.show()

def objectdetect(image, text):
    print(f"text is {text}")
    inputs = processor(text=text, images=image, return_tensors="pt")
    outputs = model(**inputs)

    target_size = torch.tensor([image.size[::-1]])
    results = processor.post_process(outputs=outputs, target_sizes=target_size)

    boxes, scores, labels = results[0]["boxes"], results[0]["scores"], results[0]["labels"]
    score_threshold = 0.07

    filtered_boxes = []
    filtered_scores = []
    filtered_labels = []
    found = False
    for box, score, label in zip(boxes, scores, labels):
        if score >= score_threshold:
            print("passed")
            filtered_boxes.append(box)
            filtered_scores.append(score)
            filtered_labels.append(text)
            found = True
    print(filtered_labels, filtered_boxes)
    return found, (filtered_labels, filtered_boxes)


