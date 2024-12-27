from PIL import Image, ImageDraw, ImageFont
import torch
from transformers import OwlViTProcessor, OwlViTForObjectDetection
import gc

processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")


def tensorToImage(image, boxes, labels, scores):
    print("inside tensortoimage")
    img = image
    draw = ImageDraw.Draw(img)
    filtered_scores = torch.tensor(scores)
    filtered_scores_list = filtered_scores.tolist()

    for box, label, score in zip(boxes, labels, filtered_scores_list):
        x0, y0, x1, y1 = box
        draw.rectangle([x0, y0, x1, y1], outline = "red", width = 3)
        draw.text((x0, y0), f"{label}: {str(round(score, 3))}", fill = "red", font = ImageFont.truetype("arial.ttf", size=20))
    print("saving image to floor.jpg")
    img.save("floor.jpg")

def objectdetect(image, text):
    image = image.copy()

    inputs = processor(text=text, images=image, return_tensors="pt")
    # import pdb; pdb.set_trace()
    with torch.no_grad():
        outputs = model(**inputs)



    target_size = torch.tensor([image.size[::-1]])
    print("before post process")
    results = processor.post_process(outputs=outputs, target_sizes=target_size)
    del outputs
    print("after post process")
    found = False
    boxes, scores, labels = results[0]["boxes"], results[0]["scores"], results[0]["labels"]
    score_threshold = 0.065



    filtered_boxes = []
    filtered_scores = []
    filtered_labels = []
    for box, score, label in zip(boxes, scores, labels):
        if score >= score_threshold:
            filtered_boxes.append(box)
            filtered_scores.append(score)
            filtered_labels.append(text[label])
            found = True

    del boxes, labels, scores

 
    # for box, score, label in zip(boxes, scores, labels):
    #     print(box, score, label)

    tensorToImage(image, filtered_boxes, filtered_labels, filtered_scores)
    del image
    gc.collect()

    
    return found, (filtered_labels, filtered_boxes)

