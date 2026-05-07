from transformers import AutoProcessor, AutoModelForImageTextToText
import torch
import re
from PIL import Image
import requests

model_id="allenai/Molmo2-8B"

# load the processor
processor = AutoProcessor.from_pretrained(
    model_id,
    trust_remote_code=True,
    dtype="auto",
    device_map="auto",
    token=True
)

# load the model
model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    trust_remote_code=True,
    dtype="auto",
    device_map="auto",
    token=True
)

COORD_REGEX = re.compile(rf"<(?:points|tracks).*? coords=\"([0-9\t:;, .]+)\"/?>")
FRAME_REGEX = re.compile(rf"(?:^|\t|:|,|;)([0-9\.]+) ([0-9\. ]+)")
POINTS_REGEX = re.compile(r"([0-9]+) ([0-9]{3,4}) ([0-9]{3,4})")

def _points_from_num_str(text, image_w, image_h, extract_ids=False):
    all_points = []
    for points in POINTS_REGEX.finditer(text):
        ix, x, y = points.group(1), points.group(2), points.group(3)
        # our points format assume coordinates are scaled by 1000
        x, y = float(x)/1000*image_w, float(y)/1000*image_h
        if 0 <= x <= image_w and 0 <= y <= image_h:
            yield ix, x, y


def extract_multi_image_points(text, image_w, image_h, extract_ids=False):
    """Extract pointing coordinates as a flattened list of (frame_id, x, y) triplets from model output text."""
    all_points = []
    if isinstance(image_w, (list, tuple)) and isinstance(image_h, (list, tuple)):
        assert len(image_w) == len(image_h)
        diff_res = True
    else:
        diff_res = False
    for coord in COORD_REGEX.finditer(text):
        for point_grp in FRAME_REGEX.finditer(coord.group(1)):
            frame_id = int(point_grp.group(1)) if diff_res else float(point_grp.group(1))
            w, h = (image_w[frame_id-1], image_h[frame_id-1]) if diff_res else (image_w, image_h)
            for idx, x, y in _points_from_num_str(point_grp.group(2), w, h):
                if extract_ids:
                    all_points.append((frame_id, idx, x, y))
                else:
                    all_points.append((frame_id, x, y))
    return all_points

# process the image and text
images = [
    Image.open(requests.get("https://storage.googleapis.com/oe-training-public/demo_images/boat1.jpeg", stream=True).raw),
    Image.open(requests.get("https://storage.googleapis.com/oe-training-public/demo_images/boat2.jpeg", stream=True).raw)
]

messages = [
    {
        "role": "user",
        "content": [
            dict(type="text", text="Point to the boats"),
            dict(type="image", image=images[0]),
            dict(type="image", image=images[1]),
        ],
    }
]

inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
)

inputs = {k: v.to(model.device) for k, v in inputs.items()}

# generate output
with torch.inference_mode():
    generated_ids = model.generate(**inputs, max_new_tokens=2048)

# only get generated tokens; decode them to text
generated_tokens = generated_ids[0, inputs['input_ids'].size(1):]
generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

points = extract_multi_image_points(
    generated_text,
    [images[0].width, images[1].width],
    [images[0].height, images[1].height],
)
print(points)