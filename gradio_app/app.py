import numpy as np
import os
import time
import sys
import torch
import zipfile
import gradio as gr
import u2net_load
import u2net_run
from rembg import remove
from PIL import Image, ImageOps
from predict_pose import generate_pose_keypoints


# Use GPU if available
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"####Using {device}.#####") 



# Make directories
os.system("mkdir ./Data_preprocessing")
os.system("mkdir ./Data_preprocessing/test_color")
os.system("mkdir ./Data_preprocessing/test_colormask")
os.system("mkdir ./Data_preprocessing/test_edge")
os.system("mkdir ./Data_preprocessing/test_img")
os.system("mkdir ./Data_preprocessing/test_label")
os.system("mkdir ./Data_preprocessing/test_mask")
os.system("mkdir ./Data_preprocessing/test_pose")
os.system("mkdir ./inputs")
os.system("mkdir ./inputs/img")
os.system("mkdir ./inputs/cloth")
os.system("mkdir ./saved_models/")
os.system("mkdir ./saved_models/u2net")
os.system("mkdir ./saved_models/u2netp")
os.system("mkdir ./pose")
os.system("mkdir ./checkpoints")


# Get pose model
if not os.path.exists("./pose/pose_deploy_linevec.prototxt"):
    os.system("wget -O ./pose/pose_deploy_linevec.prototxt https://github.com/hasibzunair/fifa-demo/releases/download/v1.0/pose_deploy_linevec.prototxt")
if not os.path.exists("./pose/pose_iter_440000.caffemodel"):
    os.system("wget -O ./pose/pose_iter_440000.caffemodel https://github.com/hasibzunair/fifa-demo/releases/download/v1.0/pose_iter_440000.caffemodel")

# For segmentation mask generation
if not os.path.exists("lip_final.pth"):
    os.system("wget https://github.com/hasibzunair/fifa-demo/releases/download/v1.0/lip_final.pth")

# Get U-2-Net weights
if not os.path.exists("saved_models/u2netp/u2netp.pth"):
    os.system("wget -P saved_models/u2netp/ https://github.com/hasibzunair/fifa-demo/releases/download/v1.0/u2netp.pth")
if not os.path.exists("saved_models/u2net/u2net.pth"):
    os.system("wget -P saved_models/u2net/ https://github.com/hasibzunair/fifa-demo/releases/download/v1.0/u2net.pth")

# Get model checkpoints
if not os.path.exists("./checkpoints/decavtonfifapretrain/"):
    os.system("wget -O ./checkpoints/decavtonfifapretrain.zip https://github.com/hasibzunair/vton-demo/releases/download/v1.0/decavtonfifapretrain.zip")
    with zipfile.ZipFile('./checkpoints/decavtonfifapretrain.zip', 'r') as zip_ref:
        zip_ref.extractall('./checkpoints/')

print("########################Setup done!########################")

# Load U-2-Net model
print(f"####Using {device}.#####")
u2net = u2net_load.model(model_name = 'u2netp')

def composite_background(img_mask, person_image_path, tryon_image_path):
    """Put background back on the person image after tryon."""
    person = np.array(Image.open(person_image_path))
    # tryon image
    tryon = np.array(Image.open(tryon_image_path))
    # persom image mask from rembg
    p_mask = np.array(img_mask)
    # make binary mask
    p_mask = np.where(p_mask>0, 1, 0)
    # invert mask
    p_mask_inv = np.logical_not(p_mask)
    # make bg without person
    background = person * np.stack((p_mask_inv, p_mask_inv, p_mask_inv), axis=2)
    # make tryon image without background
    tryon_nobg = tryon * np.stack((p_mask, p_mask, p_mask), axis=2)
    tryon_nobg = tryon_nobg.astype("uint8")
    # composite 
    tryon_with_bg = np.add(tryon_nobg, background)
    tryon_with_bg_pil = Image.fromarray(np.uint8(tryon_with_bg)).convert('RGB')
    tryon_with_bg_pil.save("results/test/try-on/tryon_with_bg.png")


# Main inference function
def inference(clothing_image, person_image, retrieve_bg):
    """
    Do try-on!
    """
    remove_bg = "no"
    
    # Read cloth and person images
    cloth = Image.open(clothing_image) # cloth
    person = Image.open(person_image) # person
    # Save cloth and person images in "input" folder
    cloth.save(os.path.join("inputs/cloth/cloth.png"))
    person.save(os.path.join("inputs/img/person.png"))

    ############## Clothing image pre-processing
    cloth_name = 'cloth.png'
    cloth_path = os.path.join('inputs/cloth', sorted(os.listdir('inputs/cloth'))[0])
    cloth = Image.open(cloth_path)
    # Resize cloth image
    cloth = ImageOps.fit(cloth, (192, 256), Image.BICUBIC).convert("RGB")
    # Save resized cloth image
    cloth.save(os.path.join('Data_preprocessing/test_color', cloth_name))
    # 1. Get binary mask for clothing image
    u2net_run.infer(u2net, 'Data_preprocessing/test_color', 'Data_preprocessing/test_edge')

    ############## Person image pre-processing
    start_time = time.time()
    # Person image
    img_name = 'person.png'
    img_path = os.path.join('inputs/img', sorted(os.listdir('inputs/img'))[0])
    img = Image.open(img_path)
    if remove_bg == "yes":
        # Remove background
        img = remove(img, alpha_matting=True, alpha_matting_erode_size=15)
        print("Removing background from person image..")
    img = ImageOps.fit(img, (192, 256), Image.BICUBIC).convert("RGB")
    # Get binary from person image, used in def_composite_background
    img_mask = remove(img, alpha_matting=True, alpha_matting_erode_size=15, only_mask=True)
    img_path = os.path.join('Data_preprocessing/test_img', img_name)
    img.save(img_path)
    resize_time = time.time()
    print('Resized image in {}s'.format(resize_time-start_time))

    # 2. Get parsed person image (test_label), uses person image
    os.system("python Self-Correction-Human-Parsing-for-ACGPN/simple_extractor.py --dataset 'lip' --model-restore 'lip_final.pth' --input-dir 'Data_preprocessing/test_img' --output-dir 'Data_preprocessing/test_label'")
    parse_time = time.time()
    print('Parsing generated in {}s'.format(parse_time-resize_time))
    
    # 3. Get pose map from person image
    pose_path = os.path.join('Data_preprocessing/test_pose', img_name.replace('.png', '_keypoints.json'))
    generate_pose_keypoints(img_path, pose_path)
    pose_time = time.time()
    print('Pose map generated in {}s'.format(pose_time-parse_time))
    os.system("rm -rf Data_preprocessing/test_pairs.txt")
    
    # Format: person, cloth image
    with open('Data_preprocessing/test_pairs.txt','w') as f:
        f.write('person.png cloth.png')
    
    # Do try-on
    os.system("python test.py --name decavtonfifapretrain")
    tryon_image = Image.open("results/test/try-on/person.png")
    print("Size of image is: ", tryon_image.size)
    
    # Return try-on with background added back on the person image
    if retrieve_bg == "yes":
        composite_background(img_mask, 'Data_preprocessing/test_img/person.png',
                     'results/test/try-on/person.png')
        return os.path.join("results/test/try-on/tryon_with_bg.png")
    # Return only try-on result
    else:
        return os.path.join("results/test/try-on/person.png")


title = "Image based Virtual Try-On"
description = "This is a demo for an image based virtual try-on system. It generates a synthetic image of a person wearing a target clothing item. To use it, simply upload your clothing item and person images, or click one of the examples to load them. This demo currently uses a temporary GPU. You can always run the demo locally, of course on a machine with a GPU!"
article = "<p style='text-align: center'><a href='will_be_added' target='_blank'>Fill in Fabrics: Body-Aware Self-Supervised Inpainting for Image-Based Virtual Try-On (Under Review!)</a> | <a href='https://github.com/dktunited/fifa_demo' target='_blank'>Github</a></p>"
thumbnail = None # "./pathtothumbnail.png"

# todos
# train model with background removed then add feature, also add remove_bg in inferene()
# add gr.inputs.Radio(choices=["yes","no"], label="Remove background from the person image?", type='index') in inputs
    
gr.Interface(
    inference,
    [gr.inputs.Image(type='file', label="Clothing Image"),
     gr.inputs.Image(type='file', label="Person Image"),
     gr.inputs.Radio(choices=["yes","no"], label="Retrieve original background from the person image?", type='index')],
    gr.outputs.Image(type="file", label="Predicted Output"),
    examples=[["./sample_images/1/cloth.jpg", "./sample_images/1/person.jpg", "yes"],
              ["./sample_images/2/cloth.jpg", "./sample_images/2/person.jpg", "no"]],
    title=title,
    description=description,
    article=article,
    allow_flagging=False,
    analytics_enabled=False,
    thumbnail=thumbnail,
    ).launch(debug=True, enable_queue=True)