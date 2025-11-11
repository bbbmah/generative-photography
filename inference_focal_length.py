import os
import torch
import logging
import argparse
import json
import numpy as np
import torch.nn.functional as F
from pathlib import Path
from omegaconf import OmegaConf
from torch.utils.data import Dataset
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, DDIMScheduler
from einops import rearrange

from genphoto.pipelines.pipeline_animation import GenPhotoPipeline
from genphoto.models.unet import UNet3DConditionModelCameraCond
from genphoto.models.camera_adaptor import CameraCameraEncoder, CameraAdaptor
from genphoto.utils.util import save_videos_grid

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_focal_length_embedding(focal_length_values, target_height, target_width, base_focal_length=24.0, sensor_height=24.0, sensor_width=36.0):
    device = 'cpu'
    focal_length_values = focal_length_values.to(device)
    f = focal_length_values.shape[0]  # Number of frames


    # Convert constants to tensors to perform operations with focal_length_values
    sensor_width = torch.tensor(sensor_width, device=device)
    sensor_height = torch.tensor(sensor_height, device=device)
    base_focal_length = torch.tensor(base_focal_length, device=device)

    # Calculate the FOV for the base focal length (min_focal_length)
    base_fov_x = 2.0 * torch.atan(sensor_width * 0.5 / base_focal_length)
    base_fov_y = 2.0 * torch.atan(sensor_height * 0.5 / base_focal_length)

    # Calculate the FOV for each focal length in focal_length_values
    target_fov_x = 2.0 * torch.atan(sensor_width * 0.5 / focal_length_values)
    target_fov_y = 2.0 * torch.atan(sensor_height * 0.5 / focal_length_values)

    # Calculate crop ratio: how much of the image is cropped at the current focal length
    crop_ratio_xs = target_fov_x / base_fov_x  # Crop ratio for horizontal axis
    crop_ratio_ys = target_fov_y / base_fov_y  # Crop ratio for vertical axis

    # Get the center of the image
    center_h, center_w = target_height // 2, target_width // 2

    # Initialize a mask tensor with zeros on CPU
    focal_length_embedding = torch.zeros((f, 3, target_height, target_width), dtype=torch.float32)  # Shape [f, 3, H, W]

    # Fill the center region with 1 based on the calculated crop dimensions
    for i in range(f):
        # Crop dimensions calculated using rounded float values
        crop_h = torch.round(crop_ratio_ys[i] * target_height).int().item()  # Rounded cropped height for the current frame
        crop_w = torch.round(crop_ratio_xs[i] * target_width).int().item()  # Rounded cropped width for the current frame

        # Ensure the cropped dimensions are within valid bounds
        crop_h = max(1, min(target_height, crop_h))
        crop_w = max(1, min(target_width, crop_w))

        # Set the center region of the focal_length embedding to 1 for the current frame
        focal_length_embedding[i, :,
        center_h - crop_h // 2: center_h + crop_h // 2,
        center_w - crop_w // 2: center_w + crop_w // 2] = 1.0

    return focal_length_embedding


class Camera_Embedding(Dataset):
    def __init__(self, focal_length_values, tokenizer, text_encoder, device, sample_size=[256, 384]):
        self.focal_length_values = focal_length_values.to(device)                                                 
        self.tokenizer = tokenizer                 
        self.text_encoder = text_encoder       
        self.device = device               
        self.sample_size = sample_size

    def load(self):

        if len(self.focal_length_values) != 7:
            raise ValueError("Expected 7 focal_length values")

        # Generate prompts for each focal length value and append focal_length information to caption
        prompts = []
        for fl in self.focal_length_values:
            prompt = f"<focal length: {fl.item()}>"
            prompts.append(prompt)
        

        # Tokenize prompts and encode to get embeddings
        with torch.no_grad():
            prompt_ids = self.tokenizer(
                prompts, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
            ).input_ids.to(self.device)

            encoder_hidden_states = self.text_encoder(input_ids=prompt_ids).last_hidden_state  # Shape: (f, sequence_length, hidden_size)
        

        # Calculate differences between consecutive embeddings (ignoring sequence_length)
        differences = []
        for i in range(1, encoder_hidden_states.size(0)):
            diff = encoder_hidden_states[i] - encoder_hidden_states[i - 1]
            diff = diff.unsqueeze(0)
            differences.append(diff)  

        # Add the difference between the last and the first embedding
        final_diff = encoder_hidden_states[-1] - encoder_hidden_states[0]
        final_diff = final_diff.unsqueeze(0)
        differences.append(final_diff)

        # Concatenate differences along the batch dimension (f-1)
        concatenated_differences = torch.cat(differences, dim=0) 
        frame = concatenated_differences.size(0)
        concatenated_differences = torch.cat(differences, dim=0)

        pad_length = 128 - concatenated_differences.size(1)
        if pad_length > 0:
        # Pad along the second dimension (77 -> 128), pad only on the right side
            concatenated_differences_padded = F.pad(concatenated_differences, (0, 0, 0, pad_length))


        ccl_embedding = concatenated_differences_padded.reshape(frame, self.sample_size[0], self.sample_size[1])
        ccl_embedding = ccl_embedding.unsqueeze(1)  
        ccl_embedding = ccl_embedding.expand(-1, 3, -1, -1)
        ccl_embedding = ccl_embedding.to(self.device)
        focal_length_embedding = create_focal_length_embedding(self.focal_length_values, self.sample_size[0], self.sample_size[1]).to(self.device)

        camera_embedding = torch.cat((focal_length_embedding, ccl_embedding), dim=1)
        return camera_embedding


def load_models(cfg):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    noise_scheduler = DDIMScheduler(**OmegaConf.to_container(cfg.noise_scheduler_kwargs))
    vae = AutoencoderKL.from_pretrained(cfg.pretrained_model_path, subfolder="vae").to(device)
    vae.requires_grad_(False)
    tokenizer = CLIPTokenizer.from_pretrained(cfg.pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(cfg.pretrained_model_path, subfolder="text_encoder").to(device)
    text_encoder.requires_grad_(False)
    unet = UNet3DConditionModelCameraCond.from_pretrained_2d(
        cfg.pretrained_model_path,
        subfolder=cfg.unet_subfolder,
        unet_additional_kwargs=cfg.unet_additional_kwargs
    ).to(device)
    unet.requires_grad_(False)

    camera_encoder = CameraCameraEncoder(**cfg.camera_encoder_kwargs).to(device)
    camera_encoder.requires_grad_(False)
    camera_adaptor = CameraAdaptor(unet, camera_encoder)
    camera_adaptor.requires_grad_(False)
    camera_adaptor.to(device)

    logger.info("Setting the attention processors")
    unet.set_all_attn_processor(
        add_spatial_lora=cfg.lora_ckpt is not None,
        add_motion_lora=cfg.motion_lora_rank > 0,
        lora_kwargs={"lora_rank": cfg.lora_rank, "lora_scale": cfg.lora_scale},
        motion_lora_kwargs={"lora_rank": cfg.motion_lora_rank, "lora_scale": cfg.motion_lora_scale},
        **cfg.attention_processor_kwargs
    )

    if cfg.lora_ckpt is not None:
        print(f"Loading the lora checkpoint from {cfg.lora_ckpt}")
        lora_checkpoints = torch.load(cfg.lora_ckpt, map_location=unet.device)
        if 'lora_state_dict' in lora_checkpoints.keys():
            lora_checkpoints = lora_checkpoints['lora_state_dict']
        _, lora_u = unet.load_state_dict(lora_checkpoints, strict=False)
        assert len(lora_u) == 0
        print(f'Loading done')

    if cfg.motion_module_ckpt is not None:
        print(f"Loading the motion module checkpoint from {cfg.motion_module_ckpt}")
        mm_checkpoints = torch.load(cfg.motion_module_ckpt, map_location=unet.device)
        _, mm_u = unet.load_state_dict(mm_checkpoints, strict=False)
        assert len(mm_u) == 0
        print("Loading done")
    
    if cfg.camera_adaptor_ckpt is not None:
        logger.info(f"Loading camera adaptor from {cfg.camera_adaptor_ckpt}")
        camera_adaptor_checkpoint = torch.load(cfg.camera_adaptor_ckpt, map_location=device)
        camera_encoder_state_dict = camera_adaptor_checkpoint['camera_encoder_state_dict']
        attention_processor_state_dict = camera_adaptor_checkpoint['attention_processor_state_dict']
        camera_enc_m, camera_enc_u = camera_adaptor.camera_encoder.load_state_dict(camera_encoder_state_dict, strict=False)

        assert len(camera_enc_m) == 0 and len(camera_enc_u) == 0
        _, attention_processor_u = camera_adaptor.unet.load_state_dict(attention_processor_state_dict, strict=False)
        assert len(attention_processor_u) == 0
        
        logger.info("Camera Adaptor loading done")
    else:
        logger.info("No Camera Adaptor checkpoint used")

    pipeline = GenPhotoPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=noise_scheduler,
        camera_encoder=camera_encoder
    ).to(device)
    pipeline.enable_vae_slicing()

    return pipeline, device

# 수정 sample1.gif 등으로 저장하기 위해 인덱스 값을 받도록 바꾸었다.
def run_inference(pipeline, tokenizer, text_encoder, base_scene, focal_length_list, output_dir, device, index, video_length=7, height=256, width=384):
    os.makedirs(output_dir, exist_ok=True)

    focal_length_list_str = focal_length_list
    focal_length_values = json.loads(focal_length_list_str)
    focal_length_values = torch.tensor(focal_length_values).unsqueeze(1)

    # Ensure camera_embedding is on the correct device
    camera_embedding = Camera_Embedding(focal_length_values, tokenizer, text_encoder, device).load()
    camera_embedding = rearrange(camera_embedding.unsqueeze(0), "b f c h w -> b c f h w")

    with torch.no_grad():
        sample = pipeline(
            prompt=base_scene,
            camera_embedding=camera_embedding,
            video_length=video_length,
            height=height,
            width=width,
            num_inference_steps=25,
            guidance_scale=8.0
        ).videos[0]
    # 수정 저장되는 파일에 번호 붙이도록 바꾸었다.     여기
    sample_save_path = os.path.join(output_dir, f'sample{index}.gif')
    save_videos_grid(sample[None, ...], sample_save_path)
    logger.info(f"Saved generated sample to {sample_save_path}")


def main(config_path, base_scene, focal_length_list):
    torch.manual_seed(42)
    cfg = OmegaConf.load(config_path)
    logger.info("Loading models...")
    pipeline, device = load_models(cfg)
    logger.info("Starting inference...")

    #수정, 평가용 프롬프트를 목록화.
    scene_list = [
        "A silver truck in an empty parking lot. The truck is parked in front of a Men's Warehouse store. Multiple traffic lights are visible in the distance. The scene is set on a sunny day. Bright sunlight.",
        "A row of cars in a tree-lined street. A black car and a red car are parked on the side of the road. A stop sign is visible in the background. The scene is set on a cloudy day. Soft, diffuse light.",
        "A white car in a large parking lot. The car is parked near a curb. The parking lot is surrounded by trees. The sky is visible above. Clear, bright daylight.",
        "A blue Nissan in a parking lot. The car is parked in a row, next to a blue parking sign. Trees surround the lot. The sky is clear in the background. Vivid daylight.",
        "A silver car on the side of a street. The car is waiting at a red light. Other cars are parked nearby. A stop sign is visible. The scene is set on a sunny day with a clear blue sky. Crisp, clear light.",
        "A white SUV on a street next to a tree. The street is lined with trees and buildings. The sky is visible. Soft daylight.",
        "A black and white photo of a street with two cars. A brick building is on the corner with a sign on the sidewalk. A street light illuminates the area. Monochromatic, well-lit.",
        "A white truck in a parking lot. The truck is parked next to a church, surrounded by a fence. A traffic cone is nearby. The sky is cloudy. Dim, overcast light.",
        "A large building with a brown roof on a sidewalk. A black trash can is located nearby. The building is surrounded by trees. The sky is overcast. Muted natural light.",
        "A large building with a sign on its side. The building is surrounded by a parking lot with a few cars. Trees are in the background. The scene is bright and sunny. Strong direct light.",
        "A white building with many windows, surrounded by green bushes and trees. People are sitting on a bench in front of the building. Sunlight shines through the trees, creating shade. Pleasant natural light.",
        "A large building with a palm tree in front. The street is empty, with a few cars parked along the side. The sky is blue with some clouds. Clear, bright light.",
        "A large building with a restaurant sign. The building is surrounded by a parking lot with cars. A tree is in front, and a potted plant is on the sidewalk. The sky is blue and clear. Inviting atmosphere.",
        "A white building with a brick facade and a large window. A white car is parked on the left side. The sky is visible in the background. Well-lit with natural light.",
        "A white building with a black roof at the end of a street. The building is surrounded by trees. A large window is on its side. Soft natural light.",
        "A large white building with many windows, surrounded by trees and greenery. Sunlight is shining on the building. Bright and sunny.",
        "An office cubicle with a gray wall. A gray filing cabinet is present. A black computer monitor, keyboard, and mouse are on the desk. The cubicle is in a large office space with a white wall and a brown floor. Standard office lighting.",
        "An office cubicle with a desk, chair, and computer. The cubicle is empty with a white tile wall. The office is dimly lit. Overhead lights.",
        "A long hallway with a white printer on a desk. The printer is connected to a computer. The hallway is lined with cubicles with white walls, separated by partitions. The hallway is brightly lit. Overhead lights.",
        "A large office cubicle with white walls, a white desk, and a whiteboard. The desk has a computer, keyboard, and mouse. Whiteboards and sticky notes cover the walls. Well-lit with overhead lights.",
        "A kitchen with two stainless steel refrigerators and a sink. The refrigerators are placed next to each other. The kitchen is dimly lit. Cozy atmosphere.",
        "A large, open office space with a white couch, a coffee table, and a television. The couch is positioned in the center. The television is mounted on the wall. A large window provides natural light. Well-lit with natural light.",
        "A small office with a desk, chair, and computer monitor. The desk has a keyboard and mouse. A window is in the background. Dimly lit. Natural light from window.",
        "A large flower pot with red flowers on a brick sidewalk. The pot is in front of a building. A bench is nearby. Brightly lit with sunlight."
    ]
    # 수정 추론시 run_inference 명령 하나만 하지만, for문으로 다 하도록 바꿈.
    index = 1
    for scene in scene_list:
        #run_inference(pipeline, pipeline.tokenizer, pipeline.text_encoder, base_scene, focal_length_list, cfg.output_dir, device=device)
        run_inference(pipeline, pipeline.tokenizer, pipeline.text_encoder, scene, focal_length_list, cfg.output_dir, device=device, index=index)
        index=index+1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration file")
    parser.add_argument("--base_scene", type=str, required=True, help="invariant scene caption as JSON string")
    parser.add_argument("--focal_length_list", type=str, required=True, help="focal_length values as JSON string")
    args = parser.parse_args()
    main(args.config, args.base_scene, args.focal_length_list)

    # usage example
    # python inference_focal_length.py --config configs/inference_genphoto/adv3_256_384_genphoto_relora_focal_length.yaml --base_scene "A cozy living room with a large, comfy sofa and a coffee table." --focal_length_list "[25.0, 35.0, 45.0, 55.0, 65.0]"

