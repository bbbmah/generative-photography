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

    sample_save_path = os.path.join(output_dir, f'sample{index}.gif')
    save_videos_grid(sample[None, ...], sample_save_path)
    logger.info(f"Saved generated sample to {sample_save_path}")


def main(config_path, base_scene, focal_length_list):
    torch.manual_seed(42)
    cfg = OmegaConf.load(config_path)
    logger.info("Loading models...")
    pipeline, device = load_models(cfg)
    logger.info("Starting inference...")

    scene_list = [
        "A silver truck in an empty parking lot. The truck is parked in front of a retail store building. A row of vertical street lights is visible in the distance. The scene is set on a sunny day. Bright sunlight.",
        "A row of cars in a tree-lined street. A black car and a red car are parked parallel to the curb. A stop sign and distinct curbside lines are visible in the background. The scene is set on a cloudy day. Soft, diffuse light.",
        "A white car in a large parking lot. The car is parked near a straight curb. The parking lot boundary is surrounded by a low wall. The sky is visible above. Clear, bright daylight.",
        "A sedan car in a parking lot. The car is parked in a row, next to a blue vertical parking sign. Straight parking lines and trees surround the lot. The sky is clear in the background. Vivid daylight.",
        "A silver car on the side of a street. The car is waiting at a red light near a crosswalk. Straight road markings are visible. A stop sign is visible. The scene is set on a sunny day with a clear blue sky. Crisp, clear light.",
        "A white SUV on a street next to a large tree. The street is lined with straight buildings. The horizontal skyline is visible. Soft daylight.",
        "A black and white photo of a street with two cars. A straight brick building is on the corner. A vertical street light illuminates the area. Monochromatic, well-lit.",
        "A white truck in a parking lot. The truck is parked next to a church-like building, surrounded by a metal fence. A traffic cone is near the truck. The sky is cloudy. Dim, overcast light.",
        "A white hatchback in an asphalt parking lot. The car is parked diagonally in a clearly marked spot. A long, straight fence runs along the lot's edge. The sky is deep blue and clear. Harsh midday light.",
        "A dark blue sedan on a suburban street. The car is parked next to a straight white curb. A vertical traffic pole stands nearby. The scene is slightly foggy in the morning. Cool, soft light.",
        "A red pickup truck in a gravel lot. The truck is centered in the frame, near a horizontal retaining wall. The parking area is defined by wooden posts. The sky is yellow at sunset. Warm, low light.",
        "A black vintage car on a wide city street. The car is stopped at a crosswalk with thick white lines. Tall, rectangular buildings line the background. The scene is overcast and gray. Flat, diffuse light.",
        "A silver minivan in a multi-story parking garage. The van is parked between two vertical concrete pillars. The ceiling lights cast strong geometric shadows. The garage is empty. Fluorescent, harsh lighting.",
        "An empty street with clear double yellow lines. A few utility poles run parallel to the road. A rectangular sign hangs overhead. The scene is set just after a rain shower. Wet, reflective light.",
        "A green sports car parked at an industrial loading dock. The car is aligned with the dock's straight edge. Large, rectangular metal structures dominate the background. The sky is clear. Direct afternoon light.",
        "A beige delivery van on a narrow road. The van is next to a low, straight brick wall. A straight asphalt path leads into the distance. The scene is late evening. Dim, twilight.",
        "A black motorcycle parked on a sidewalk. The sidewalk tiles form a repeating grid pattern. A clear glass window facade reflects the light. The sun is high. Bright, sharp light.",
        "A taxi cab turning a corner. The corner has sharp 90-degree curbing and sidewalk lines. A pedestrian crosswalk is visible. The scene is in motion. Vivid, moving light.",
        "A light gray electric car in an underground parking lot. The car is positioned under a long, straight light strip. The ceiling beams create parallel lines. The light is artificial and uniform. Consistent fluorescent light.",
        "A long trailer truck parked on a highway shoulder. The truck is parallel to the straight white shoulder line. The highway barriers form a strong leading line. The scene is hazy. Soft, distant sunlight.",
        "An orange compact car parked in a field with tall grass. The field is bordered by a straight wooden fence line. A distant rectangular barn stands on the horizon. The sun is setting. Golden hour light.",
        "A white luxury sedan in a cobblestone driveway. The driveway is bordered by a low, straight stone wall. The house facade features vertical pillars. The day is cloudy. Muted, even light.",
        "A police car parked beside a tall glass building. The car is next to the straight edge of the building facade. The windows reflect the blue sky in a grid pattern. The sky is clear. Reflective, sharp light.",
        "A construction vehicle on a dirt road. The road is bordered by stacked concrete blocks forming straight lines. The horizon is flat. The scene is dusty. Hot, diffuse sunlight.",
        "A black bicycle leaning against a vertical metal railing. The railing is attached to a long, straight bridge over water. The water forms a clear horizontal line. The scene is breezy. Clear daylight.",
        "A row of identical grey SUVs in a large dealership lot. The vehicles are aligned perfectly along white painted lines. The building in the background has horizontal louvers. The sky is partly cloudy. High contrast light.",
        "A blue cargo container loaded onto a flatbed truck. The truck is stopped under a highway overpass. The concrete columns and beams form strong vertical and horizontal lines. The light is shadowed. Dark, under-bridge shadow.",
        "A yellow school bus parked outside a rectangular school building. The bus is aligned with the building's straight edge. The school fence forms a clear boundary. The scene is morning. Clean, early daylight.",
        "A small motorized scooter parked on a checkered tile path. The path leads toward a square entrance archway. The building is made of smooth stone. The scene is humid. Bright, wet light.",
        "A vintage convertible car driving on a straight rural road. The road is flanked by vertical telephone poles. The sky is wide and blue. Scenic, open light.",
        "A large black dumpster placed against a concrete service alley wall. The wall is straight, featuring repeated horizontal grooves. The scene is industrial. Harsh, directional lighting.",
        "A line of traffic cones defining a straight lane closure. The cones are centered on a wide asphalt road. The distant buildings have flat roofs. The sky is white and bright. Overexposed daylight.",

        "A large building with a brown roof on a sidewalk. A vertical trash can is located near the building's corner. The building is surrounded by trees. The vertical lines of the building corner are visible. The sky is overcast. Muted natural light.",
        "A large building with a vertical sign on its side. The building is surrounded by a parking lot with a few cars. Straight edge rooflines are in the background. The scene is bright and sunny. Strong direct light.",
        "A large white building with many vertical windows. The building is surrounded by a straight green hedge. A few benches are placed in front of the building. The sunlight creates strong shadows. Pleasant natural light.",
        "A large building with a palm tree in front. The street is empty, with a few cars parked parallel to the curb. The straight horizon line is visible. The sky is blue with some clouds. Clear, bright light.",
        "A large building with a restaurant sign. The building is surrounded by a parking lot with cars. A straight sidewalk runs in front of the building. The sky is blue and clear. Inviting atmosphere.",
        "A white building with a brick facade and a large rectangular window. A white car is parked parallel to the building. The straight facade lines are clear. The sky is visible in the background. Well-lit with natural light.",
        "A white building with a flat black roof at the end of a straight street. The building is surrounded by trees. The straight roof and street lines define the scene. Soft natural light.",
        "A large white building with many windows. The building is surrounded by trees and straight greenery. Sunlight is shining directly on the facade, emphasizing verticality. Bright and sunny.",
        "A tall brick office building with repeated vertical window panes. The building stands next to a straight paved sidewalk. A rectangular advertising banner is attached to the side. The scene is foggy. Muted, atmospheric light.",
        "A modern white commercial building with horizontal glass sections. The structure is set back from a clear, straight curb. The foreground features a low, straight stone retainer wall. The sun is high. Reflective, sharp light.",
        "An old industrial warehouse with a corrugated metal roof. The building's walls meet at a sharp 90-degree corner. A parallel railroad track runs nearby. The scene is afternoon. Strong directional light.",
        "A municipal library building with imposing vertical columns. The front steps are wide and straight, leading to the entrance. The sky is dark with rain clouds. Heavy, dramatic light.",
        "A residential apartment complex featuring repeating square balconies. The complex is bordered by a uniform height privacy fence. A straight concrete driveway leads to the street. The sun is setting. Soft, orange light.",
        "A large concrete stadium structure with exposed beams and pillars. The stadium wall forms a long, uninterrupted straight line. The asphalt foreground is empty. The sky is clear. Clean, architectural light.",
        "A long, low-rise commercial strip mall with uniform rectangular storefronts. A straight metal awning extends over the sidewalk. The parking lot is wide and flat. The scene is set on a clear morning. Crisp, low-angle sunlight.",
        "A futuristic building with angled but straight glass panels. The ground level is defined by a perfectly straight base line. The surrounding plaza has a geometric tile pattern. The sky is blue. High-tech, sharp light.",
        "A neoclassical government building with symmetrical vertical columns. The building is approached by a wide, straight walkway. The surrounding area is paved with square stones. The scene is very bright. Intense direct light.",
        "A tall, thin water tower standing against the sky. The tower's structure features prominent vertical and horizontal crossbeams. The ground is flat grass. The sky is hazy. Distant, diffused light.",
        "A large glass bank building reflecting the clouds. The vertical window mullions are dominant. The entrance is marked by a straight marble step. The scene is midday. High glare light.",
        "A chain-link fence running in a straight line across a grassy field. The fence is taut and the posts are vertical. A small utility building with a flat roof is visible. The scene is calm. Even daylight.",
        "A small, square utility shed painted deep red. The shed is situated next to a straight wooden board fence. The roofline is perfectly horizontal. The background features dense trees. The scene is shadowed. Deep contrast light.",
        "A large, modern art museum with white, interlocking cuboid sections. The facade is defined by sharp, straight edges and corners. The foreground is a flat, empty plaza. The sky is dark gray. Soft, exhibition light.",
        "A pedestrian overpass made of straight concrete segments. The railing features a repeating horizontal pattern. The bridge spans a wide, straight road. The sun is low. Elongated shadows.",
        "A long, straight row of garage doors on a residential block. Each door is rectangular and aligned perfectly. The concrete driveway is flat. The sky is bright. Standard sunlight.",
        "A small telephone booth standing alone on a street corner. The booth structure has vertical glass panels and straight metal framing. The adjacent building is low and long. The light is diffused. Mild afternoon light.",
        "A loading bay area with a large, rectangular, open dock door. The concrete floor is flat and level. The overhead lighting is mounted in straight lines. The scene is industrial. Harsh artificial light.",
        "A park pavilion with a flat roof supported by four vertical wood pillars. The pavilion structure is perfectly square. A straight gravel path leads to it. The scene is outdoors. Soft morning light.",
        "A large power substation featuring numerous straight metal support frames. The structures form intersecting right angles. The scene is open. High visibility light.",
        "A simple concrete block wall running horizontally across the frame. The wall has a uniform, rough texture. The ground is flat and featureless. The scene is isolated. Neutral daylight.",
        "A row of vending machines lined up against a plain exterior wall. The machines are rectangular and parallel to the wall. The sidewalk is straight. The sun is strong. Sharp contrast light.",
        "A modern school building featuring long, horizontal bands of windows. The roofline is flat and straight. The front lawn is clearly edged by a low curb. The sky is mostly clear. Bright, educational light.",
        "A two-story house with straight vertical siding and a prominent horizontal trim line. The house is centered, emphasizing its symmetry. The foreground lawn is flat. The scene is bright and quiet. Even daylight.",

        "An office cubicle with a straight gray wall. A gray filing cabinet is present. A black computer monitor and keyboard are on the desk. The cubicle walls define a clear rectangular space. Standard office lighting.",
        "An office cubicle with a desk, chair, and computer. The cubicle is empty with a straight white tiled wall. The office is dimly lit, emphasizing the wall boundaries. Overhead lights.",
        "A long, straight hallway lined with cubicles. A white printer is on a desk along the wall. The hallway is brightly lit, emphasizing the convergence lines. Overhead lights.",
        "A large office cubicle with white walls, a straight white desk, and a whiteboard. A computer, keyboard, and whiteboard cover the walls. The scene emphasizes clear 90-degree corners. Well-lit with overhead lights.",
        "An office cubicle with a beige fabric wall. A vertical filing cabinet is placed against the back wall. A large, square computer monitor sits on the straight desk surface. The office is brightly lit. Uniform fluorescent light.",
        "A cubicle workspace with a tall, partition wall. The horizontal surface of the desk is clear and straight. A telephone and a stack of books sit neatly in the corner. The light is focused on the desk. Direct overhead light.",
        "A row of adjacent cubicles with low, straight dividing walls. The central aisle between the rows is straight and long. The floor features uniform carpet squares. The office is well-lit. Consistent, flat lighting.",
        "An empty cubicle viewed from the side, emphasizing the parallel walls. The desk surface extends linearly into the frame. The ceiling lights cast strong geometric shadows. The garage is empty. Fluorescent, harsh lighting.",
        "A corner cubicle with two perpendicular white walls. A large L-shaped desk defines the workspace. The computer monitor is rectangular and centered. The office is quiet. Standard fluorescent light.",
        "A long office hallway with a single straight line of light fixtures recessed into the ceiling. The walls are smooth and feature a repeated horizontal chair rail. The scene is empty. Clean, directional light.",
        "A meeting room cubicle defined by three glass panels framed in black metal. A long, rectangular table is in the center. The light reflects off the straight glass surfaces. The room is modern. Bright, diffused light.",
        "A cubicle with a dark gray wall and a shelf unit featuring vertical dividers. A keyboard is centered on the straight desk edge. The floor tiles are large and square. The light is slightly dim. Mild overhead light.",
        "An empty desk inside a cubicle with a white laminate surface. The partition wall is perfectly vertical and smooth. A small trash can is tucked into the corner. The light is uniform. Standard office light.",
        "A bank of cubicles extending into the distance, emphasizing the converging lines. The walls are covered in a uniform grey paneling. The ceiling grid forms a visible pattern. The light is strong and internal. High visibility lighting.",

        "A kitchen with two large stainless steel refrigerators and a sink. The appliances are placed next to a straight counter. The kitchen is dimly lit, emphasizing the cabinet lines. Cozy atmosphere.",
        "A large, open office space with a white couch and a coffee table. The room features straight walls and a large rectangular window. The window and walls define the scene's geometry. Well-lit with natural light.",
        "A small office with a straight desk, chair, and computer monitor. A large rectangular window is in the background. The room is dimly lit, emphasizing the window frame's straight edges. Natural light from window.",
        "A residential kitchen with straight white cabinets and stainless steel handles. The counter surface is long and flat. A rectangular window is above the sink. The scene is mid-morning. Bright natural light.",
        "A living room with a large rectangular television mounted on a plain wall. The sofa is placed parallel to the wall. A low, square coffee table is in the center. The room is well-lit. Even ambient light.",
        "A small dining area with a rectangular wooden table and four chairs. The table edges are straight and parallel to the walls. A picture frame hangs symmetrically on the wall. The scene is focused. Soft directional light.",
        "A home hallway with a straight wooden floor and a vertical row of coat hooks. The baseboards and ceiling lines are clearly defined. A closed rectangular door is at the end. The light is low. Dim, cozy light.",
        "A bedroom featuring a large rectangular dresser and a mirror. The dresser surface is straight and level. The carpet has a faint linear pattern. The light is warm. Late afternoon sunlight.",
        "An open-plan loft apartment with a prominent straight staircase railing. The railing runs linearly up the wall. The ceiling beams are exposed and parallel. The scene is spacious. Bright overhead light.",
        "A study room with a long, straight bookshelf running along one wall. The books are aligned vertically and horizontally. A simple wooden desk is nearby. The room is quiet. Mild daylight from a side window.",
        "A laundry room with white vertical washer and dryer units. The units are placed against a straight tiled wall. The counter above the machines is flat. The light is strong and white. Harsh utility light.",

        "A large planter pot with red flowers sits on a brick sidewalk. The sidewalk features clear straight lines leading toward a building facade. The straight lines of the brick pattern and building are distinct. Brightly lit with sunlight.",
        "A tall, straight concrete pillar supporting an elevated highway. The pillar stands in an open, paved area. The light from the setting sun illuminates the pillar's vertical edge. Golden hour light.",
        "A long, straight brick wall covered with minimal graffiti. A black metal fence runs parallel to the wall. The scene is set on a bright day. High contrast shadow.",
        "A large, rectangular advertising billboard mounted on two vertical poles. The billboard is visible against a clear sky. The ground below is flat asphalt. The scene is open. Direct, high visibility light.",
        "A public park area with a long, straight gravel path bordered by a low metal curb. The path leads toward a square stone fountain structure. The light is even. Standard daylight.",
        "A small, square brick shed located next to a perfectly straight wooden picket fence. The fence is centered in the foreground. The sky is blue. Clear, uniform light.",
        "A fire hydrant located at the corner of two intersecting straight sidewalks. The hydrant is brightly colored against the gray concrete. The building nearby has strong vertical lines. The scene is sunny. Sharp afternoon light."
    ]

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

