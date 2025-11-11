"""
这是image 遮挡 to mvimg to 3dmesh 加上了SAM+mutual Mask的点击功能
"""
import sys
import os
import imageio
import numpy as np
import torch
import rembg
from PIL import Image
from torchvision.transforms import v2
from pytorch_lightning import seed_everything
from omegaconf import OmegaConf
from einops import rearrange, repeat
from tqdm import tqdm
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler
import cv2
from io import BytesIO


current_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_path)
new_path = os.path.join(current_dir, '../')
sys.path.append(new_path)

from src.sam_utils import get_sam_predictor, run_sam 
from src.utils.train_util import instantiate_from_config
from src.utils.camera_util import (
    FOV_to_intrinsics, 
    get_zero123plus_input_cameras,
    get_circular_camera_poses,
)
from src.utils.mesh_util import save_obj, save_glb
from src.utils.infer_util import remove_background,remove_background_with_mask, resize_foreground, resize_foreground_with_mask, images_to_video

import tempfile
from huggingface_hub import hf_hub_download


if torch.cuda.is_available() and torch.cuda.device_count() >= 2:
    device0 = torch.device('cuda:0')
    device1 = torch.device('cuda:1')
else:
    device0 = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device1 = device0

# Define the cache directory for model files
model_cache_dir = './ckpts/'
os.makedirs(model_cache_dir, exist_ok=True)

global_raw_processed_image = None
global_sam_mask = None

def get_render_cameras(batch_size=1, M=120, radius=2.5, elevation=10.0, is_flexicubes=False):
    """
    Get the rendering camera parameters.
    """
    c2ws = get_circular_camera_poses(M=M, radius=radius, elevation=elevation)
    if is_flexicubes:
        cameras = torch.linalg.inv(c2ws)
        cameras = cameras.unsqueeze(0).repeat(batch_size, 1, 1, 1)
    else:
        extrinsics = c2ws.flatten(-2)
        intrinsics = FOV_to_intrinsics(30.0).unsqueeze(0).repeat(M, 1, 1).float().flatten(-2)
        cameras = torch.cat([extrinsics, intrinsics], dim=-1)
        cameras = cameras.unsqueeze(0).repeat(batch_size, 1, 1)
    return cameras


def images_to_video(images, output_path, fps=30):
    # images: (N, C, H, W)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    frames = []
    for i in range(images.shape[0]):
        frame = (images[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8).clip(0, 255)
        assert frame.shape[0] == images.shape[2] and frame.shape[1] == images.shape[3], \
            f"Frame shape mismatch: {frame.shape} vs {images.shape}"
        assert frame.min() >= 0 and frame.max() <= 255, \
            f"Frame value out of range: {frame.min()} ~ {frame.max()}"
        frames.append(frame)
    imageio.mimwrite(output_path, np.stack(frames), fps=fps, codec='h264')


###############################################################################
# Configuration.
###############################################################################

seed_everything(0)

config_path = 'configs/instant-mesh-large.yaml'
config = OmegaConf.load(config_path)
config_name = os.path.basename(config_path).replace('.yaml', '')
model_config = config.model_config
infer_config = config.infer_config

IS_FLEXICUBES = True if config_name.startswith('instant-mesh') else False

device = torch.device('cuda')

# load diffusion model
print('Loading diffusion model ...')
pipeline = DiffusionPipeline.from_pretrained(
    "sudo-ai/zero123plus-v1.2", 
    custom_pipeline="zero123plus",
    torch_dtype=torch.float16,
    cache_dir=model_cache_dir,
)
pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
    pipeline.scheduler.config, timestep_spacing='trailing'
)

unet_ckpt_path = "ckpts/deocc123_step_15000.ckpt"

checkpoint = torch.load(unet_ckpt_path, map_location='cpu')
filtered_checkpoint = {k.replace('unet.unet.', ''): v for k, v in checkpoint['state_dict'].items()}

# pipeline.unet.load_state_dict(filtered_checkpoint, strict=True)
pipeline.unet.load_state_dict(filtered_checkpoint, strict=False)

pipeline = pipeline.to(device0)

# load reconstruction model
print('Loading reconstruction model ...')
model_ckpt_path = hf_hub_download(repo_id="TencentARC/InstantMesh", filename="instant_mesh_large.ckpt", repo_type="model", cache_dir=model_cache_dir)
model = instantiate_from_config(model_config)
state_dict = torch.load(model_ckpt_path, map_location='cpu')['state_dict']
state_dict = {k[14:]: v for k, v in state_dict.items() if k.startswith('lrm_generator.') and 'source_camera' not in k}
model.load_state_dict(state_dict, strict=True)

model = model.to(device1)
if IS_FLEXICUBES:
    model.init_flexicubes_geometry(device1, fovy=30.0)
model = model.eval()

print('Loading Finished!')


def check_input_image(input_image):
    if input_image is None:
        raise gr.Error("No image uploaded!")


def preprocess(processed_image, mask, do_remove_background):
    global global_raw_processed_image, global_sam_mask
    if global_raw_processed_image is None:
        global_raw_processed_image = processed_image
    if global_sam_mask is None:
        global_sam_mask = mask

    rgb_image = global_raw_processed_image
    mask = global_sam_mask
    rembg_session = rembg.new_session() if do_remove_background else None

    if do_remove_background:
        if mask is None:
            rgb_image = remove_background(rgb_image, rembg_session)
            rgb_image = resize_foreground(rgb_image, 0.85)
            rgb_image = np.array(rgb_image)
        else:
            rgb_image = remove_background_with_mask(rgb_image, mask)
            rgb_image, mask = resize_foreground_with_mask(rgb_image, mask, 0.85)
            rgb_image = np.array(rgb_image)
    else:
        rgb_image = np.array(rgb_image)

    rgb_image = Image.fromarray(rgb_image)
    global_raw_processed_image = rgb_image
    global_sam_mask = mask
    return rgb_image

def save_mask_to_file(input_image):
    img_name = "DEOCC"
    log_path = "/home/dsh/code_cache/real23D/InstantMesh/apps/DEOCC_mask"
    os.makedirs(log_path, exist_ok=True)
    
    if isinstance(input_image, dict) and 'image' in input_image:
        input_image = input_image['image']

    if global_sam_mask is None:
        print("[Warning] global_sam_mask is None, saving the original image without mask processing!")
        final_name = img_name
        if os.path.exists(os.path.join(log_path, f'{img_name}_masked_image.png')):
            i = 1
            while os.path.exists(os.path.join(log_path, f'{img_name}{i}_masked_image.png')):
                i += 1
            final_name = f'{img_name}{i}'
        input_image.save(os.path.join(log_path, f'{final_name}_masked_image.png'))
        global_raw_processed_image.save(os.path.join(log_path, f'{final_name}_input_image.png'))
        return

    mask_array = np.array(global_sam_mask)
    if mask_array.ndim < 3:
        print("[Warning] global_sam_mask is not a valid 3-channel image, cannot save mask!")
        return
    mask_array = mask_array[:,:,0]
    input_image_array = np.array(input_image)
    img_rgba = cv2.cvtColor(input_image_array, cv2.COLOR_RGB2RGBA)
    mask_bool = mask_array > 127

    alpha_channel = img_rgba[:, :, 3] > 0
    darken_mask = (~mask_bool) & alpha_channel
    img_rgba[darken_mask, :3] = (img_rgba[darken_mask, :3] * 0.5).astype(np.uint8)
    
    img_rgba = Image.fromarray(img_rgba)

    final_name = img_name
    if os.path.exists(os.path.join(log_path, f'{img_name}_masked_image.png')):
        i = 1
        while os.path.exists(os.path.join(log_path, f'{img_name}{i}_masked_image.png')):
            i += 1
        final_name = f'{img_name}{i}'
        img_rgba.save(os.path.join(log_path, f'{final_name}_masked_image.png'))
    else:
        final_name = img_name
        img_rgba.save(os.path.join(log_path, f'{final_name}_masked_image.png'))

    input_image.save(os.path.join(log_path, f'{final_name}_input_image.png'))

    mask_array = np.array(global_sam_mask)
    mask_array = mask_array[:,:,0]
    mask_array = Image.fromarray(mask_array)
    mask_array.save(os.path.join(log_path, f'{final_name}_mask.png'))
    return

def generate_mvs(input_image, sample_steps, sample_seed):

    seed_everything(sample_seed)
    # sampling
    generator = torch.Generator(device=device0)
    z123_image = pipeline(
        input_image, 
        num_inference_steps=sample_steps, 
        generator=generator,
        # width=512,
        # height=768,
    ).images[0]

    show_image = np.asarray(z123_image, dtype=np.uint8)
    show_image = torch.from_numpy(show_image)     # (960, 640, 3)
    show_image = rearrange(show_image, '(n h) (m w) c -> (n m) h w c', n=3, m=2)
    show_image = rearrange(show_image, '(n m) h w c -> (n h) (m w) c', n=2, m=3)
    show_image = Image.fromarray(show_image.numpy())

    return z123_image, show_image


def make_mesh(mesh_fpath, planes):

    mesh_basename = os.path.basename(mesh_fpath).split('.')[0]
    mesh_dirname = os.path.dirname(mesh_fpath)
    mesh_glb_fpath = os.path.join(mesh_dirname, f"{mesh_basename}.glb")
        
    with torch.no_grad():
        # get mesh

        mesh_out = model.extract_mesh(
            planes,
            use_texture_map=False,
            **infer_config,
        )

        vertices, faces, vertex_colors = mesh_out
        vertices = vertices[:, [1, 2, 0]]
        
        save_glb(vertices, faces, vertex_colors, mesh_glb_fpath)
        save_obj(vertices, faces, vertex_colors, mesh_fpath)
        
        print(f"[Info] Mesh saved to {mesh_fpath} and {mesh_glb_fpath}")

    return mesh_fpath, mesh_glb_fpath


def make3d(images):
    images = np.asarray(images, dtype=np.float32) / 255.0
    images = torch.from_numpy(images).permute(2, 0, 1).contiguous().float()     # (3, 960, 640)
    images = rearrange(images, 'c (n h) (m w) -> (n m) c h w', n=3, m=2)        # (6, 3, 320, 320)

    input_cameras = get_zero123plus_input_cameras(batch_size=1, radius=4.0).to(device1)
    render_cameras = get_render_cameras(
        batch_size=1, radius=4, elevation=20.0, is_flexicubes=IS_FLEXICUBES).to(device1)

    images = images.unsqueeze(0).to(device1)
    images = v2.functional.resize(images, (320, 320), interpolation=3, antialias=True).clamp(0, 1)

    mesh_fpath = tempfile.NamedTemporaryFile(suffix=f".obj", delete=False).name
    mesh_basename = os.path.basename(mesh_fpath).split('.')[0]
    mesh_dirname = os.path.dirname(mesh_fpath)
    video_fpath = os.path.join(mesh_dirname, f"{mesh_basename}.mp4")

    with torch.no_grad():
        # get triplane
        planes = model.forward_planes(images, input_cameras)

        # get video
        chunk_size = 20 if IS_FLEXICUBES else 1
        render_size = 384
        
        frames = []
        for i in tqdm(range(0, render_cameras.shape[1], chunk_size)):
            if IS_FLEXICUBES:
                frame = model.forward_geometry(
                    planes,
                    render_cameras[:, i:i+chunk_size],
                    render_size=render_size,
                )['img']
            else:
                frame = model.synthesizer(
                    planes,
                    cameras=render_cameras[:, i:i+chunk_size],
                    render_size=render_size,
                )['images_rgb']
            frames.append(frame)
        frames = torch.cat(frames, dim=1)

        images_to_video(
            frames[0],
            video_fpath,
            fps=30,
        )

        print(f"[Info] Video saved to {video_fpath}")

    mesh_fpath, mesh_glb_fpath = make_mesh(mesh_fpath, planes)

    return video_fpath, mesh_fpath, mesh_glb_fpath


import gradio as gr

_HEADER_ = '''
<h2><b>Official DeOcc123  Gradio Demo</b></h2><h2><a href='https://github.com/TencentARC/InstantMesh' target='_blank'><b>DeOcc123 is a deocclusion framework</b></a></h2>

'''

def select_point(fg_bg_radio, selected_points, evt: gr.SelectData):
    global global_raw_processed_image, global_sam_mask

    # add prompts
    if fg_bg_radio == 'positive_prompt':
        selected_points.append(((int(evt.index[0]), int(evt.index[1])), 1))
    else:
        selected_points.append(((int(evt.index[0]), int(evt.index[1])), 0))

    img = np.array(global_raw_processed_image)[:,:,:3]
    h_original, w_original, _ = img.shape
    h_new, w_new = 256, 256
    scale_x = w_new / w_original
    scale_y = h_new / h_original
    processed_sel_pix = []
    for point, label in selected_points:
        x, y = point
        new_x = int(x * scale_x)
        new_y = int(y * scale_y)
        processed_sel_pix.append(([new_x, new_y], label))

    # set image
    img_resized = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
    predictor.value.set_image(img_resized)

    # call SAM
    visible_mask, overlay_mask = run_sam(predictor.value, processed_sel_pix)
    mask = np.squeeze(overlay_mask[0][0])  # (256, 256)
    resized_mask = cv2.resize(mask.astype(np.uint8) * 255, (w_original, h_original), interpolation=cv2.INTER_AREA)
    resized_mask = (resized_mask > 127).astype(np.uint8) * 255

    COLORS = [(255, 0, 0), (0, 255, 0)]
    MARKERS = [1, 4]
    img_with_points = img.copy()

    scaling_factor = min(h_original / 256, w_original / 256)
    marker_size = int(6 * scaling_factor)
    marker_thickness = int(2 * scaling_factor)
    circle_radius = int(10 * scaling_factor)
    for point, label in selected_points:
        cv2.drawMarker(img_with_points, point, COLORS[label], markerType=MARKERS[label], markerSize=marker_size, thickness=marker_thickness)

    img_rgba = cv2.cvtColor(img_with_points, cv2.COLOR_RGB2RGBA)
    mask_bool = resized_mask > 127

    alpha_mask = 0.5
    img_rgba[mask_bool, 0] = (1 - alpha_mask) * img_rgba[mask_bool, 0] + alpha_mask * 255
    img_rgba[mask_bool, 1] = (1 - alpha_mask) * img_rgba[mask_bool, 1]
    img_rgba[mask_bool, 2] = (1 - alpha_mask) * img_rgba[mask_bool, 2]
    img_rgba[mask_bool, 3] = 200
    # dim the non-mask area
    img_rgba[~mask_bool, :3] = (img_rgba[~mask_bool, :3] * 0.5).astype(np.uint8)
    
    # return the image with points and mask, the selected points, the mask
    img_with_points = Image.fromarray(img_rgba)

    resized_mask_3ch = np.stack([resized_mask] * 3, axis=-1)
    global_sam_mask = Image.fromarray(resized_mask_3ch)
    return img_with_points, selected_points, Image.fromarray(resized_mask)

with gr.Blocks() as demo:
    
    predictor = gr.State(value=get_sam_predictor(device=device0, model_type='vit_h'))
    selected_points = gr.State(value=[])
    mask_state = gr.State()

    
    gr.Markdown(_HEADER_)
    with gr.Row(variant="panel"):
        with gr.Column():
            with gr.Row():

                input_image = gr.ImageMask(
                    label="Input Image",
                    image_mode="RGBA",
                    width=256,
                    height=256,
                    type="pil",
                    elem_id="content_image",
                    # image_mode='L'
                    # interactive=True
                )
                
                processed_image = gr.Image(
                    label="Processed Image", 
                    image_mode="RGBA", 
                    width=256,
                    height=256,
                    type="pil", 
                    interactive=True,
                    # tool="select"
                )
                
            with gr.Row():
                with gr.Group():
                    do_remove_background = gr.Checkbox(
                        label="Remove Background", value=True
                    )
                    sample_seed = gr.Number(value=42, label="Seed Value", precision=0)

                    sample_steps = gr.Slider(
                        label="Sample Steps",
                        minimum=30,
                        maximum=75,
                        value=75,
                        step=5
                    )
            with gr.Row():
                with gr.Group():
                    fg_bg_radio = gr.Radio(
                        ['positive_prompt', 'negative_prompt'],
                        label='Point Prompt Type')

                    fg_bg_radio.value = 'positive_prompt'
            
                
                
            with gr.Row():
                Manual_mask_button = gr.Button("Manual Mask", elem_id="manual_mask", variant="secondary")
                preprocess_button = gr.Button("Preprocess", elem_id="preprocess", variant="secondary")


            with gr.Row():
                submit = gr.Button("Generate", elem_id="generate", variant="primary")
            
            with gr.Row():
                save_mask = gr.Button("Save Mask", elem_id="save_mask", variant="secondary")
                
            with gr.Row(variant="panel"):
                gr.Examples(
                    examples=[
                        os.path.join("examples", img_name) for img_name in sorted(os.listdir("examples"))
                    ],
                    inputs=[input_image],
                    label="Examples",
                    examples_per_page=20
                )

        with gr.Column():

            with gr.Row():

                with gr.Column():
                    mv_show_images = gr.Image(
                        label="Generated Multi-views",
                        type="pil",
                        width=379,
                        interactive=False
                    )

                with gr.Column():
                    output_video = gr.Video(
                        label="video", format="mp4",
                        width=379,
                        autoplay=True,
                        interactive=False
                    )

            with gr.Row():
                with gr.Tab("OBJ"):
                    output_model_obj = gr.Model3D(
                        label="Output Model (OBJ Format)",
                        #width=768,
                        interactive=False,
                    )
                    gr.Markdown("Note: Downloaded .obj model will be flipped. Export .glb instead or manually flip it before usage.")
                with gr.Tab("GLB"):
                    output_model_glb = gr.Model3D(
                        label="Output Model (GLB Format)",
                        #width=768,
                        interactive=False,
                    )
                    gr.Markdown("Note: The model shown here has a darker appearance. Download to get correct results.")

            with gr.Row():
                gr.Markdown('''Try a different <b>seed value</b> if the result is unsatisfying (Default: 42).''')


    mv_images = gr.State()

    def set_preprocessed_image_after_manual(input_image):
        global global_raw_processed_image, global_sam_mask
        
        # 首先判断是否有手动遮挡
        rgb_image = input_image['image']
        mask = input_image['mask']
        if mask is not None:
            rgb_image = np.array(rgb_image)
            # 只需要把mask部分设置透明度为0即可
            add_mask = np.array(mask)[:,:,0]    
            rgb_image[add_mask == 255] = [0, 0, 0, 0]
            rgb_image = Image.fromarray(rgb_image)
            
        global_raw_processed_image = rgb_image
        global_sam_mask = None
        selected_points = []
        mask_state = None
        return rgb_image, selected_points, mask_state

    
    Manual_mask_button.click(
        fn=set_preprocessed_image_after_manual,
        inputs=[input_image],
        outputs=[processed_image, selected_points, mask_state]
    )

    processed_image.select(
        select_point,
        [fg_bg_radio, selected_points],
        [processed_image, selected_points, mask_state]
    )
    
    preprocess_button.click(
        fn=preprocess,
        inputs=[processed_image, mask_state, do_remove_background],
        outputs=[processed_image],
    )
    submit.click(fn=check_input_image, inputs=[input_image]).success(
        fn=generate_mvs,
        inputs=[processed_image, sample_steps, sample_seed],
        outputs=[mv_images, mv_show_images],
    ).success(
        fn=make3d,
        inputs=[mv_images],
        outputs=[output_video, output_model_obj, output_model_glb]
    )

    save_mask.click(
        fn=save_mask_to_file,
        inputs=[input_image],
        outputs=[]
    )
    
demo.queue(max_size=10)
demo.launch(server_name="0.0.0.0", server_port=8012, share=True)

