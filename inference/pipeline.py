import os
from typing import List, Optional
import cv2
import numpy as np
import torch
from PIL import Image
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler
from channel_processor import ChannelProcessor


class MultiControlNetPipeline3Chan:
    """Single-ControlNet pipeline with channel-time modulation in condition space."""

    def __init__(
        self,
        sd_path: str,
        cn_path: str,
        r_range: tuple,
        g_range: tuple,
        b_range: tuple,
        debug_shape: bool = True,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.r_range = r_range
        self.g_range = g_range
        self.b_range = b_range
        self.debug_shape = debug_shape
        self.channel_processor = ChannelProcessor()

        print(f"[INFO] Loading ControlNet from: {cn_path}")
        if os.path.isfile(cn_path):
            self.controlnet = ControlNetModel.from_single_file(cn_path, torch_dtype=torch.float16).to(self.device)
        else:
            self.controlnet = ControlNetModel.from_pretrained(cn_path, torch_dtype=torch.float16).to(self.device)

        print(f"[INFO] Loading SD1.5 from: {sd_path}")
        if os.path.isfile(sd_path):
            self.pipe = StableDiffusionControlNetPipeline.from_single_file(
                sd_path,
                controlnet=self.controlnet,
                safety_checker=None,
                torch_dtype=torch.float16,
            )
        else:
            self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
                sd_path,
                controlnet=self.controlnet,
                safety_checker=None,
                torch_dtype=torch.float16,
            )

        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.to(self.device)

        try:
            self.pipe.enable_xformers_memory_efficient_attention()
            print("[INFO] xFormers memory efficient attention activated.")
        except Exception as exc:
            print("[WARNING] xFormers not available:", exc)

    def _prep_control_img(self, img_array: np.ndarray) -> torch.Tensor:
        if img_array.ndim == 2:
            img_array = img_array[..., None]
        if img_array.shape[2] == 1:
            img_array = np.concatenate([img_array] * 3, axis=2)

        h, w = img_array.shape[:2]
        new_h = ((h - 1) // 8 + 1) * 8
        new_w = ((w - 1) // 8 + 1) * 8

        if (new_h != h) or (new_w != w):
            if self.debug_shape:
                print(f"[DEBUG] resizing from {w}x{h} => {new_w}x{new_h}")
            img_array = cv2.resize(img_array, (new_w, new_h), interpolation=cv2.INTER_AREA)

        if img_array.max() > 1.0:
            img_array = img_array / 255.0

        return torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0).to(self.device, torch.float16)

    def _get_active_scale(self, step_i: int, total_steps: int, ranges: List[List[float]]) -> float:
        ratio = step_i / float(total_steps - 1)
        for start, end in zip(ranges[0], ranges[1]):
            if start <= ratio < end:
                return 1.0
        return 0.0

    @torch.no_grad()
    def __call__(
        self,
        poly_edge_image: np.ndarray,
        prompt: str,
        num_inference_steps: int,
        guidance_scale: float,
        seed: Optional[int] = None,
        step_callback=None,
    ) -> Image.Image:
        combined_channels, time_ranges = self.channel_processor.process_image_channels(
            poly_edge_image,
            (self.r_range[0], self.r_range[1]),
            (self.g_range[0], self.g_range[1]),
            (self.b_range[0], self.b_range[1]),
        )

        control_images = [self._prep_control_img(img) for img in combined_channels]

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        do_cfg = guidance_scale > 1.0
        pipe = self.pipe
        pipe.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = pipe.scheduler.timesteps

        text_embeds = pipe._encode_prompt(
            prompt=prompt,
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=do_cfg,
            negative_prompt="",
        )

        generator = torch.Generator(device=self.device).manual_seed(seed) if seed is not None else None
        latents = pipe.prepare_latents(
            batch_size=1,
            num_channels_latents=pipe.unet.config.in_channels,
            height=poly_edge_image.shape[0],
            width=poly_edge_image.shape[1],
            dtype=text_embeds.dtype,
            device=self.device,
            generator=generator,
        )

        for i, t in enumerate(timesteps):
            scale = self._get_active_scale(i, num_inference_steps, time_ranges)
            lat_in = torch.cat([latents] * 2) if do_cfg else latents
            lat_in = pipe.scheduler.scale_model_input(lat_in, t)

            active_idx = None
            step_ratio = i / float(num_inference_steps - 1)
            for idx, (start, end) in enumerate(zip(time_ranges[0], time_ranges[1])):
                if start <= step_ratio < end:
                    active_idx = idx
                    break

            if active_idx is not None and scale > 0:
                control_img = control_images[active_idx]
                down_block_res_samples, mid_block_res_sample = self.controlnet(
                    lat_in,
                    t,
                    encoder_hidden_states=text_embeds,
                    controlnet_cond=control_img,
                    return_dict=False,
                )

                noise_pred = pipe.unet(
                    lat_in,
                    t,
                    encoder_hidden_states=text_embeds,
                    down_block_additional_residuals=[d * scale for d in down_block_res_samples],
                    mid_block_additional_residual=mid_block_res_sample * scale,
                ).sample
            else:
                noise_pred = pipe.unet(
                    lat_in,
                    t,
                    encoder_hidden_states=text_embeds,
                ).sample

            if do_cfg:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample

            if step_callback is not None:
                step_callback(i, t, latents, float(scale), active_idx)

        image = pipe.vae.decode(latents / pipe.vae.config.scaling_factor, return_dict=False)[0]
        return pipe.image_processor.postprocess(image, output_type="pil")[0]
