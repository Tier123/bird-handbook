from diffusers import StableDiffusionPipeline
import torch

pretrained_model_name_or_path = "/home/xiezisheng/aigc/checkpoints/stable-diffusion-v1-5" #"runwayml/stable-diffusion-v1-5"
repo_id_embeds = "./embeddings/crow_pt"

pipeline = StableDiffusionPipeline.from_pretrained(pretrained_model_name_or_path, torch_dtype=torch.float16).to("cuda")
pipeline.load_textual_inversion(repo_id_embeds)
image = pipeline("A <crow_pt> party", num_inference_steps=50).images[0]
image.save("./images/output/crow_party.png")