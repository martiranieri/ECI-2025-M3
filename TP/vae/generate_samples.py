from torchmetrics.image.fid import FrechetInceptionDistance
import torchvision.transforms as T
from PIL import Image
import torch
import os

fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)

transform = T.Compose([
    T.Resize(299),        # InceptionV3 espera 299x299
    T.ToTensor(),
    T.Normalize(mean=[0.5]*3, std=[0.5]*3),
])

def load_images_to_tensor(path, n=None):
    imgs = []
    files = [f for f in os.listdir(path) if f.endswith('.png')]
    if n:
        files = files[:n]
    for f in files:
        img = Image.open(os.path.join(path, f)).convert('RGB')
        img = transform(img)
        imgs.append(img)
    return torch.stack(imgs)

# Cargá imágenes reales y generadas
real_images = load_images_to_tensor('../data/mnist_real_test', n=1000)
vae_images = load_images_to_tensor('../punto_vae/generated_vae_run00', n=1000)
gmvae_images = load_images_to_tensor('../punto_gmvae/generated_gmvae_run00', n=1000)

fid.reset()
fid.update(real_images, real=True)
fid.update(vae_images, real=False)
fid_vae = fid.compute().item()

fid.reset()
fid.update(real_images, real=True)
fid.update(gmvae_images, real=False)
fid_gmvae = fid.compute().item()

print(f"FID VAE: {fid_vae}")
print(f"FID GMVAE: {fid_gmvae}")
