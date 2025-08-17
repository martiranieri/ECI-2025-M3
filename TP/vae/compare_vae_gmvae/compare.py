import os
from PIL import Image
import torch
from torchvision import transforms
import sys
import os

# para poder importar codebase
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from codebase.models.nns.v1 import Classifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# clasificador preentrenado
classifier = Classifier(y_dim=10)
checkpoint_path = 'classifier_mnist.pt'
checkpoint = torch.load(checkpoint_path, map_location=device)
classifier.load_state_dict(checkpoint['model_state_dict'])
classifier.to(device)
classifier.eval()

# transformaciones para las imágenes
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
])

def clasificar_imgs(model, path):
    model.eval()
    counts = torch.zeros(10)
    with torch.no_grad():
        for filename in os.listdir(path):
            if filename.endswith(".png"):
                img_path = os.path.join(path, filename)
                img = Image.open(img_path).convert('L')
                x = transform(img).unsqueeze(0).to(device)
                x = x.view(x.size(0), -1)
                pred = model(x).argmax(dim=1)
                counts[pred.item()] += 1
    return counts

vae_dir = '../punto_vae/generated_images_run0000'
gmvae_dir = '../punto_gmvae/generated_images_run0000'

vae_digs = clasificar_imgs(classifier, vae_dir)
gmvae_digs = clasificar_imgs(classifier, gmvae_dir)

print("VAE clasificación (conteo por dígito):", vae_digs)
print("GMVAE clasificación (conteo por dígito):", gmvae_digs)
