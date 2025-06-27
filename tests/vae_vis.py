import os, sys, torch
sys.path.append(os.path.dirname(os.path.dirname("..")))

from vae_model import VAE
from dataset.m_dataloader import MDataset
from torch.nn.functional import mse_loss, kl_div, binary_cross_entropy
from helpers.utils_proj import device, show_image

LATENT_DIM = 200

def main():
    model = VAE(latent_dim=LATENT_DIM)
    model.load_state_dict(torch.load(f"models/vae_model_{LATENT_DIM}_kl1_.pth", map_location=device))
    model.eval().to(device)

    dataset = MDataset("dataset", "no_obj")
    # when executed to get numbers the dataloader used old data so it is a tests set (data the model never saw)
    loader, _ = dataset.get_VAE_loaders(["VISION", "POS"], batch_size=100, shuffle=False)
    
    total_visual_loss = 0.0
    total_kl_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for elements in loader:
            vision = elements[0].to(device)
            v_pred, mu, lv = model.forward(vision)

            v_pred = torch.clamp(v_pred, 0.0, 1.0)            
            #visual_loss = mse_loss(v_pred, vision, reduction='sum') / vision.size(0)
            visual_loss = binary_cross_entropy(v_pred, vision, reduction='sum') / vision.size(0)
            kl_loss = -0.5 * torch.mean(1 + lv - mu.pow(2) - lv.exp())
            
            total_visual_loss += visual_loss.item()
            total_kl_loss += kl_loss.item()
            num_batches += 1

        avg_visual_loss = total_visual_loss / num_batches
        avg_kl_loss = total_kl_loss / num_batches

    # Show a couple of images: original and reconstructed using show_image
    loader_iter = iter(loader)
    vision = next(loader_iter)[0].to(device)
    v_pred, _, _ = model.forward(vision)
    v_pred = torch.clamp(v_pred, 0.0, 1.0)
    # Show first 2 images: original and reconstructed
    for i in [2,4,8]:
        #show_image(vision[i], title="Original")
        show_image(v_pred[i].cpu().detach(), title="Reconstructed", save_path=f"figures/imgs/reconstructed_{i}_{LATENT_DIM}_vae.png")

    print(f"Average Visual Loss: {avg_visual_loss}")
    print(f"Average KL Loss: {avg_kl_loss}")
if __name__ == "__main__":
    main()