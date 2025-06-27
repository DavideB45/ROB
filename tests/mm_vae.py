import os, sys, torch
sys.path.append(os.path.dirname(os.path.dirname("..")))

from moe_vae import MoE_VAE, load_mmvae_model
from dataset.m_dataloader import MDataset
from helpers.utils_proj import show_image, device

LATENT_DIM = 200

def main():
    model = load_mmvae_model(f"models/moe_vae_model_{LATENT_DIM}_.pth", proprioception_input_dim=20, latent_dim=LATENT_DIM)
    model.eval()
    model.to(device)
    dataset = MDataset("dataset", "no_obj")
    # when executed to get numbers the dataloader used old data so it is a tests set (data the model never saw)
    loader, _ = dataset.get_VAE_loaders(["VISION", "POS"], batch_size=100, shuffle=False)

    total_visual_loss = 0.0
    total_position_loss = 0.0
    total_kl_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for elements in loader:
            vision, pos = elements[0], elements[1]
            vision = vision.to(device)
            pos = pos.to(device)
            v_pred, p_pred, mu, lv = model.forward(vision, pos)

            v_pred = torch.clamp(v_pred, 0.0, 1.0)
            visual_loss = torch.nn.functional.cross_entropy(v_pred, vision, reduction='sum') / vision.size(0)
            #visual_loss = torch.nn.functional.mse_loss(v_pred, vision, reduction='sum') / vision.size(0)
            position_loss = model.proprioception_loss(pos, p_pred)
            kl_loss = model.kl_divergence_loss(mu, lv)

            total_visual_loss += visual_loss.item()
            total_position_loss += position_loss.item()
            total_kl_loss += kl_loss.item()
            num_batches += 1

        avg_visual_loss = total_visual_loss / num_batches
        avg_position_loss = total_position_loss / num_batches
        avg_kl_loss = total_kl_loss / num_batches

    # Show a couple of images: original and reconstructed using show_image
    loader_iter = iter(loader)
    vision, pos = next(loader_iter)
    vision = vision.to(device)
    pos = pos.to(device)
    v_pred, _, _, _ = model.forward(vision, pos)
    v_pred = torch.clamp(v_pred, 0.0, 1.0)

    # Show first 2 images: original and reconstructed
    for i in [2,4,8]:
        show_image(vision[i], title="Original", save_path=f"figures/imgs/original_{i}.png")
        show_image(v_pred[i].cpu().detach(), title="Reconstructed", save_path=f"figures/imgs/reconstructed_{i}_{LATENT_DIM}_mmvae.png")

    print(f"Average Visual Loss: {avg_visual_loss}")
    print(f"Average Position Loss: {avg_position_loss}")
    print(f"Average KL Loss: {avg_kl_loss}")

if __name__ == "__main__":
    main()