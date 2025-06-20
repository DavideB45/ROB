import os, sys, torch
sys.path.append(os.path.dirname(os.path.dirname("..")))

from moe_vae import MoE_VAE, load_mmvae_model
from dataset.m_dataloader import MDataset
from helpers.utils_proj import show_image

def main():
    model = load_mmvae_model("models/moe_vae_model.pth", proprioception_input_dim=20, latent_dim=200)
    model.eval()
    dataset = MDataset("dataset", "no_obj")
    _, loader = dataset.get_VAE_loaders(["VISION", "POS"], batch_size=100)
    elements = next(iter(loader))
    vision, pos = elements[0], elements[1]

    with torch.no_grad():
        predicted = model.position_to_vision(pos)

    print(f"Predicted shape: {predicted.shape}, Vision shape: {vision.shape}")

    # Show the first image from the predicted and actual vision data
    for i in range(0):
        show_image(predicted[i].cpu().numpy(), title="Predicted Vision")
        show_image(vision[i].cpu().numpy(), title="Actual Vision")

    v_pred, p_pred, mu, lv = model.forward(vision, pos)  # Forward pass to compute losses
    #show visual loss
    visual_loss = model.visual_loss(vision, v_pred)
    print(f"Visual Loss: {visual_loss.item()}")
    vis, _, _ = model.vision.forward(vision)
    visual_loss = model.visual_loss(vision, vis)
    print(f"Visual Loss: {visual_loss.item()}")

    #show position loss
    position_loss = model.proprioception_loss(pos, p_pred)
    print(f"Position Loss: {position_loss.item()}")
    pos, _, _ = model.proprioception.forward(pos)
    position_loss = model.proprioception_loss(pos, p_pred)
    print(f"Position Loss: {position_loss.item()}")

    #show kl loss
    kl_loss = model.kl_divergence_loss(mu, lv)
    print(f"KL Loss: {kl_loss.item()}")

    #show average proprioception latent:
    mu1, lv = model.proprioception.encode(pos[0].unsqueeze(0))
    print(f"Average Proprioception over sample: \n{mu1[0][0:10]}")
    #show average vision latent:
    mu2, lv = model.vision.encode(vision[0].unsqueeze(0))
    print(f"Average Vision over sample: \n{mu2[0][0:10]}")

    # show average latent:
    mu, lv = model.encode(vision[0].unsqueeze(0), pos[0].unsqueeze(0))
    print(f"Average Latent over sample: \n{mu[0][0:10]}")
if __name__ == "__main__":
    main()