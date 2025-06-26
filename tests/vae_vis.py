import os, sys, torch
sys.path.append(os.path.dirname(os.path.dirname("..")))

from vae_model import VAE
from dataset.dataloader import Dataset
from torch.nn.functional import mse_loss, kl_div
from helpers.utils_proj import device, show_image

def main():
    model = VAE(latent_dim=30)
    model.load_state_dict(torch.load("models/vae_model_30_kl1_.pth", map_location=device))
    model.eval().to(device)

    dataset = Dataset("dataset", "no_obj")
    val_loader = torch.utils.data.DataLoader(dataset.get_validation_set()[0], batch_size=100, shuffle=True)
    
    vision = next(iter(val_loader))

    with torch.no_grad():        
        v_pred, mu, lv = model.forward(vision)
        for i in range(4):
            show_image(v_pred[i].cpu().numpy(), title="Predicted Vision")
            show_image(vision[i].cpu().numpy(), title="Actual Vision")

        #show visual loss
        visual_loss = mse_loss(v_pred, vision, reduction='sum') / vision.size(0)
        print(f"Visual Loss: {visual_loss.item()}")
        
        #show kl loss
        kl_loss = -0.5 * torch.mean(1 + lv - mu.pow(2) - lv.exp())
        print(f"KL Loss: {kl_loss.item()}")

if __name__ == "__main__":
    main()