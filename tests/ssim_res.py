import numpy as np
import matplotlib.pyplot as plt

vae_200 = [np.float32(0.4211031), np.float32(0.42464978), np.float32(0.43033737), np.float32(0.42695057), np.float32(0.42710468), np.float32(0.42769343), np.float32(0.4262688), np.float32(0.4291225), np.float32(0.43194717), np.float32(0.42845476), np.float32(0.43388653), np.float32(0.4327781), np.float32(0.42572471), np.float32(0.4346181), np.float32(0.430501), np.float32(0.4287467), np.float32(0.41862753), np.float32(0.42367354)]
vae_30 = [np.float32(0.5860825), np.float32(0.6087385), np.float32(0.6192831), np.float32(0.62249756), np.float32(0.61793697), np.float32(0.6146126), np.float32(0.6147128), np.float32(0.6063799), np.float32(0.6022653), np.float32(0.60770106), np.float32(0.607994), np.float32(0.5973177), np.float32(0.6004302), np.float32(0.5897242), np.float32(0.59241354), np.float32(0.6034459), np.float32(0.5998864), np.float32(0.5946182)]


def plot_ssim_comparison():
    """
    Plot the SSIM values for VAE with different latent dimensions.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(3, 21), vae_200, marker='o', label='VAE Latent Dim 200')
    plt.plot(range(3, 21), vae_30, marker='x', label='VAE Latent Dim 30')
    
    plt.title('SSIM Comparison for VAE with Different Latent Dimensions')
    plt.xlabel('Sequence Length')
    plt.ylabel('SSIM Value')
    plt.xticks(range(3, 21))
    plt.grid(True)
    plt.legend()
    
    plt.savefig("vae_ssim_comparison.png")
    plt.show()

if __name__ == "__main__":
    plot_ssim_comparison()