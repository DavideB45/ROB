import numpy as np
import matplotlib.pyplot as plt

vae_200 = [np.float32(0.4211031), np.float32(0.42464978), np.float32(0.43033737), np.float32(0.42695057), np.float32(0.42710468), np.float32(0.42769343), np.float32(0.4262688), np.float32(0.4291225), np.float32(0.43194717), np.float32(0.42845476), np.float32(0.43388653), np.float32(0.4327781), np.float32(0.42572471), np.float32(0.4346181), np.float32(0.430501), np.float32(0.4287467), np.float32(0.41862753), np.float32(0.42367354)]
vae_30 = [np.float32(0.5860825), np.float32(0.6087385), np.float32(0.6192831), np.float32(0.62249756), np.float32(0.61793697), np.float32(0.6146126), np.float32(0.6147128), np.float32(0.6063799), np.float32(0.6022653), np.float32(0.60770106), np.float32(0.607994), np.float32(0.5973177), np.float32(0.6004302), np.float32(0.5897242), np.float32(0.59241354), np.float32(0.6034459), np.float32(0.5998864), np.float32(0.5946182)]
mmvae_200 = [np.float32(0.6138557), np.float32(0.6290499), np.float32(0.63737506), np.float32(0.6359979), np.float32(0.6350231), np.float32(0.63537633), np.float32(0.6392437), np.float32(0.6356227), np.float32(0.63527954), np.float32(0.63024175), np.float32(0.6418934), np.float32(0.63231), np.float32(0.6188449), np.float32(0.6257703), np.float32(0.637124), np.float32(0.6402844), np.float32(0.6272882), np.float32(0.6245136)]
mmvae_30 = [np.float32(0.5333122), np.float32(0.56933457), np.float32(0.59203976), np.float32(0.59295106), np.float32(0.58929914), np.float32(0.5795502), np.float32(0.5738843), np.float32(0.57616717), np.float32(0.57075244), np.float32(0.5745595), np.float32(0.56165075), np.float32(0.5749516), np.float32(0.55598867), np.float32(0.54571134), np.float32(0.556718), np.float32(0.55663586), np.float32(0.54841834), np.float32(0.56710684)]
mmvae_200_ = [np.float32(0.5112665), np.float32(0.51367164), np.float32(0.5167942), np.float32(0.51532996), np.float32(0.5108945), np.float32(0.5155803), np.float32(0.5081246), np.float32(0.5072494), np.float32(0.5129439), np.float32(0.50791967), np.float32(0.51613605), np.float32(0.50256044), np.float32(0.50061643), np.float32(0.51069134), np.float32(0.505236), np.float32(0.5056488), np.float32(0.4980954), np.float32(0.49644756)]
mmvae_30_ = [np.float32(0.58146137), np.float32(0.583751), np.float32(0.5635989), np.float32(0.5324056), np.float32(0.5127249), np.float32(0.50051945), np.float32(0.49776375), np.float32(0.49907842), np.float32(0.5053676), np.float32(0.514452), np.float32(0.5224711), np.float32(0.534763), np.float32(0.5264624), np.float32(0.546393), np.float32(0.55045885), np.float32(0.54416823), np.float32(0.54636014), np.float32(0.5502838)]

def plot_ssim_comparison():
    """
    Plot the SSIM values for VAE with different latent dimensions.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(3, 21), vae_200, marker='o', label='VAE Latent Dim 200')
    plt.plot(range(3, 21), vae_30, marker='x', label='VAE Latent Dim 30')
    plt.plot(range(3, 21), mmvae_200, marker='s', label='MMVAE Latent Dim 200 (1 Pos)')
    plt.plot(range(3, 21), mmvae_30, marker='^', label='MMVAE Latent Dim 30 (1 Pos)')
    plt.plot(range(3, 21), mmvae_200_, marker='d', label='MMVAE Latent Dim 200 (150 Pos)')
    plt.plot(range(3, 21), mmvae_30_, marker='*', label='MMVAE Latent Dim 30 (150 Pos)')

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