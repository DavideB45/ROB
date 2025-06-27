import numpy as np
import matplotlib.pyplot as plt

vae_200 = [np.float32(0.4211031), np.float32(0.4219108), np.float32(0.42809835), np.float32(0.4257251), np.float32(0.42644083), np.float32(0.42784607), np.float32(0.426277), np.float32(0.43024027), np.float32(0.43294412), np.float32(0.42985037), np.float32(0.43537608), np.float32(0.43395287), np.float32(0.42699233), np.float32(0.43661243), np.float32(0.4327297), np.float32(0.42994088), np.float32(0.42076072), np.float32(0.42581022), np.float32(0.42648298), np.float32(0.43449846), np.float32(0.4353074), np.float32(0.42721447), np.float32(0.4386107), np.float32(0.42365804), np.float32(0.41894022), np.float32(0.4299085), np.float32(0.4287121), np.float32(0.41558442)]
vae_30 = [np.float32(0.5860825), np.float32(0.62228125), np.float32(0.6429395), np.float32(0.65139306), np.float32(0.64696264), np.float32(0.64419407), np.float32(0.6401189), np.float32(0.6310025), np.float32(0.6254809), np.float32(0.6316891), np.float32(0.63253856), np.float32(0.6195329), np.float32(0.6229145), np.float32(0.61310697), np.float32(0.6136863), np.float32(0.6242935), np.float32(0.6249187), np.float32(0.6154825), np.float32(0.61272216), np.float32(0.60594773), np.float32(0.62225276), np.float32(0.6221228), np.float32(0.61420214), np.float32(0.60586256), np.float32(0.6079289), np.float32(0.5991249), np.float32(0.61642087), np.float32(0.6102206)]
mmvae_200 = [np.float32(0.6138557), np.float32(0.63431495), np.float32(0.64595973), np.float32(0.6465755), np.float32(0.647229), np.float32(0.6501415), np.float32(0.6538948), np.float32(0.6514886), np.float32(0.65006125), np.float32(0.64549446), np.float32(0.6565754), np.float32(0.64701277), np.float32(0.63608694), np.float32(0.64591813), np.float32(0.6533551), np.float32(0.6560473), np.float32(0.64381117), np.float32(0.6408198), np.float32(0.6402793), np.float32(0.6486872), np.float32(0.6605083), np.float32(0.64429694), np.float32(0.6659268), np.float32(0.6520394), np.float32(0.64418834), np.float32(0.64862436), np.float32(0.6525227), np.float32(0.63173914)]
mmvae_30 = [np.float32(0.5333122), np.float32(0.58668065), np.float32(0.6250323), np.float32(0.6324498), np.float32(0.63450414), np.float32(0.6257582), np.float32(0.62034184), np.float32(0.62017286), np.float32(0.61573166), np.float32(0.61773103), np.float32(0.6129578), np.float32(0.62127614), np.float32(0.60440445), np.float32(0.5931782), np.float32(0.5960192), np.float32(0.5970362), np.float32(0.5895575), np.float32(0.6055506), np.float32(0.5934033), np.float32(0.59817386), np.float32(0.5902288), np.float32(0.60172075), np.float32(0.59166604), np.float32(0.6074008), np.float32(0.5879855), np.float32(0.6004157), np.float32(0.6132004), np.float32(0.5876755)]
mmvae_200_ = [np.float32(0.5112665), np.float32(0.5145088), np.float32(0.5199958), np.float32(0.5186908), np.float32(0.5161489), np.float32(0.5190673), np.float32(0.5161714), np.float32(0.5116857), np.float32(0.5182985), np.float32(0.5127304), np.float32(0.5216932), np.float32(0.50855976), np.float32(0.5061629), np.float32(0.51700705), np.float32(0.5099999), np.float32(0.5141439), np.float32(0.5022975), np.float32(0.49996004), np.float32(0.51306385), np.float32(0.5104999), np.float32(0.51673245), np.float32(0.5049438), np.float32(0.5267198), np.float32(0.50543606), np.float32(0.5131107), np.float32(0.5151663), np.float32(0.5076957), np.float32(0.49846938)]
mmvae_30_ = [np.float32(0.58146137), np.float32(0.5871502), np.float32(0.5713579), np.float32(0.5432074), np.float32(0.52458906), np.float32(0.5139445), np.float32(0.51121265), np.float32(0.51062196), np.float32(0.5156069), np.float32(0.52144474), np.float32(0.53202355), np.float32(0.53877217), np.float32(0.5295701), np.float32(0.54748404), np.float32(0.55377966), np.float32(0.54566985), np.float32(0.54285216), np.float32(0.5498238), np.float32(0.5450644), np.float32(0.55080384), np.float32(0.5622138), np.float32(0.5458916), np.float32(0.5570124), np.float32(0.54733884), np.float32(0.5410365), np.float32(0.5440219), np.float32(0.5512764), np.float32(0.54348636)]

def plot_ssim_comparison():
    """
    Plot the SSIM values for VAE with different latent dimensions.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(3, 31), vae_200, marker='o', label='VAE Latent Dim 200')
    plt.plot(range(3, 31), vae_30, marker='x', label='VAE Latent Dim 30')
    plt.plot(range(3, 31), mmvae_200, marker='s', label='MMVAE Latent Dim 200 (1 Pos)')
    plt.plot(range(3, 31), mmvae_30, marker='^', label='MMVAE Latent Dim 30 (1 Pos)')
    plt.plot(range(3, 31), mmvae_200_, marker='d', label='MMVAE Latent Dim 200 (150 Pos)')
    plt.plot(range(3, 31), mmvae_30_, marker='*', label='MMVAE Latent Dim 30 (150 Pos)')
    
    plt.title('SSIM Comparison for VAE and MMVAE with Different Latent Dimensions')
    plt.xlabel('Sequence Length')
    plt.ylabel('SSIM Value')
    plt.xticks(range(3, 31))
    plt.grid(True)
    plt.legend()
    
    plt.savefig("vae_mmvae_ssim_comparison_.png", dpi=500)
    plt.show()

if __name__ == "__main__":
    plot_ssim_comparison()