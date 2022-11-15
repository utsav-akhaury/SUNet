import numpy as np
import pickle
import torch
from collections import OrderedDict
from model.SUNet import SUNet_model
import yaml
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns

sns.set_theme()
sns.set_context("paper", font_scale=2, rc={"lines.linewidth": 2.5})
sns.set_style("whitegrid", {'axes.grid' : False})


dat_dir = '/home/users/a/akhaury/scratch/SingleChannel_Deconv/'
model_dir = dat_dir+'Trained_Models/SUNet/Denoising/models/'

f = open(dat_dir+'TEST_candels_lsst_sim.pkl', 'rb')
dico = pickle.load(f)
f.close()


# Norm
noise_sigma_orig = dico['noisemap']
x_test = dico['inputs_tikho_laplacian']
y_test = dico['targets']
noisy = dico['noisy']

# Normalize targets
y_test = y_test - np.mean(y_test, axis=(1,2), keepdims=True)
norm_fact = np.max(y_test, axis=(1,2), keepdims=True) 
y_test /= norm_fact

# Normalize & scale tikho inputs
x_test = x_test - np.mean(x_test, axis=(1,2), keepdims=True)
x_test /= norm_fact

# Normalize & scale noisy images
noisy = noisy - np.mean(noisy, axis=(1,2), keepdims=True)
noisy /= norm_fact

# NCHW convention
x_test = np.expand_dims(x_test, 1)
y_test = np.expand_dims(y_test, 1)

# Convert to torch tensor
x_test = torch.tensor(x_test)
y_test = torch.tensor(y_test)


# Load model
with open('training.yaml', 'r') as config:
    opt = yaml.safe_load(config)

def load_checkpoint(model, weights):
    checkpoint = torch.load(weights)
    try:
        model.load_state_dict(checkpoint["state_dict"])
    except:
        state_dict = checkpoint["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)

model = SUNet_model(opt)
model.cuda()

load_checkpoint(model, model_dir+'model_latest_ep-100_bs-32_ps-4_ws-8_embdim-48.pth')
model.eval()


res_sunet = torch.zeros(x_test.size())
print(res_sunet.size())

for i in range(0, x_test.size()[0], 500):

    if i+500 > x_test.size()[0]:
        ind = x_test.size()[0]
    else:
        ind = i+500

    input_ = x_test[i:ind].cuda()
    print(input_.size())

    with torch.no_grad():
        res_sunet[i:ind] = model(input_)

# Convert arrays
res_sunet = np.squeeze(res_sunet.permute(0, 2, 3, 1).cpu().detach().numpy())
x_test = np.squeeze(x_test.permute(0, 2, 3, 1).cpu().detach().numpy())
y_test = np.squeeze(y_test.permute(0, 2, 3, 1).cpu().detach().numpy())

with open(dat_dir+'outputs/sunet_ep-100.pkl', 'wb') as f1:
    pickle.dump(res_sunet, f1, protocol=pickle.HIGHEST_PROTOCOL)


#### Plots

def plot_comparison(noisy, sunet, targets, psf, labels, figsize):
    
    list_im, list_min, list_max = [], [], []

    n_row = targets.shape[0]
    n_col = len(labels)
    
    error1 = np.zeros((noisy.shape))

    for i in range(error1.shape[0]):
        error1[i] = sunet[i] - targets[i]

    resid_sunet = np.zeros((noisy.shape))

    from scipy.signal import convolve
    for i in range(sunet.shape[0]):
        resid_sunet[i] = noisy[i] - convolve(psf, sunet[i], mode='same')

    for i in range(n_row):       
        list_im += [noisy[i], sunet[i], targets[i], error1[i], resid_sunet[i]]
        list_min += [np.min(targets[i])]
        list_max += [np.max(targets[i])]
    
    fig, axes = plt.subplots(n_row, n_col, figsize=figsize)
        
    for i in range(n_row):
        
        for k in range(n_col):  
            if k==n_col-1 or k==n_col-2:
                im = axes[k].imshow(list_im[k]) 
            else:
                im = axes[k].imshow(list_im[k], vmin=list_min[k//n_col], vmax=list_max[k//n_col]) 
            
            div = make_axes_locatable(axes[k])
            cax = div.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(im, ax=axes[k], cax=cax)

        for j in range(n_col):
            axes[j].set_axis_off()
            if i==0:
                axes[j].set_title(labels[j])
        
    plt.tight_layout()
    
    return fig


def nmse(signal_1, signal_2):
    return np.around(np.linalg.norm(signal_2 - signal_1, axis=(0,1))**2 / np.linalg.norm(signal_1, axis=(0,1))**2, 2)


selection = [899,932,938,962] #,1004,1094,1210,1283,1302,1308,1356,1382,1392,1574,1623,1688,1742,1838,1899,1954,2048,
            #  2132,2157,29,34,47,52,81,163,164,185,192,239,241,300,319,324,365,367,373,378,381,403,455,647]

for i in range(len(selection)):
    fig = plot_comparison(np.expand_dims(noisy[selection][i], 0),
                          np.expand_dims(res_sunet[selection][i], 0), 
                          np.expand_dims(y_test[selection][i], 0),
                          dico['psf'][0],
                          labels = ['Convolved Noisy Image',                                
                                    'SUNet  |  NMSE = {}'.format(nmse(y_test[selection][i], res_sunet[selection][i])), 
                                    'Target Image - {}'.format(selection[i]),
                                    'Error (Output - Target)',
                                    'Residual (Noisy - PSF * Ouput)'], figsize=(40,7))

    plt.savefig(dat_dir+'outputs/Plots/SUNet/{}.jpeg'.format(selection[i]), bbox_inches='tight')      