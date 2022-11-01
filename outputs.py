import numpy as np
import pickle
import torch
from collections import OrderedDict
from model.SUNet import SUNet_model
import yaml


dat_dir = '/home/users/a/akhaury/scratch/SingleChannel_Deconv/'
model_dir = dat_dir+'Trained_Models/SUNet/Denoising/models/'

f = open(dat_dir+'TEST_candels_lsst_sim.pkl', 'rb')
dico = pickle.load(f)
f.close()


# Norm
noise_sigma_orig = dico['noisemap']
x_test = dico['inputs_tikho_laplacian']
y_test = dico['targets']

# Normalize targets
y_test = y_test - np.mean(y_test, axis=(1,2), keepdims=True)
norm_fact = np.max(y_test, axis=(1,2), keepdims=True) 
y_test /= norm_fact

# Normalize & scale tikho inputs
x_test = x_test - np.mean(x_test, axis=(1,2), keepdims=True)
x_test /= norm_fact

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

res_sunet = res_sunet.permute(0, 2, 3, 1).cpu().detach().numpy()

with open(dat_dir+'outputs/sunet_ep-100.pkl', 'wb') as f1:
    pickle.dump(res_sunet, f1, protocol=pickle.HIGHEST_PROTOCOL)