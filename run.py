import numpy as np
import torch
from algs import rds, ti, ais
from tqdm import tqdm
import argparse
from utils.densities import get_distribution
from utils.densities_np import get_distribution as get_distribution_np
from utils.metrics import get_mmd, get_w2

parser = argparse.ArgumentParser()
parser.add_argument('--alg', type=str, choices=['ti', 'ais', 'rdmc', 'rsdmc', 'zodmc', 'sndmc'], required=True)
parser.add_argument('--dist', type=str, choices=['2d_gmm', 'mueller'], required=True) 
parser.add_argument('--rounds', type=int, default=1024)
parser.add_argument('--device', type=str, default='cuda:0')
args = parser.parse_args()

if args.alg in ['ti', 'ais']:
    dist = get_distribution_np(args.dist)
else:
    dist = get_distribution(args.dist, device=args.device)

all_z = []
if 'gmm' in args.dist:
    all_mmd = []; all_w2 = []
for _ in tqdm(range(args.rounds), desc='Rounds'):
    try:
        if args.alg == 'ti':
            x, z = ti(dist, beta=50, eps=1, batch_size=1024, 
                      num_samples=32, ula_step_size=0.01, ula_steps=50)
            x = torch.from_numpy(x).to(device=args.device, dtype=torch.float32)
        elif args.alg == 'ais':
            x, z = ais(dist, lamda0=100, batch_size=1024, steps=60000, step_size=0.01)
            x = torch.from_numpy(x).to(device=args.device, dtype=torch.float32)
        elif args.alg == 'rdmc':
            x, z, count = rds('rdmc', dist, num_steps=50, T=5, batch_size=1024, init='importance', ula_steps=16, num_estimator_samples=64, device=args.device)
        elif args.alg == 'rsdmc':
            x, z, count = rds('rsdmc', dist, num_steps=50, T=5, batch_size=1024, init='importance', ula_steps=10, rsdmc_num_recursive_steps=2, num_estimator_samples=16, device=args.device)
        elif args.alg == 'zodmc':
            x, z, count = rds('zodmc', dist, num_steps=50, T=5, batch_size=1024, init='importance', num_estimator_samples=1024, device=args.device)
        elif args.alg == 'sndmc':
            x, z, count = rds('sndmc', dist, num_steps=50, T=5, batch_size=1024, init='importance', num_estimator_samples=1024, device=args.device)

        all_z.append(z.mean().item())
        if 'gmm' in args.dist:
            x_gt = dist.sample(x.shape[0])
            all_mmd.append(get_mmd(x, x_gt, args.device).item())
            all_w2.append(get_w2(x, x_gt, args.device).item())
    
    except Exception as e:
        print(f"An error occurred: {e}")


print(f"Algorithm: {args.alg}")
all_z_normalized = np.array(all_z) / dist.Z
print(fr"\hat Z/Z:")
print(fr"${all_z_normalized.mean():.4f} \pm {all_z_normalized.std():.4f}$")
if args.dist == '2d_gmm':
    all_mmd = np.array(all_mmd); all_w2 = np.array(all_w2)
    print(fr"mmd:")
    print(fr"${all_mmd.mean():.4f} \pm {all_mmd.std():.4f}$")
    print(fr"W2:")
    print(fr"${all_w2.mean():.4f} \pm {all_w2.std():.4f}$")