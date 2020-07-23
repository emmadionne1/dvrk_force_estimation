import torch
import numpy as np
from pathlib import Path
from dataset import indirectDataset
from network import torqueNetwork
from torch.utils.data import DataLoader

def calculate_force(jacobian, joints):
    force = torch.zeros((joints.shape[0], 6))
    for i in range(joints.shape[0]):
        j = jacobian[i].reshape(6,6)
        jacobian_inv_t = torch.from_numpy(np.linalg.inv(j).transpose())
        force[i,:] = torch.matmul(jacobian_inv_t, joints[i])
    return force

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
JOINTS = 6

phantom = 'b'
test_path = '../data/csv/test/phantoms/' + phantom + '/'
#test_path = '../data/csv/val/'
root = Path('checkpoints' )

batch_size = 100000000
epoch_to_use = 2000

networks = []
for j in range(JOINTS):
    networks.append(torqueNetwork())
    networks[j].to(device)
                          

test_dataset = indirectDataset(test_path)
test_loader = DataLoader(dataset=test_dataset, batch_size = batch_size, shuffle=False)

model_root = root / "models_indirect"
for j in range(JOINTS):
    model_path = model_root / 'model_joint_{}_{}.pt'.format(j, epoch_to_use)
    if model_path.exists():
        state = torch.load(str(model_path))
        epoch = state['epoch'] + 1
        networks[j].load_state_dict(state['model'])
        networks[j].eval()
        print('Restored model, epoch {}, joint {}'.format(epoch-1, j))
    else:
        print('Failed to restore model')
        exit()

test_loss = np.zeros(JOINTS)
    
for i, (posvel, torque, jacobian, cartesian) in enumerate(test_loader):
    posvel = posvel.to(device)
    all_torques = torch.zeros((posvel.size()[0], 6))
    for j in range(JOINTS):
        pred = networks[j](posvel).cpu().detach()
        diff_torque = torque[:,j] - pred.transpose(1,0)
        all_torques[:,j] = diff_torque       

    pred_force = calculate_force(jacobian, all_torques)
    pred_force = np.concatenate((cartesian, pred_force), axis=1)

np.savetxt('phantom_test_indirect_' + phantom + '.csv', pred_force) 
