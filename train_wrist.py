import sys
import tqdm
import torch
from pathlib import Path
from dataset import indirectDataset
from network import torqueNetwork
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils import init_weights

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_joints = 3
joints = [3,4,5]
window = 10
skip = 10

data = sys.argv[1]
train_path = '../data/csv/train/' + data + '/'
val_path = '../data/csv/val/' + data + '/'
root = Path('checkpoints' ) 
folder = data + "_wrist_window" + str(window) + '_' + str(skip)

lr = 1e-2
batch_size = 4096
epochs = 1000
validate_each = 5
use_previous_model = False
epoch_to_use = 995

networks = []
optimizers = []
schedulers = []
for j in range(num_joints):
    networks.append(torqueNetwork(window, num_joints))
    networks[j].to(device)
    optimizers.append(torch.optim.SGD(networks[j].parameters(), lr))
    schedulers.append(ReduceLROnPlateau(optimizers[j], verbose=True))
                          

train_dataset = indirectDataset(train_path, window, skip, joints)
val_dataset = indirectDataset(val_path, window, skip, joints)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size = batch_size, shuffle=False)
    
loss_fn = torch.nn.MSELoss()

try:
    model_root = root / "models_indirect" / folder
    model_root.mkdir(mode=0o777, parents=False)
except OSError:
    print("Model path exists")

if use_previous_model:
    for j in range(num_joints):
        model_path = model_root / 'model_joint_{}_{}.pt'.format(j, epoch_to_use)
        if model_path.exists():
            state = torch.load(str(model_path))
            epoch = state['epoch'] + 1
            networks[j].load_state_dict(state['model'])
            optimizers[j].load_state_dict(state['optimizer'])
            schedulers[j].load_state_dict(state['scheduler'])
            print('Restored model, epoch {}, joint {}'.format(epoch-1, j))
        else:
            print('Failed to restore model')
            exit()
else:
    for j in range(num_joints):
        init_weights(networks[j])
    epoch = 1

save = lambda ep, model, model_path, error, optimizer, scheduler: torch.save({
    'model': model.state_dict(),
    'epoch': ep,
    'error': error,
    'optimizer': optimizer.state_dict(),
    'scheduler': scheduler.state_dict(),
}, str(model_path))

print('Training for ' + str(epochs))
for e in range(epoch, epochs + 1):

    tq = tqdm.tqdm(total=(len(train_loader) * batch_size))
    tq.set_description('Epoch {}, lr {}'.format(e, optimizers[0].param_groups[0]['lr']))
    epoch_loss = torch.zeros(num_joints)

    for j in range(num_joints):
        networks[j].train()
    
    for i, (posvel, torque, jacobian) in enumerate(train_loader):
        posvel = posvel.to(device)
        torque = torque.to(device)

        step_loss = torch.zeros(num_joints)

        for j in range(num_joints):
            pred = networks[j](posvel)
            loss = loss_fn(pred.squeeze(), torque[:,j])
            step_loss[j] += loss.item()
            optimizers[j].zero_grad()
            loss.backward()
            optimizers[j].step()

        tq.update(batch_size)
        tq.set_postfix(loss=' loss={:.5f}'.format(torch.mean(step_loss)))
        epoch_loss += step_loss

    tq.set_postfix(loss=' loss={:.5f}'.format(torch.mean(epoch_loss)/len(train_loader)))
    
    if e % validate_each == 0:
        for j in range(num_joints):
            networks[j].eval()

        val_loss = torch.zeros(num_joints)
        for i, (posvel, torque, jacobian) in enumerate(val_loader):
            posvel = posvel.to(device)
            torque = torque.to(device)

            for j in range(num_joints):
                pred = networks[j](posvel)
                loss = loss_fn(pred.squeeze(), torque[:,j])
                val_loss[j] += loss.item()

        for j in range(num_joints):
            schedulers[j].step(val_loss[j])
        tq.set_postfix(loss='validation loss={:5f}'.format(torch.mean(val_loss)/len(val_loader)))

        for j in range(num_joints):
            model_path = model_root / "model_joint_{}_{}.pt".format(j, e)
            save(e, networks[j], model_path, val_loss[j]/len(val_loader), optimizers[j], schedulers[j])
        
    tq.close()