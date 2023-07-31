import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from config.data_config import data_config
from config.T5config import Magenta_T5Config
from model.T5 import Transformer
from data.MAESTRO_loader import MIDIDataset
import wandb
from tqdm import tqdm
from data.constants import *
import gc

experiment_config={
        "learning_rate": 2e-5,
        "architecture": "Magenta",
        "dataset": "Slakh2100",
        "training_steps": 1000000,
        "checking_steps": 100000,
        "batch": 4,
        "kernel_size": 0,
        "expansion_factor": 0
        }

def training(model, dataloader, optimizer, criterion, cpt_path):
    wandb.init(
        project="mt3-pytorch",
        config=experiment_config
    )

    with tqdm(dataloader, unit='step') as tstep:
        for batch_idx, batch in enumerate(tstep):
            inputs = batch['inputs'].to(device)
            targets = batch['targets'].to(device)
            outputs = model.forward(encoder_input_tokens=inputs, decoder_target_tokens=targets, decode=False)
            
            loss = criterion(outputs.permute(0,2,1), targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            tstep.set_postfix(loss=loss.item())
            wandb.log({"loss": loss})

            if (batch_idx + 1) % (experiment_config['checking_steps'] / experiment_config['batch']) == 0:
                torch.save(model.state_dict(), cpt_path + experiment_config['architecture'] + '/' + str((batch_idx+1)*experiment_config['batch'])+'.ckpt')

            if batch_idx > experiment_config['training_steps'] / experiment_config['batch']:
                break

    wandb.finish()

if __name__ == "__main__":
    torch.cuda.empty_cache()
    gc.collect()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = Transformer(config=Magenta_T5Config).to(device)
    
    print(model)

    data = MIDIDataset()
    dataloader = DataLoader(data, batch_size=experiment_config["batch"], num_workers=0, pin_memory=True)
    
    criterion = nn.CrossEntropyLoss(ignore_index=TOKEN_PAD)
    optimizer = AdamW(model.parameters(), experiment_config['learning_rate'])

    print(experiment_config)

    training(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer, 
            criterion=criterion, 
            cpt_path=data_config.cpt_path    
            )