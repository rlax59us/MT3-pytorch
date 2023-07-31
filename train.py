import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from config.data_config import data_config
from config.T5config import Magenta_T5Config
from model.T5 import Transformer
from data.MAESTRO_loader import MIDIDataset
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from data.constants import *
import gc

device = "cuda" if torch.cuda.is_available() else "cpu"

experiment_config={
        "learning_rate": 1e-4,
        "architecture": "Magenta",
        "training_steps": 1000000,
        "checking_steps": 100000,
        "batch": 16,
        "kernel_size": 0,
        "expansion_factor": 0
        }

class MT3Trainer(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = Transformer(config=Magenta_T5Config)
        self.criterion = nn.CrossEntropyLoss(ignore_index=TOKEN_PAD)
        self.cpt_path = data_config.cpt_path

    def forward(self, encoder_input_tokens, decoder_target_tokens, decode):
        return self.model.forward(encoder_input_tokens, decoder_target_tokens, decoder_input_tokens=None)
    
    def training_step(self, batch, batch_idx):
        inputs = batch['inputs']
        targets = batch['targets']
        outputs = self.forward(encoder_input_tokens=inputs, decoder_target_tokens=targets, decode=False)
        loss = self.criterion(outputs.permute(0,2,1), targets)
        
        if (batch_idx + 1) % (experiment_config['checking_steps'] / experiment_config['batch']) == 0:
            torch.save(model.state_dict(), self.cpt_path + experiment_config['architecture'] + '/' + str((batch_idx+1)*experiment_config['batch'])+'.ckpt')
        
        self.log("train/loss", loss)
        
        return loss
    
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        inputs = batch['inputs']
        targets = batch['targets']
        outputs = self.forward(encoder_input_tokens=inputs, decoder_target_tokens=targets, decode=False)
        loss = self.criterion(outputs.permute(0,2,1), targets)

        self.log("train/loss", loss)

        return loss
    
    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), experiment_config['learning_rate'])
        
        return optimizer
    
    def train_dataloader(self):
        train_data = MIDIDataset(type='train')
        trainloader = DataLoader(train_data, batch_size=experiment_config["batch"], num_workers=4)
        return trainloader
    
    def val_dataloader(self):
        validation_data = MIDIDataset(type='validation')
        validloader = DataLoader(validation_data, batch_size=experiment_config["batch"], num_workers=4)
        return validloader
    
if __name__ == "__main__":
    torch.cuda.empty_cache()
    gc.collect()
    
    model = MT3Trainer()
    
    print(model)
    print(experiment_config)

    wandb_logger = WandbLogger(project="mt3-pytorch")
    trainer = pl.Trainer(gpus=1,
                         logger=wandb_logger,
                         check_val_every_n_epoch=experiment_config["checking_steps"],
                         max_steps=experiment_config['training_steps']
                         )
    trainer.fit(model)