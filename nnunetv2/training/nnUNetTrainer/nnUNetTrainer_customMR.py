import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler

class nnUNetTrainer_1000epochs_LRfinetune(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 2000
        self.initial_lr = 0.001
        
        
class nnUNetTrainer_2000epochs_LRdecay(nnUNetTrainer):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super(nnUNetTrainer_2000epochs_LRdecay, self).__init__(
            plans, configuration, fold, dataset_json, device
        )
        self.num_epochs = 2000
        self.num_iterations_per_epoch = 250
        self.num_val_iterations_per_epoch = 100

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.network.parameters(),
            self.initial_lr,
            weight_decay=self.weight_decay,
            momentum=0.99,
            nesterov=True,
        )
        lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, 5000, exponent=10)
        return optimizer, lr_scheduler
    
class nnUNetTrainer_1000epochs_LRdecay(nnUNetTrainer):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super(nnUNetTrainer_1000epochs_LRdecay, self).__init__(
            plans, configuration, fold, dataset_json, device
        )
        self.num_epochs = 1000
        self.num_iterations_per_epoch = 250
        self.num_val_iterations_per_epoch = 100

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.network.parameters(),
            self.initial_lr,
            weight_decay=self.weight_decay,
            momentum=0.99,
            nesterov=True,
        )
        lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, 2500, exponent=10)
        return optimizer, lr_scheduler

class nnUNetTrainer_2000epochs_LRdecay4000e16(nnUNetTrainer):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super(nnUNetTrainer_2000epochs_LRdecay4000e16, self).__init__(
            plans, configuration, fold, dataset_json, device
        )
        self.num_epochs = 2000
        self.num_iterations_per_epoch = 250
        self.num_val_iterations_per_epoch = 100

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.network.parameters(),
            self.initial_lr,
            weight_decay=self.weight_decay,
            momentum=0.99,
            nesterov=True,
        )
        lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, 4000, exponent=16)
        return optimizer, lr_scheduler