import numpy as np
import torch
import torch.nn as nn

from src.model_utils import setup_model, warm_start, save_model
from src.config_utils import setup_config_variables
from src.animate_loader import animate_loader
from src.data_utils import prepare_dataloader
from src.distributed_utils import ddp_setup, ddp_cleanup, ddp_model
from src.logging_utils import log_loss, report_progress, setup_tensorboard, close_tensorboard
from src.training_utils import run_epoch

def main():
    config = setup_config_variables()
    ddp_setup(config["LOCAL_RANK"])
    model = setup_model(config["MODEL_TYPE"], config["MODEL_DIMS"], config["LOCAL_RANK"])
    model = ddp_model(model, config["LOCAL_RANK"])

    writer = setup_tensorboard(config["MODEL_SAVE_DIR"], config["GLOBAL_RANK"])

    train_loader, test_loader = prepare_dataloader(config["TRAINING_DIR"], config["TEST_DIR"], config["BATCH_SIZE"], config["AUGMENT_DATA"])
    report_progress(config["GLOBAL_RANK"], f"Train loader has {len(train_loader)} batches of length {config['WINDOW_SIZE']}, equalling {len(train_loader) * config['BATCH_SIZE'] * config['WINDOW_SIZE']} frames.")
    report_progress(config["GLOBAL_RANK"], f"Test loader has {len(test_loader)} batches of length {config['WINDOW_SIZE']}, equalling {len(test_loader) * config['BATCH_SIZE'] * config['WINDOW_SIZE']} frames.")
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["LEARNING_RATE"], weight_decay=config["WEIGHT_DECAY"])
    def lr_lambda(epoch):
        warmup_epochs = config["EPOCHS"] // 10
        T_max = config["EPOCHS"] - warmup_epochs
        # linear warmup
        if epoch < warmup_epochs:
            return float(epoch) / float(max(1, warmup_epochs))
        # cosine annealing
        else:
            progress = float(epoch - warmup_epochs) / float(max(1, T_max))
            return 0.5 * (1.0 + np.cos(np.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
    starting_epoch = 0
    best_test_loss = np.inf
    last_model_dir = f"{config['MODEL_SAVE_DIR']}/last"
    best_model_dir = f"{config['MODEL_SAVE_DIR']}/best"
    if config["WARM_START"]:
        starting_epoch, best_test_loss = warm_start(model, optimizer, config["LOCAL_RANK"], last_model_dir)
    # only animate at the beginning of training,no need to re-do it on warm start
    elif config["ANIMATE_LOADER"]:
        animate_loader(train_loader, save_path=config["IMAGES_DIR"], global_rank=config["GLOBAL_RANK"])

    scaler = torch.amp.GradScaler("cuda")
    report_progress(config["GLOBAL_RANK"], f"Training on {100 * config['PORTION_OF_DATASET']}% of the dataset, as set by PORTION_OF_DATASET in config.")
    for epoch in range(starting_epoch, config["EPOCHS"]):
        train_loader.sampler.set_epoch(epoch)
        test_loader.sampler.set_epoch(epoch)
        
        if config["LOCAL_RANK"] == 0:
            report_progress(config["GLOBAL_RANK"], "Beginning Epoch", epoch=epoch, total_epochs=config["EPOCHS"])
        
        epoch_train_loss = run_epoch(model, train_loader, optimizer, scaler, criterion,
                                    "TRAIN", config["PORTION_OF_DATASET"], config["LOCAL_RANK"],
                                    config["GLOBAL_RANK"], config["MODEL_TYPE"], writer=writer, epoch=epoch)
        epoch_test_loss = run_epoch(model, test_loader, optimizer, scaler, criterion,
                                    "TEST", config["PORTION_OF_DATASET"], config["LOCAL_RANK"],
                                    config["GLOBAL_RANK"], config["MODEL_TYPE"])
        scheduler.step()

        # sync losses across all devices, so the reports are accurate
        epoch_train_loss = torch.tensor([epoch_train_loss], device=f"cuda:{config['LOCAL_RANK']}")
        epoch_test_loss = torch.tensor([epoch_test_loss], device=f"cuda:{config['LOCAL_RANK']}")
        torch.distributed.all_reduce(epoch_train_loss, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(epoch_test_loss, op=torch.distributed.ReduceOp.SUM)
        epoch_train_loss = epoch_train_loss.item() / torch.distributed.get_world_size()
        epoch_test_loss = epoch_test_loss.item() / torch.distributed.get_world_size()

        is_new_best = epoch_test_loss < best_test_loss
        if config["LOCAL_RANK"] == 0:
            report_progress(config["GLOBAL_RANK"], f"Train Loss (RMSE in degs of steering): {180 * (epoch_train_loss ** 0.5): .4f}", epoch=epoch, total_epochs=config["EPOCHS"])
            report_progress(config["GLOBAL_RANK"], f"Test Loss (RMSE in degs of steering): {180 * (epoch_test_loss ** 0.5): .4f}", epoch=epoch, total_epochs=config["EPOCHS"])
            if is_new_best:
                report_progress(config["GLOBAL_RANK"], f"New best model found, prev. best test loss {180 * (best_test_loss ** 0.5)}", epoch=epoch, total_epochs=config["EPOCHS"])

        # Global rank 0 handles logging, saving, and updating best_test_loss
        if config["GLOBAL_RANK"] == 0:
            log_loss(epoch, epoch_train_loss, epoch_test_loss, config["MODEL_SAVE_DIR"])
            save_model(last_model_dir, model, optimizer, epoch, epoch_test_loss, best_test_loss, clean_old=True)
            if is_new_best:
                best_test_loss = epoch_test_loss
                save_model(best_model_dir, model, optimizer, epoch, epoch_test_loss, best_test_loss, clean_old=True)

        # Synchronize best_test_loss across all processes
        best_test_loss_tensor = torch.tensor([best_test_loss], device=model.device)
        torch.distributed.broadcast(best_test_loss_tensor, src=0)
        best_test_loss = best_test_loss_tensor.item()
    
    ddp_cleanup()
    close_tensorboard(writer)
    
if __name__ == "__main__":
    main()