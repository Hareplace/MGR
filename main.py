# main.py

import torch
import yaml
import argparse
import numpy as np
from utils.seed import set_seed
from utils.logger import setup_logger
from models.MGR import MGR
from dataset.dataloader import get_dataloaders  # You must implement this

def train_one_epoch(model, dataloader, optimizer):
    model.train()
    total_loss = 0
    for batch in dataloader:
        x_dict, modality_mask, labels = batch['inputs'], batch['modality_mask'], batch['labels']
        optimizer.zero_grad()
        output = model(x_dict, modality_mask, labels)
        loss = output['loss']
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model, dataloader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in dataloader:
            x_dict, modality_mask, labels = batch['inputs'], batch['modality_mask'], batch['labels']
            output = model(x_dict, modality_mask)
            preds = output['logits'].argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total

def main(config_path):
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    logger = setup_logger()

    results = []
    for absence_seed in range(4):
        for train_seed in range(4):
            seed = absence_seed * 10 + train_seed
            set_seed(seed)
            logger.info(f"==> Seed: {seed}")

            # Load data
            train_loader, val_loader, test_loader = get_dataloaders(config, absence_seed=absence_seed)

            # Initialize model
            model = MGR(config).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

            # Training loop
            best_val_acc, best_model = 0, None
            for epoch in range(config['epochs']):
                train_loss = train_one_epoch(model, train_loader, optimizer)
                val_acc = evaluate(model, val_loader)
                logger.info(f"[Epoch {epoch}] Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f}")
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_model = model.state_dict()

            # Test with best model
            model.load_state_dict(best_model)
            test_acc = evaluate(model, test_loader)
            results.append(test_acc)
            logger.info(f"[Test @ seed {seed}] Acc: {test_acc:.4f}")

    # Report mean ± std
    mean_acc = np.mean(results)
    std_acc = np.std(results)
    logger.info(f"Final Test Accuracy over 16 runs: {mean_acc:.4f} ± {std_acc:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    args = parser.parse_args()
    main(args.config)
