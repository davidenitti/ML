import torch
import os


def save_model(checkpoint, epoch, model, optimizer, scheduler):
    if os.path.exists(checkpoint):
        os.rename(checkpoint, checkpoint + '.old')
    if not os.path.exists(os.path.dirname(checkpoint)):
        os.makedirs(os.path.dirname(checkpoint))
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict()
    }, checkpoint)
    # torch.save(model, checkpoint + "raw")
    print('saved')


def load_model(checkpoint, model, optimizer, scheduler):
    if os.path.exists(checkpoint):
        try:
            checkpoint = torch.load(checkpoint)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        except Exception as e:
            print('error checkpoint', e)
            checkpoint = torch.load(checkpoint + '.old')
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return checkpoint['epoch']
    else:
        return 0


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
