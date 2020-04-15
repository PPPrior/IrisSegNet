import logging

import torch
import torch.nn as nn
from torch import optim
from skimage import filters
# from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from tensorboardX import SummaryWriter

from model import fcn_vgg16
from data.dataset import IrisDataset
from data.preprocess import *


def structure():
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dummy_input = torch.randn(size=[1, 1, 288, 288])  # (batch_size, channel, width, width)
    model = fcn_vgg16(arch_type='fcn16s')
    out = model(dummy_input)
    print(out.size())


def fit(model, device, data_path, epochs=5, batch_size=2, lr=0.001, val_percent=0.1):
    # get image/mask data
    dataset = IrisDataset(*data_path)
    n_valid = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_valid
    train_ds, valid_ds = random_split(dataset, [n_train, n_valid])
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    valid_dl = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True,
                          drop_last=True)

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_valid}
        Device:          {device.type}
    ''')

    writer = SummaryWriter()
    global_step = 0

    optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    loss_func = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='image') as pbar:
            model.train()
            for image, mask in train_dl:
                loss, _ = loss_batch(model, device, loss_func, image, mask, optimizer)

                writer.add_scalar('Loss/train', loss, global_step)
                pbar.set_postfix(**{'loss (batch)': loss})
                pbar.update(image.shape[0])

                global_step += 1
                if global_step % (len(dataset) // (10 * batch_size)) == 0:
                    for tag, value in model.named_parameters():
                        tag = tag.replace('.', '/')
                        writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
                        writer.add_histogram('grads/' + tag, value.data.cpu().numpy(), global_step)

                    model.eval()
                    with torch.no_grad():
                        losses, nums = zip(
                            *[loss_batch(model, device, loss_func, image, mask) for image, mask in valid_dl]
                        )
                    val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
                    scheduler.step(val_loss)

                    writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)
                    if model.n_classes > 1:
                        logging.info('Validation cross entropy: {}'.format(val_loss))
                        writer.add_scalar('Loss/test', val_loss, global_step)
                    else:
                        logging.info('Validation Dice Coeff: {}'.format(val_loss))
                        writer.add_scalar('Dice/test', val_loss, global_step)

                    writer.add_images('images', image, global_step)
                    if model.n_classes == 1:
                        writer.add_images('masks/true', mask, global_step)
                        image = image.to(device=device, dtype=torch.float32)
                        writer.add_images('masks/pred', torch.sigmoid(model(image)) > 0.5, global_step)

    writer.close()


def loss_batch(model, device, loss_func, image, mask, opt=None):
    image = image.to(device=device, dtype=torch.float32)
    mask_type = torch.long if model.n_classes > 1 else torch.float32
    mask = mask.to(device=device, dtype=mask_type)

    loss = loss_func(model(image), mask)

    if opt is not None:
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_value_(model.parameters(), 0.1)
        opt.step()

    return loss.item(), len(image)


if __name__ == '__main__':
    # structure()
    model = fcn_vgg16(arch_type='fcn8s')
    # model.load_state_dict(torch.load('./model_3.pth'))
    # path = ('C:/Users/Prior-Laptop/Desktop/data/image', 'C:/Users/Prior-Laptop/Desktop/data/mask')
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model.to(device=device)
    # fit(model, device, path, lr=0.00005, epochs=5, batch_size=4)
    # torch.save(model.state_dict(), './model_v2.pth')

    id = '020-R-30'
    image = 'C:/Users/Prior-Laptop/Desktop/data/image/' + id + '.png'
    mask = 'C:/Users/Prior-Laptop/Desktop/data/mask/' + id + '.png'
    image, mask = roi(image, mask)

    fig, axes = plt.subplots(ncols=3, sharex=True, sharey=True, figsize=(12, 4))
    ax = axes.ravel()
    ax[0].imshow(np.squeeze(image), cmap=plt.cm.gray)
    ax[1].imshow(np.squeeze(mask), cmap='binary')

    model.load_state_dict(torch.load('./model_v1.pth'))
    image = np.array([image])
    pred = model(torch.from_numpy(image))
    pred = torch.sigmoid(pred).detach().numpy()
    pred = np.squeeze(pred)
    pred = filters.gaussian(pred, sigma=5)
    ax[2].imshow(pred > 0.5, cmap='binary')

    plt.show()
