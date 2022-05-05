import torch
from tqdm import tqdm
from loguru import logger


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


@logger.catch
def fit_one_epoch(model,
                  model_train,
                  train_dataloader,
                  yolo_loss,
                  loss_history,
                  optimizer,
                  curr_epoch: int,
                  epoch_step: int,
                  epoch: int,
                  cuda: bool,
                  save_period: int):
    loss = 0

    model_train.train()
    logger.info('Start Train.')
    with tqdm(total=epoch_step, desc=f'epoch {curr_epoch + 1}/{epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(train_dataloader):

            # 一个epoch的结束
            if iteration >= epoch_step:
                break

            images, targets = batch[0], batch[1]
            with torch.no_grad():
                images = images.cuda() if cuda else images
                targets = [ann.cuda() for ann in targets] if cuda else [ann for ann in targets]

            optimizer.zero_grad()

            outputs = model_train(images)

            loss_value_all = 0

            for l in range(len(outputs)):
                loss_item = yolo_loss(l, outputs[l], targets)
                loss_value_all += loss_item

            loss_value = loss_value_all
            loss_value.backward()
            optimizer.step()
            loss += loss_value.item()
            pbar.set_postfix(**{
                'loss': loss / (iteration + 1),
                'lr': get_lr(optimizer)
            })

            pbar.update(1)

    logger.info('Finish Train.')

    loss_history.append_loss(loss / epoch_step)
    logger.info('epoch:' + str(curr_epoch + 1) + '/' + str(epoch))
    logger.info(f'Total Loss: {loss / epoch_step}')
    if (curr_epoch + 1) % save_period == 0 or curr_epoch + 1 == epoch:
        torch.save(model.state_dict(), f'logs/epoch{curr_epoch + 1}_loss{loss / epoch_step}.pth')
