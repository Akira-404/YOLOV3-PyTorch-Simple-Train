import torch
from tqdm import tqdm
from loguru import logger


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


@logger.catch
def fit_one_epoch(model_train,
                  model,
                  yolo_loss,
                  loss_history,
                  optimizer,
                  curr_epoch,
                  epoch_step,
                  epoch_step_val,
                  train_dataloader,
                  val_dataloader,
                  epoch,
                  cuda,
                  save_period):
    loss = 0
    val_loss = 0

    model_train.train()
    logger.info('Start Train.')
    with tqdm(total=epoch_step, desc=f'epoch {curr_epoch + 1}/{epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(train_dataloader):

            # 一个epoch的结束
            if iteration >= epoch_step:
                break

            images, targets = batch[0], batch[1]
            with torch.no_grad():
                if cuda:
                    images = images.cuda()
                    targets = [ann.cuda() for ann in targets]
                    # images = torch.from_numpy(images).type(torch.FloatTensor).cuda()
                    # targets = [torch.from_numpy(ann).type(torch.FloatTensor).cuda() for ann in targets]
                else:
                    images = images
                    targets = [ann for ann in targets]
                    # images = torch.from_numpy(images).type(torch.FloatTensor)
                    # targets = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in targets]

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
            pbar.set_postfix(**{'loss': loss / (iteration + 1),
                                'lr': get_lr(optimizer)})
            pbar.update(1)

    logger.info('Finish Train.')

    # model_train.eval()
    # logger.info('Start Validation.')
    # with tqdm(total=epoch_step_val, desc=f'epoch {curr_epoch + 1}/{epoch}', postfix=dict, mininterval=0.3) as pbar:
    #     for iteration, batch in enumerate(val_dataloader):
    #         if iteration >= epoch_step_val:
    #             break
    #         images, targets = batch[0], batch[1]
    #         with torch.no_grad():
    #             if cuda:
    #                 images = torch.from_numpy(images).type(torch.FloatTensor).cuda()
    #                 targets = [torch.from_numpy(ann).type(torch.FloatTensor).cuda() for ann in targets]
    #             else:
    #                 images = torch.from_numpy(images).type(torch.FloatTensor)
    #                 targets = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in targets]
    #             #   清零梯度
    #             optimizer.zero_grad()
    #             #   前向传播
    #             outputs = model_train(images)
    #
    #             loss_value_all = 0
    #             #   计算损失
    #             for l in range(len(outputs)):
    #                 loss_item = yolo_loss(l, outputs[l], targets)
    #                 loss_value_all += loss_item
    #             loss_value = loss_value_all
    #
    #         val_loss += loss_value.item()
    #         pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1)})
    #         pbar.update(1)
    #
    # logger.info('Finish Validation')

    loss_history.append_loss(curr_epoch + 1, loss / epoch_step, val_loss / epoch_step_val)
    logger.info('epoch:' + str(curr_epoch + 1) + '/' + str(epoch))
    logger.info('Total Loss: %.3f || Val Loss: %.3f ' % (loss / epoch_step, val_loss / epoch_step_val))
    if (curr_epoch + 1) % save_period == 0 or curr_epoch + 1 == epoch:
        torch.save(model.state_dict(),
                   'logs/ep%03d-loss%.3f-val_loss%.3f.pth' % (
                       curr_epoch + 1, loss / epoch_step, val_loss / epoch_step_val))
