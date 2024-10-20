import sys, os, datetime, time, torch, pickle, ast, json, argparse, warnings
warnings.filterwarnings('ignore')

from torch import nn
import numpy as np
from torch.cuda.amp import autocast,GradScaler
from torch.utils.data import WeightedRandomSampler
from torch.cuda.amp import autocast,GradScaler
from torch.utils.data import DataLoader
from CAVMAE import *

audio_model=CAVMAE()

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



def validate(audio_model,test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_time = AverageMeter()
    if not isinstance(audio_model, nn.DataParallel):
        audio_model = nn.DataParallel(audio_model)
    audio_model = audio_model.to(device)
    audio_model.eval()

    end = time.time()
    A_loss, A_loss_mae, A_loss_mae_a, A_loss_mae_v, A_loss_c, A_c_acc = [], [], [], [], [], []
    with torch.no_grad():
        for i, (a_input, v_input) in enumerate(zip(test_loader[0],test_loader[1])):
            a_input = a_input.squeeze(0)
            v_input = v_input.squeeze(0)
            a_input = a_input.to(device)
            v_input = v_input.to(device)
            with autocast():
                loss, loss_mae, loss_mae_a, loss_mae_v, loss_c, mask_a, mask_v, c_acc = audio_model(a_input, v_input)
                loss, loss_mae, loss_mae_a, loss_mae_v, loss_c, c_acc = loss.sum(), loss_mae.sum(), loss_mae_a.sum(), loss_mae_v.sum(), loss_c.sum(), c_acc.mean()
            A_loss.append(loss.to('cpu').detach())
            A_loss_mae.append(loss_mae.to('cpu').detach())
            A_loss_mae_a.append(loss_mae_a.to('cpu').detach())
            A_loss_mae_v.append(loss_mae_v.to('cpu').detach())
            A_loss_c.append(loss_c.to('cpu').detach())
            A_c_acc.append(c_acc.to('cpu').detach())
            batch_time.update(time.time() - end)
            end = time.time()

        loss = np.mean(A_loss)
        loss_mae = np.mean(A_loss_mae)
        loss_mae_a = np.mean(A_loss_mae_a)
        loss_mae_v = np.mean(A_loss_mae_v)
        loss_c = np.mean(A_loss_c)
        c_acc = np.mean(A_c_acc)

    return loss, loss_mae, loss_mae_a, loss_mae_v, loss_c, c_acc


def train(audio_model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("*"*20+' running on ' + str(device)+"*"*20)
    print()
    print()
    torch.set_grad_enabled(True)

    batch_time, per_sample_time, data_time, per_sample_data_time, per_sample_dnn_time = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    loss_av_meter, loss_a_meter, loss_v_meter, loss_c_meter = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    progress = []

    best_epoch, best_loss = 0, np.inf
    global_step, epoch = 0, 0
    start_time = time.time()
    exp_dir = '/home/viaj/project/src'

    def _save_progress():
        progress.append([epoch, global_step, best_epoch, best_loss,
                time.time() - start_time])
        with open("%s/progress.pkl" % exp_dir, "wb") as f:
            pickle.dump(progress, f)

    if not isinstance(audio_model, nn.DataParallel):
        audio_model = nn.DataParallel(audio_model)

    audio_model = audio_model.to(device)
    trainables = [p for p in audio_model.parameters() if p.requires_grad]
    print('+ Total parameter number is : {:.3f} million'.format(sum(p.numel() for p in audio_model.parameters()) / 1e6))
    print('+ Total trainable parameter number is : {:.3f} million'.format(sum(p.numel() for p in trainables) / 1e6))
    lr=0.001
    optimizer = torch.optim.Adam(trainables, lr, weight_decay=5e-7, betas=(0.95, 0.999))

    # use adapt learning rate scheduler, for preliminary experiments only, should not use for formal experiments
    lr_adapt=True
    if lr_adapt == True:
        lr_patience=2
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=lr_patience, verbose=True)
        print('+ Override to use adaptive learning rate scheduler.')
    else:
        lrscheduler_start=10
        lrscheduler_step=5
        lrscheduler_decay=0.5
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, list(range(lrscheduler_start, 1000,lrscheduler_step)),gamma=lrscheduler_decay)
        print('The learning rate scheduler starts at {:d} epoch with decay rate of {:.3f} every {:d} epoches'.format(lrscheduler_start, lrscheduler_decay, lrscheduler_step))
    print()
    #print('learning rate scheduler: {:s}'.format(str(scheduler.__dict__)))
    # #optional, save epoch 0 untrained model, for ablation study on model initialization purpose
    # torch.save(audio_model.state_dict(), "%s/models/audio_model.%d.pth" % (exp_dir, epoch))

    epoch += 1
    scaler = GradScaler()
    print()
    print("\t- current #steps=%s, #epochs=%s" % (global_step, epoch))
    print("\t- start training: ")
    n_epochs=2
    result = np.zeros([n_epochs, 10])  # for each epoch, 10 metrics to record
    audio_model.train()
    while epoch < n_epochs + 1:
        begin_time = time.time()
        end_time = time.time()
        audio_model.train()
        print()
        print(' '*10,'='*80)
        print()
        #print(datetime.datetime.now())
        print("\tcurrent #epochs=%s, #steps=%s" % (epoch, global_step))
        
        ###################################################
        a=torch.rand(100,1,128,128)
        v=torch.rand(100,224,224,3)
        a_train_loader=DataLoader(a)
        v_train_loader=DataLoader(v)
        ###################################################
        for i, (a_input, v_input) in enumerate(zip(a_train_loader,v_train_loader)):

            B = a_input.size(0)
            a_input = a_input.squeeze(0)
            v_input = v_input.squeeze(0)
            a_input = a_input.to(device, non_blocking=True)
            v_input = v_input.to(device, non_blocking=True)
            data_time.update(time.time() - end_time)
            per_sample_data_time.update((time.time() - end_time) / a_input.shape[0])
            dnn_start_time = time.time()

            with autocast():
                loss, loss_mae, loss_mae_a, loss_mae_v, loss_c, mask_a, mask_v, c_acc = audio_model(a_input, v_input)
                # this is due to for torch.nn.DataParallel, the output loss of 4 gpus won't be automatically averaged, need to be done manually
                loss, loss_mae, loss_mae_a, loss_mae_v, loss_c, c_acc = loss.sum(), loss_mae.sum(), loss_mae_a.sum(), loss_mae_v.sum(), loss_c.sum(), c_acc.mean()

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # loss_av is the main loss
            loss_av_meter.update(loss.item(), B)
            loss_a_meter.update(loss_mae_a.item(), B)
            loss_v_meter.update(loss_mae_v.item(), B)
            loss_c_meter.update(loss_c.item(), B)
            batch_time.update(time.time() - end_time)
            per_sample_time.update((time.time() - end_time)/a_input.shape[0])
            per_sample_dnn_time.update((time.time() - dnn_start_time)/a_input.shape[0])
            n_print_steps=100
            print_step = global_step % n_print_steps == 0
            early_print_step = epoch == 0 and global_step % (n_print_steps/10) == 0
            print_step = print_step or early_print_step

            if print_step and global_step != 0:
                print('\tEpoch:  + [{0}] - [{1}/{2}]\t\n'
                  '\t\t+ Per Sample Total Time {per_sample_time.avg:.5f}\t\n'
                  '\t\t+ Per Sample Data Time {per_sample_data_time.avg:.5f}\t\n'
                  '\t\t+ Per Sample DNN Time {per_sample_dnn_time.avg:.5f}\t\n'
                  '\t\t+ Train Total Loss {loss_av_meter.val:.4f}\t\n'
                  '\t\t+ Train MAE Loss Audio {loss_a_meter.val:.4f}\t\n'
                  '\t\t+ Train MAE Loss Visual {loss_v_meter.val:.4f}\t\n'
                  '\t\t+ Train Contrastive Loss {loss_c_meter.val:.4f}\t\n'
                  '\t\t+ Train Contrastive Acc {c_acc:.3f}\t\n'.format(
                   epoch, i, a.shape[0], per_sample_time=per_sample_time, per_sample_data_time=per_sample_data_time,
                      per_sample_dnn_time=per_sample_dnn_time, loss_av_meter=loss_av_meter, loss_a_meter=loss_a_meter, loss_v_meter=loss_v_meter, loss_c_meter=loss_c_meter, c_acc=c_acc), flush=True)
                if np.isnan(loss_av_meter.avg):
                    print("training diverged...")
                    return

            end_time = time.time()
            global_step += 1

        print('\t- start validation: ')
        a=torch.rand(10,1,128,128)
        v=torch.rand(10,224,224,3)
        a_test_loader=DataLoader(a)
        v_test_loader=DataLoader(v)
        test_loader = [a_test_loader,v_test_loader]
        eval_loss_av, eval_loss_mae, eval_loss_mae_a, eval_loss_mae_v, eval_loss_c, eval_c_acc = validate(audio_model, test_loader)

        print("\t\t\t+ Eval Audio MAE Loss: {:.6f}".format(eval_loss_mae_a))
        print("\t\t\t+ Eval Visual MAE Loss: {:.6f}".format(eval_loss_mae_v))
        print("\t\t\t+ Eval Total MAE Loss: {:.6f}".format(eval_loss_mae))
        print("\t\t\t+ Eval Contrastive Loss: {:.6f}".format(eval_loss_c))
        print("\t\t\t+ Eval Total Loss: {:.6f}".format(eval_loss_av))
        print("\t\t\t+ Eval Contrastive Accuracy: {:.6f}".format(eval_c_acc))

        print("\t\t\t+ Train Audio MAE Loss: {:.6f}".format(loss_a_meter.avg))
        print("\t\t\t+ Train Visual MAE Loss: {:.6f}".format(loss_v_meter.avg))
        print("\t\t\t+ Train Contrastive Loss: {:.6f}".format(loss_c_meter.avg))
        print("\t\t\t+ Train Total Loss: {:.6f}".format(loss_av_meter.avg))

        # train audio mae loss, train visual mae loss, train contrastive loss, train total loss
        # eval audio mae loss, eval visual mae loss, eval contrastive loss, eval total loss, eval contrastive accuracy, lr
        result[epoch-1, :] = [loss_a_meter.avg, loss_v_meter.avg, loss_c_meter.avg, loss_av_meter.avg, eval_loss_mae_a, eval_loss_mae_v, eval_loss_c, eval_loss_av, eval_c_acc, optimizer.param_groups[0]['lr']]
        np.savetxt('/home/viaj/project/src/result.csv', result, delimiter=',')
        print('\t- validation finished')

        if eval_loss_av < best_loss:
            best_loss = eval_loss_av
            best_epoch = epoch

        if best_epoch == epoch:
            torch.save(audio_model.state_dict(), "%s/models/best_audio_model.pth" % (exp_dir))
            torch.save(optimizer.state_dict(), "%s/models/best_optim_state.pth" % (exp_dir))
        save_model=False
        if save_model == True:
            torch.save(audio_model.state_dict(), "%s/models/audio_model.%d.pth" % (exp_dir, epoch))

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(-eval_loss_av)
        else:
            scheduler.step()

        print('Epoch-{0} lr: {1}'.format(epoch, optimizer.param_groups[0]['lr']))

        _save_progress()

        finish_time = time.time()
        print('epoch {:d} training time: {:.3f}'.format(epoch, finish_time-begin_time))

        epoch += 1

        batch_time.reset()
        per_sample_time.reset()
        data_time.reset()
        per_sample_data_time.reset()
        per_sample_dnn_time.reset()
        loss_av_meter.reset()
        loss_a_meter.reset()
        loss_v_meter.reset()
        loss_c_meter.reset()



        

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        


train(audio_model)