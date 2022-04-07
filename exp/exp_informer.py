from data.data_loader import Dataset_ETT_ms, Dataset_ETT_hour, Dataset_ETT_hour2, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred
from exp.exp_basic import Exp_Basic
from models.model import Informer, InformerStack

from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric

import numpy as np

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

import os
import time

import warnings
warnings.filterwarnings('ignore')

class Exp_Informer(Exp_Basic):
    def __init__(self, args):
        super(Exp_Informer, self).__init__(args)
        #self.debug = args.debug
    
    def _build_model(self):
        model_dict = {
            'informer':Informer,
            'informerstack':InformerStack,
        }
        if self.args.model=='informer' or self.args.model=='informerstack':
            if self.args.debug:
                import ipdb; ipdb.set_trace()
            e_layers = self.args.e_layers if self.args.model=='informer' else self.args.s_layers
            model = model_dict[self.args.model](
                self.args.enc_in,
                self.args.dec_in, 
                self.args.c_out, 
                self.args.seq_len, 
                self.args.label_len,
                self.args.pred_len, 
                self.args.factor,
                self.args.d_model, 
                self.args.n_heads, 
                e_layers, # self.args.e_layers,
                self.args.d_layers, 
                self.args.d_ff,
                self.args.dropout, 
                self.args.attn,
                self.args.embed,
                self.args.freq,
                self.args.activation,
                self.args.output_attention,
                self.args.distil,
                self.args.mix,
                self.device
            ).float()
        
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        args = self.args

        data_dict = {
            'ETTh1':Dataset_ETT_hour,
            'ETTh2':Dataset_ETT_hour,
            'ETTm1':Dataset_ETT_minute,
            'ETTm2':Dataset_ETT_minute,
            'WTH':Dataset_Custom,
            'ECL':Dataset_Custom,
            'Solar':Dataset_Custom,
            'custom':Dataset_Custom,
            'ETTh2ms1f2': Dataset_ETT_ms, # -> ms 
            'ETTh2f4t1': Dataset_ETT_hour2, # -> ms 
        }
        Data = data_dict[self.args.data] # data.data_loader.Dataset_ETT_ms
        timeenc = 0 if args.embed!='timeF' else 1 # timeenc=1

        if flag == 'test':
            shuffle_flag = False; drop_last = True; batch_size = args.batch_size; freq=args.freq
        elif flag=='pred':
            shuffle_flag = False; drop_last = False; batch_size = 1; freq=args.detail_freq
            Data = Dataset_Pred
        else:
            shuffle_flag = True; drop_last = True; batch_size = args.batch_size; freq=args.freq

        if args.debug:
            import ipdb; ipdb.set_trace()
        data_set = Data( # data.data_loader.Dataset_ETT_hour, the name of the class does not matter!
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag, # 'train'
            size=[args.seq_len, args.label_len, args.pred_len], # [336, 336, 168] -> [1152, 1152, 576]
            features=args.features, # "M" -> 'ms'
            target=args.target, # 'OT' = oil temperature
            inverse=args.inverse, # False
            timeenc=timeenc, # 1
            freq=freq, # 'h' -> 'ms'
            cols=args.cols, # None
            train_ratio = args.train_ratio,
            dev_ratio = args.dev_ratio,
            test_ratio = args.test_ratio,
            debug = args.debug,
        )
        if args.debug:
            import ipdb; ipdb.set_trace()
        print(flag, len(data_set)) # "train 97805"; "val 11867"; "test 11867" 
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)

        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim
    
    def _select_criterion(self):
        criterion =  nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        self.model.eval()
        if self.args.debug:
            import ipdb; ipdb.set_trace()
        total_loss = []
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(vali_loader):
            pred, true = self._process_one_batch(
                vali_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
            loss = criterion(pred.detach().cpu(), true.detach().cpu())
            total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        if self.args.debug:
            import ipdb; ipdb.set_trace()
        train_data, train_loader = self._get_data(flag = 'train')
        if self.args.debug:
            import ipdb; ipdb.set_trace()
        vali_data, vali_loader = self._get_data(flag = 'val')
        if self.args.debug:
            import ipdb; ipdb.set_trace()
        test_data, test_loader = self._get_data(flag = 'test')
        
        if self.args.debug:
            import ipdb; ipdb.set_trace()
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        
        train_steps = len(train_loader) # 3056
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        
        model_optim = self._select_optimizer()
        criterion =  self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            
            if self.args.debug:
                import ipdb; ipdb.set_trace()
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(train_loader):
                if self.args.debug:
                    import ipdb; ipdb.set_trace()
                iter_count += 1
                
                model_optim.zero_grad()
                if self.args.debug:
                    import ipdb; ipdb.set_trace()
                pred, true = self._process_one_batch(
                    train_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
                # TODO we shall only use the "target" as our loss!
                #import ipdb ;ipdb.set_trace()
                loss = criterion(pred[:, :, -1], true[:, :, -1])
                train_loss.append(loss.item())
                
                if (i+1) % 100==0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time()-time_now)/iter_count
                    left_time = speed*((self.args.train_epochs - epoch)*train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                
                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch+1, time.time()-epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop: # and epoch == self.args.train_epochs-1:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch+1, self.args)
            
        best_model_path = path+'/'+'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        
        return self.model

    def average(self, preds):
        id2values = dict() # batch.id : a list of predicted values
        bnum, blen, targetlen = preds.shape
        # NOTE only take the right-most target for now!
        for bid in range(bnum):
            # bid = batch index
            for j in range(blen):
                # e.g. j=0 to 167
                real_data_id = bid + j # + seq_len if required
                vlist = id2values[real_data_id] if real_data_id in id2values else []
                vlist.append(preds[bid, j, -1]) # TODO can change -1 to other values or ranges...
                id2values[real_data_id] = vlist

        outlist = list()
        dkeys = id2values.keys()

        for data_id in sorted(dkeys):
            vlist = id2values[data_id]
            amean = np.mean(vlist)
            outlist.append(amean)

        outlist = np.array(outlist)
        return outlist

    def save_txt(self, afn, preds):
        with open(afn, 'w') as bw:
            for pred in preds:
                bw.write('{}\n'.format(pred))
    
    def save_txt2(self, afn, trues, preds):
        with open(afn, 'w') as bw:
            bw.write('true\tpred\n')
            for true, pred in zip(trues, preds):
                bw.write('{}\t{}\n'.format(true, pred))


    def test(self, setting):
        test_data, test_loader = self._get_data(flag='test')
        
        self.model.eval()
        
        preds = []
        trues = []
        
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(test_loader):
            pred, true = self._process_one_batch(
                test_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
            preds.append(pred.detach().cpu().numpy())
            trues.append(true.detach().cpu().numpy())

        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # preds.shape = [5088, 168, 5]
        # trues.shape = [5088, 168, 5]
        # NOTE, average the values of duplicately predicted points 

        #if args.debug:
        #import ipdb; ipdb.set_trace()
        preds = self.average(preds)
        trues = self.average(trues)


        # result save
        folder_path = './results/' + setting +'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}, rmse:{}, mape:{}, mspe:{}'.format(mse, mae, rmse, mape, mspe))

        np.save(folder_path+'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path+'pred.npy', preds)
        np.save(folder_path+'true.npy', trues)

        self.save_txt(folder_path + 'pred.txt', preds)
        self.save_txt(folder_path + 'true.txt', trues)
        self.save_txt2(folder_path + 'true_pred.txt', trues, preds)

        # recover the original data values:
        preds_orig = test_data.inverse_transform_tar(torch.from_numpy(preds))
        trues_orig = test_data.inverse_transform_tar(torch.from_numpy(trues))


        preds_orig_np = preds_orig.numpy()
        trues_orig_np = trues_orig.numpy()

        self.save_txt(folder_path + 'pred_orig.txt', preds_orig_np)
        self.save_txt(folder_path + 'true_orig.txt', trues_orig_np)
        self.save_txt2(folder_path + 'true_pred_orig_all.txt', trues_orig_np, preds_orig_np)


        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')
        
        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path+'/'+'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        self.model.eval()
        
        preds = []
        
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(pred_loader):
            pred, true = self._process_one_batch(
                pred_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
            preds.append(pred.detach().cpu().numpy())

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        
        # result save
        folder_path = './results/' + setting +'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        np.save(folder_path+'real_prediction.npy', preds)
        
        return

    def _process_one_batch(self, dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark):
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float()

        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)

        # decoder input
        if self.args.padding==0:
            dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()
        elif self.args.padding==1:
            dec_inp = torch.ones([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()
        dec_inp = torch.cat([batch_y[:,:self.args.label_len,:], dec_inp], dim=1).float().to(self.device)
        # encoder - decoder
        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        else:
            if self.args.output_attention:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
            else:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        if self.args.inverse:
            # TODO only change "outputs" is not enough... "batch_y"=the reference should also be changed...
            outputs = dataset_object.inverse_transform(outputs)
        f_dim = -1 if self.args.features=='MS' else 0
        batch_y = batch_y[:,-self.args.pred_len:,f_dim:].to(self.device)

        return outputs, batch_y
