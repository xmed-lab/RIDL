"""

训练脚本
"""

import os
from time import time

import argparse


import numpy as np
import shutil
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score

import SimpleITK as sitk


import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from dataset.dataset import Dataset_AF_BXSEG_PYRAD_SHAPE




parser = argparse.ArgumentParser(description='Hyper-parameters management')


parser.add_argument('--experiment',default = 'experiment/dbg',help='Testset path')
parser.add_argument('--initweight',default = None,help='Testset path')

parser.add_argument('--Ytestonly', dest='testonly', action='store_true')
parser.add_argument('--Ntestonly', dest='testonly', action='store_false')
parser.set_defaults(testonly=False)

parser.add_argument('--input_type', type=int, default=2, help='stride of sliding window')
parser.add_argument('--output_type', type=int, default=2, help='stride of sliding window')

parser.add_argument('--Yfconnect', dest='fconnect', action='store_true')
parser.add_argument('--Nfconnect', dest='fconnect', action='store_false')
parser.set_defaults(fconnect=False)


parser.add_argument('--loss_type',default = '',help='Testset path')

parser.add_argument('--Ygenorig', dest='genorig', action='store_true')
parser.add_argument('--Ngenorig', dest='genorig', action='store_false')
parser.set_defaults(genorig=False)


parser.add_argument('--gpu',default = '3',help='Testset path')

parser.add_argument('--pin_memory', type=bool, default=True, help='with clinic data')
parser.add_argument('--cudnn_benchmark', type=bool, default=True, help='with clinic data')

parser.add_argument('--batch_size', type=int, default=1, help='stride of sliding window')
parser.add_argument('--epoch', type=int, default=25, help='stride of sliding window')
parser.add_argument('--num_workers', type=int, default=3, help='stride of sliding window')

parser.add_argument('--slice_len', type=int, default=96, help='stride of sliding window')

parser.add_argument('--learning_rate_decay_in', type=int, default=30, help='stride of sliding window')

parser.add_argument('--dropout', type=float, default=0.3,help='learning rate (default: 0.0001)')

parser.add_argument('--learning_rate', type=float, default=1e-4,help='learning rate (default: 0.0001)')
parser.add_argument('--alpha', type=float, default=0.33,help='learning rate (default: 0.0001)')
parser.add_argument('--w_cls', type=float, default=1,help='learning rate (default: 0.0001)')

parser.add_argument('--w_rec', type=float, default=1,help='learning rate (default: 0.0001)')

parser.add_argument('--fc_layer_num', type=int, default=2,help='learning rate (default: 0.0001)')
parser.add_argument('--rad_version', type=int, default=4,help='learning rate (default: 0.0001)')

parser.add_argument('--data_dir',default = 'miccai_ridl/DATA/HU_SGBX_CLEAN_DATA_STDR/labels_SGBX_CV5.csv',help='Testset path')
parser.add_argument('--radstat_dir',default = 'miccai_ridl/DATA/HU_SGBX_CLEAN_DATA_STDR/RADFEAT_CRP/rad_feat_stat.csv',help='Testset path')


parser.add_argument('--glb_ft',default = '',help='Testset path')


args = parser.parse_args()

learning_rate_decay = [args.learning_rate_decay_in]

print("LEN GLB_FT", len(args.glb_ft))
if len(args.glb_ft) > 0:
    GLB_FT = [item.strip() for item in args.glb_ft.split(',')]
else:
    GLB_FT = []
print("GLB_FT is {}".format(GLB_FT))

print(args.__dict__, flush = True)


if os.path.isdir(args.experiment):
    pass
else:
    os.makedirs(args.experiment)

CV_NUM = 5

AVG_DICT = {
    'net_best.pth': {
        'AUC_AVG_LIST':[],
        'PRAVG_LIST':[],
        'ACC_LIST':[],
        'F1_LIST':[]
    }, 
    'net_best_cls.pth': {
        'AUC_AVG_LIST':[],
        'PRAVG_LIST':[],
        'ACC_LIST':[],
        'F1_LIST':[]
    }, 
    'net_best_auc.pth': {
        'AUC_AVG_LIST':[],
        'PRAVG_LIST':[],
        'ACC_LIST':[],
        'F1_LIST':[]
    }, 
    'net_best_avp.pth': {
        'AUC_AVG_LIST':[],
        'PRAVG_LIST':[],
        'ACC_LIST':[],
        'F1_LIST':[]
    }
}





para_experiment_orig = args.experiment



for cv_itr in range(0, CV_NUM):
    print("Processing CV {} out of {}".format(cv_itr, CV_NUM))
    args.experiment = para_experiment_orig + "/CV{}".format(cv_itr)
    
    print("Args experiment check {}".format(args.experiment))

    
    from networks.HYBRID import HYBRID, init
    net = HYBRID(training=True, 
                    out_channel = args.output_type, 
                    in_channel = args.input_type, 
                    dropout = args.dropout,
                    fconnect = args.fconnect,
                    fc_layer_num = args.fc_layer_num,
                    glb_ft_num=len(GLB_FT))
    net.apply(init)

    print('net total parameters:', sum(param.numel() for param in net.parameters()))

    if os.path.isdir(args.experiment):
        pass
    else:
        os.makedirs(args.experiment)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    cudnn.benchmark = args.cudnn_benchmark

    net = torch.nn.DataParallel(net).cuda()
    if args.initweight:
        assert "CV0" in args.initweight, "Use CV0 path"
        load_initpath = args.initweight.replace("CV0", "CV{}".format(cv_itr))
        print("loading from {}".format(load_initpath))
        checkpoint = torch.load(load_initpath)
        net.module.fc_0 = torch.nn.Linear(256, 256 ).cuda()
        net.module.fc_2 = torch.nn.Linear(256, 1 ).cuda()
        net.load_state_dict(checkpoint, strict=False)

        net.module.fc_0 = torch.nn.Linear(256 + len(GLB_FT), 256 + len(GLB_FT) ).cuda()
        net.module.fc_2 = torch.nn.Linear(256 + len(GLB_FT), 1 ).cuda()

    net.train()


    rad_version = args.rad_version

    Dataset_CHOICE = Dataset_AF_BXSEG_PYRAD_SHAPE

    train_ds = Dataset_CHOICE(split='train', 
                                cv_num = cv_itr, 
                                cv_num_total = CV_NUM,
                                slice_len = args.slice_len, 
                                rndm = True, 
                                shapeft_list = GLB_FT,
                                rad_version = rad_version,
                                RAW_LB_PATH = args.data_dir
                                )
    label_list_trn = train_ds.label
    trn_pos_weight = (len(label_list_trn) -  sum(label_list_trn)) / sum(label_list_trn)
    train_dl = DataLoader(train_ds, args.batch_size, True, num_workers=args.num_workers, pin_memory=args.pin_memory)

    val_ds = Dataset_CHOICE(split='val', 
                                cv_num = cv_itr, 
                                cv_num_total = CV_NUM,
                                slice_len = args.slice_len, 
                                rndm = False, 
                                shapeft_list = GLB_FT,
                                rad_version = rad_version,
                                RAW_LB_PATH = args.data_dir
                                )
    label_list_val = val_ds.label
    val_pos_weight = (len(label_list_val) -  sum(label_list_val)) / sum(label_list_val)
    
    val_dl = DataLoader(val_ds, args.batch_size, False, num_workers=args.num_workers, pin_memory=args.pin_memory)

    test_ds = Dataset_CHOICE(split='test', 
                        cv_num = cv_itr, 
                        cv_num_total = CV_NUM,
                        slice_len = args.slice_len, 
                        rndm = False, 
                        shapeft_list = GLB_FT,
                        rad_version = rad_version,
                        RAW_LB_PATH = args.data_dir
                        )
    test_dl = DataLoader(test_ds, 1, False, num_workers=args.num_workers, pin_memory=args.pin_memory)


    loss_func_l1 = torch.nn.L1Loss()
    
    loss_cls =  torch.nn.BCEWithLogitsLoss()
    if args.loss_type == "bln":        
        loss_cls_trn =  torch.nn.BCEWithLogitsLoss(pos_weight = torch.tensor(trn_pos_weight))
        loss_cls_val =  torch.nn.BCEWithLogitsLoss(pos_weight = torch.tensor(val_pos_weight))
    else:
        loss_cls_trn =  torch.nn.BCEWithLogitsLoss()
        loss_cls_val = torch.nn.BCEWithLogitsLoss()


    opt = torch.optim.Adam(net.parameters(), lr=args.learning_rate)

    lr_decay = torch.optim.lr_scheduler.MultiStepLR(opt, learning_rate_decay)

    # 深度监督衰减系数
    alpha = args.alpha

    best_val_loss = 99999
    best_cls_loss = 99999
    best_avp_loss = 0
    best_auc_loss = 0
    # 训练网络
    start = time()

    
    if os.path.isfile(os.path.join(args.experiment, 'net_best.pth')) or os.path.isfile(os.path.join(args.experiment, 'net_best_cls.pth')) or os.path.isfile(os.path.join(args.experiment, 'net_best_auc.pth')) or os.path.isfile(os.path.join(args.experiment, 'net_best_avp.pth')):
        assert args.testonly, "Model files already exist"


    if args.testonly:
        epoch_ttltrn = 0
        print("Skipping training")
    else:
        epoch_ttltrn = args.epoch

    pred_out_file_check = os.path.join(args.experiment, "net_best_avp", "test_pred", "pred_performance_TEST.csv")

    for epoch in range(epoch_ttltrn):    
        if os.path.isfile(pred_out_file_check):
            print("file {} already exists. Skipping train".format(pred_out_file_check))
            break

        lr_decay.step()

        mean_loss = []
        mean_loss_val = []
        mean_acc_val = []
        mean_cls_val = []

        val_pred_score = []
        val_gt = []


        
        mean_loss_tstchk = []
        mean_acc_tstchk = []
        mean_cls_tstchk = []

        tstchk_pred_score = []
        tstchk_gt = []



        net.train()
        for step, (ct, glb_shp_ft, reint_lb, pat_id, _, _) in enumerate(train_dl):

            ct = ct[:,:2, ...].cuda()
            glb_shp_ft = glb_shp_ft.cuda()
            reint_lb = reint_lb.cuda().float()

            autosegin = ct

            outputs = net(autosegin, glb_shp_ft)

            if args.output_type > 1:

                loss1_list = []
                loss2_list = []
                loss3_list = []
                loss4_list = []

                loss1_list.append(loss_func_l1(outputs[0][:,0:1,...], ct[:,0:1,...]))
                loss2_list.append(loss_func_l1(outputs[1][:,0:1,...], ct[:,0:1,...]))
                loss3_list.append(loss_func_l1(outputs[2][:,0:1,...], ct[:,0:1,...]))
                loss4_list.append(loss_func_l1(outputs[3][:,0:1,...], ct[:,0:1,...]))

                loss1_list.append(loss_cls(outputs[0][:,1:2,...], ct[:,1:2,...]))
                loss2_list.append(loss_cls(outputs[1][:,1:2,...], ct[:,1:2,...]))
                loss3_list.append(loss_cls(outputs[2][:,1:2,...], ct[:,1:2,...]))
                loss4_list.append(loss_cls(outputs[3][:,1:2,...], ct[:,1:2,...]))

                loss1 = loss1_list[0] + loss1_list[1]
                loss2 = loss2_list[0] + loss2_list[1]
                loss3 = loss3_list[0] + loss3_list[1]
                loss4 = loss4_list[0] + loss4_list[1]              

                loss5 = loss_cls_trn(outputs[4].reshape(-1), reint_lb)
                loss = ((loss1 + loss2 + loss3) * alpha + loss4) * args.w_rec + loss5 * args.w_cls
                mean_loss.append(loss4)

                loss1_0_item = loss1.item()
                loss2_0_item = loss2.item()
                loss3_0_item = loss3.item()
                loss4_0_item = loss4.item()
                loss5_item = loss5.item()

            else:
                assert 1==2, "check input dim"

            opt.zero_grad()
            loss.backward()
            opt.step()

            if step % 5 == 0:
                
                print('epoch:{}, step:{}, loss1_0:{:.3f}, loss2_0:{:.3f}, loss3_0:{:.3f}, loss4_0:{:.3f}, loss5:{:.3f}, time:{:.3f} min'
                    .format(epoch, step, loss1_0_item, loss2_0_item, loss3_0_item, loss4_0_item, loss5_item, (time() - start) / 60), flush = True)

        mean_loss = sum(mean_loss) / len(mean_loss)
        print("epoch_trn {}  mean {}".format(epoch, mean_loss), flush = True)

        net.eval()
        large_inter_0 = 0
        large_union_0 = 0
        small_inter_0 = 0
        small_union_0 = 0
        for step, (ct, glb_shp_ft, reint_lb, pat_id, _, _) in enumerate(val_dl):

            ct = ct[:,:2, ...].cuda()
            glb_shp_ft = glb_shp_ft.cuda()
            reint_lb = reint_lb.cuda().float()

            autosegin = ct

            with torch.no_grad():
                outputs_all = net(autosegin, glb_shp_ft)

            outputs = outputs_all[0]
            outputs_reint = outputs_all[2]

            val_pred_score.append(torch.sigmoid(outputs_reint.reshape(-1)).detach().cpu().numpy())
            val_gt.append(reint_lb.detach().cpu().numpy())

            outputs_reint_pred = outputs_reint > 0
            outputs_reint_pred_acc = (outputs_reint_pred.reshape(-1) == reint_lb)


            loss1_list = []
            loss1_list.append(loss_func_l1(outputs[:,0:1,...], ct[:,0:1,...]))
                
            loss1_list.append(loss_cls(outputs[:,1:2,...], ct[:,1:2,...]))
            
            loss1 = loss1_list[0] + loss1_list[1]      

            mean_loss_val.append(loss1.item())
            loss_reint = loss_cls_trn(outputs_reint.reshape(-1), reint_lb)
            mean_acc_val.append(outputs_reint_pred_acc.item())
            mean_cls_val.append(loss_reint.item())

            if step % 5 == 0:
                
                print('epoch_val:{}, step:{}, loss4:{:.3f}, dice 1:{:.3f}, dice 2:{:.3f}, dice ttl:{:.3f}, time:{:.3f} min'
                    .format(epoch, 
                            step, 
                            loss1.item(), 
                            2 * large_inter_0 / (large_union_0 + large_inter_0 + 0.00001), 
                            2 * small_inter_0 / (small_union_0 + small_inter_0 + 0.00001),  
                            2 * (large_inter_0 + small_inter_0) / (large_union_0 + large_inter_0 + small_union_0 + small_inter_0 + 0.00001),  
                            (time() - start) / 60), 
                            flush = True)

        mean_loss_val = sum(mean_loss_val) / len(mean_loss_val)
        mean_acc_val = sum(mean_acc_val) / len(mean_acc_val)
        mean_cls_val = sum(mean_cls_val) / len(mean_cls_val)
        roc_auc_value = roc_auc_score(val_gt, val_pred_score)
        prec_avg_value = average_precision_score(val_gt, val_pred_score)

        print("perextract_val_CV {}, epoch_val {}  mean {}".format(cv_itr, epoch, mean_loss_val), flush = True)
        print("perextract_val_CV {}, epoch_val {}  mean acc {}".format(cv_itr, epoch, mean_acc_val), flush = True)
        print("perextract_val_CV {}, epoch_val {}  mean cls {}".format(cv_itr, epoch, mean_cls_val), flush = True)
        print("perextract_val_CV {}, epoch_val {}  roc auc {}".format(cv_itr, epoch, roc_auc_value), flush = True)
        print("perextract_val_CV {}, epoch_val {}  avg pr {}".format(cv_itr, epoch, prec_avg_value), flush = True)
        

        if best_cls_loss > mean_cls_val:
            print("best model so far cls {}".format(epoch), flush = True)        
            torch.save(net.state_dict(), './{}/net_best_cls.pth'.format(args.experiment))
            best_cls_loss = mean_cls_val

        if epoch % 10 == 0 and epoch != 0:
            alpha *= 0.8


        net.eval()
        large_inter_0 = 0
        large_union_0 = 0
        small_inter_0 = 0
        small_union_0 = 0
        for step, (ct, glb_shp_ft, reint_lb, pat_id, _, _) in enumerate(test_dl):

            ct = ct[:,:2, ...].cuda()
            glb_shp_ft = glb_shp_ft.cuda()
            reint_lb = reint_lb.cuda().float()

            autosegin = ct

            with torch.no_grad():
                outputs_all = net(autosegin, glb_shp_ft)

            outputs = outputs_all[0]
            outputs_reint = outputs_all[2]

            tstchk_pred_score.append(torch.sigmoid(outputs_reint.reshape(-1)).detach().cpu().numpy())
            tstchk_gt.append(reint_lb.detach().cpu().numpy())

            outputs_reint_pred = outputs_reint > 0
            outputs_reint_pred_acc = (outputs_reint_pred.reshape(-1) == reint_lb)

            if args.output_type > 1:
                loss1_list = []
                loss1_list.append(loss_func_l1(outputs[:,0:1,...], ct[:,0:1,...]))
                    
                loss1_list.append(loss_cls(outputs[:,1:2,...], ct[:,1:2,...]))
                
                loss1 = loss1_list[0] + loss1_list[1]

            else:
                assert 1==2, "check output dim"

            mean_loss_tstchk.append(loss1.item())
            loss_reint = loss_cls(outputs_reint.reshape(-1), reint_lb)
            mean_acc_tstchk.append(outputs_reint_pred_acc.item())
            mean_cls_tstchk.append(loss_reint.item())

            if step % 5 == 0:
                
                print('epoch_val:{}, step:{}, loss4:{:.3f}, dice 1:{:.3f}, dice 2:{:.3f}, dice ttl:{:.3f}, time:{:.3f} min'
                    .format(epoch, 
                            step, 
                            loss1.item(), 
                            2 * large_inter_0 / (large_union_0 + large_inter_0 + 0.00001), 
                            2 * small_inter_0 / (small_union_0 + small_inter_0 + 0.00001),  
                            2 * (large_inter_0 + small_inter_0) / (large_union_0 + large_inter_0 + small_union_0 + small_inter_0 + 0.00001),  
                            (time() - start) / 60), 
                            flush = True)

        mean_loss_tstchk = sum(mean_loss_tstchk) / len(mean_loss_tstchk)
        mean_acc_tstchk = sum(mean_acc_tstchk) / len(mean_acc_tstchk)
        mean_cls_tstchk = sum(mean_cls_tstchk) / len(mean_cls_tstchk)
        roc_auc_value = roc_auc_score(tstchk_gt, tstchk_pred_score)
        prec_avg_value = average_precision_score(tstchk_gt, tstchk_pred_score)


        print("perextract_tst_CV {}, epoch_val {}  mean {}".format(cv_itr, epoch, mean_loss_tstchk), flush = True)
        print("perextract_tst_CV {}, epoch_val {}  mean acc {}".format(cv_itr, epoch, mean_acc_tstchk), flush = True)
        print("perextract_tst_CV {}, epoch_val {}  mean cls {}".format(cv_itr, epoch, mean_cls_tstchk), flush = True)
        print("perextract_tst_CV {}, epoch_val {}  roc auc {}".format(cv_itr, epoch, roc_auc_value), flush = True)
        print("perextract_tst_CV {}, epoch_val {}  avg pr {}".format(cv_itr, epoch, prec_avg_value), flush = True)
        



    print("Processing testing for CV ITr {}".format(cv_itr))
    print('net total parameters:', sum(param.numel() for param in net.parameters()))

    for split_itr in ['TEST']:
        large_inter_0_tst = 0
        large_union_0_tst = 0
        small_inter_0_tst = 0
        small_union_0_tst = 0

        model_itr_list = ['net_best_cls.pth']

        for model_load_itr in model_itr_list:
            pred_out_dir = os.path.join(args.experiment, os.path.splitext(model_load_itr)[0], "test_pred")
            os.makedirs(pred_out_dir, exist_ok = True)

            
            from networks.HYBRID import HYBRID, init
            net = HYBRID(training=True, 
                            out_channel = args.output_type, 
                            in_channel = args.input_type, 
                            dropout = args.dropout,
                            fconnect = args.fconnect,
                            fc_layer_num = args.fc_layer_num,
                            glb_ft_num=len(GLB_FT))
            net.apply(init)

            net = torch.nn.DataParallel(net).cuda()
            net.eval()
            net.load_state_dict(torch.load(os.path.join(args.experiment, model_load_itr)))


            start = time()


            mean_loss_tst = []
            mean_acc_tst = []
            mean_cls_tst = []

            tst_pred_score = []
            tst_gt = []
            pat_id_list = []

            with torch.no_grad():
                for step, (ct, glb_shp_ft, reint_lb, pat_id, ct_raw, bnd_data ) in enumerate(test_dl):

                    ct = ct[:,:2,...].cuda()
                    glb_shp_ft = glb_shp_ft.cuda()
                    reint_lb = reint_lb.cuda().float()

                    autosegin = ct

                    outputs, outputs_encoded, outputs_reint, cbam_spat  = net(autosegin, glb_shp_ft)
                    outputs_encoded_np = outputs_encoded[0].mean(dim = (-1,-2,-3)).cpu().numpy()

                    loss1_list = []
                    loss1_list.append(loss_func_l1(outputs[:,0,...], ct[:,0,...]))
                    loss1_list.append(loss_cls(outputs[:,1,...], ct[:,1,...]))
                    
                    loss1 = loss1_list[0] + loss1_list[1]
                    mean_loss_tst.append(loss1.item())

                    tst_pred_score.append(torch.sigmoid(outputs_reint.reshape(-1)).detach().cpu().numpy())
                    tst_gt.append(reint_lb.detach().cpu().numpy())
                    pat_id_list.append(str(pat_id[0].item()).zfill(4))

                    outputs_reint_pred = outputs_reint > 0
                    outputs_reint_pred_acc = (outputs_reint_pred.reshape(-1) == reint_lb)

                    loss_reint = loss_cls(outputs_reint.reshape(-1), reint_lb)

                    mean_acc_tst.append(outputs_reint_pred_acc.item())
                    mean_cls_tst.append(loss_reint.item())


                    if step % 5 == 0:
                        
                        print('epoch_tst:, step:{}, loss4:{:.3f}, dice 1:{:.3f}, dice 2:{:.3f}, dice ttl:{:.3f}, time:{:.3f} min'
                            .format(step, 
                            loss1.item(), 
                            2 * large_inter_0_tst / (large_union_0_tst + large_inter_0_tst + 0.00001), 
                            2 * small_inter_0_tst / (small_union_0_tst + small_inter_0_tst + 0.00001),  
                            2 * (large_inter_0_tst + small_inter_0_tst) / (large_union_0_tst + large_inter_0_tst + small_union_0_tst + small_inter_0_tst + 0.00001),  
                            (time() - start) / 60))

            mean_loss_tst = sum(mean_loss_tst) / len(mean_loss_tst)
            mean_acc_tst = sum(mean_acc_tst) / len(mean_acc_tst)
            mean_cls_tst = sum(mean_cls_tst) / len(mean_cls_tst)
            
            pred_lb = (np.array(tst_pred_score) > 0.5).astype(int)

            inv_tst_gt = [1-tst_gt_itr for tst_gt_itr in tst_gt]


            roc_auc_tstue = roc_auc_score(tst_gt, tst_pred_score)
            prec_avg_tstue = average_precision_score(tst_gt, tst_pred_score)
            f1_avg_value = f1_score(tst_gt, pred_lb)
        



            with open(os.path.join(pred_out_dir, "pred_performance_{}.csv".format(split_itr)), 'w') as f:
                f.write("Accuracy {}\n".format(mean_acc_tst))
                f.write("ROC {}\n".format(roc_auc_tstue))        
                f.write("AP {}\n".format(prec_avg_tstue))
            
            with open(os.path.join(pred_out_dir, "pred_values_{}.csv".format(split_itr)), 'w') as f:
                f.write("CV_NUM,pat_id,prd_val,gt\n")
                for item_itr in range(len(pat_id_list)):
                    f.write("{},{},{},{}\n".format(cv_itr,pat_id_list[item_itr], tst_pred_score[item_itr][0], tst_gt[item_itr][0] ))

            print("CV{} SPLIT:{} MODEL:{} epoch_tst mean {}".format(cv_itr, split_itr, model_load_itr, mean_loss_tst))
            print("CV{} SPLIT:{} MODEL:{} epoch_tst mean acc {}".format(cv_itr, split_itr, model_load_itr, mean_acc_tst), flush = True)
            print("CV{} SPLIT:{} MODEL:{} epoch_tst mean cls {}".format(cv_itr, split_itr, model_load_itr, mean_cls_tst), flush = True)
            print("CV{} SPLIT:{} MODEL:{} epoch_tst roc auc {}".format(cv_itr, split_itr, model_load_itr, roc_auc_tstue), flush = True)
            print("CV{} SPLIT:{} MODEL:{} epoch_tst avg pr {}".format(cv_itr, split_itr, model_load_itr, prec_avg_tstue), flush = True)
            
            if split_itr == "TEST":
                AVG_DICT[model_load_itr]['AUC_AVG_LIST'].append(roc_auc_tstue)
                AVG_DICT[model_load_itr]['PRAVG_LIST'].append(prec_avg_tstue)
                AVG_DICT[model_load_itr]['F1_LIST'].append(f1_avg_value)
                AVG_DICT[model_load_itr]['ACC_LIST'].append(mean_acc_tst)


with open(os.path.join(para_experiment_orig, "ovrl_performance_{}.csv".format(split_itr)), 'w') as f:

    f.write("{},AUC,net_best_cls,avg,{}\n".format(para_experiment_orig, sum(AVG_DICT['net_best_cls.pth']['AUC_AVG_LIST'])/len(AVG_DICT['net_best_cls.pth']['AUC_AVG_LIST'])))
    f.write("{},PR,net_best_cls,avg,{}\n".format(para_experiment_orig,  sum(AVG_DICT['net_best_cls.pth']['PRAVG_LIST'])/len(AVG_DICT['net_best_cls.pth']['PRAVG_LIST'])))
    for cv_itr_tst in range(len(AVG_DICT['net_best_cls.pth']['AUC_AVG_LIST'])):
        f.write("{},AUC,net_best_cls,{},{}\n".format(para_experiment_orig, cv_itr_tst,AVG_DICT['net_best_cls.pth']['AUC_AVG_LIST'][cv_itr_tst]))
        f.write("{},PR,net_best_cls,{},{}\n".format(para_experiment_orig, cv_itr_tst,AVG_DICT['net_best_cls.pth']['PRAVG_LIST'][cv_itr_tst]))
        

print("AVERAGE CV AUC net_best_cls.pth", sum(AVG_DICT['net_best_cls.pth']['AUC_AVG_LIST'])/len(AVG_DICT['net_best_cls.pth']['AUC_AVG_LIST']))
print("AVERAGE CV PR net_best_cls.pth", sum(AVG_DICT['net_best_cls.pth']['PRAVG_LIST'])/len(AVG_DICT['net_best_cls.pth']['PRAVG_LIST']))
print("AVERAGE CV F1 net_best_cls.pth", sum(AVG_DICT['net_best_cls.pth']['F1_LIST'])/len(AVG_DICT['net_best_cls.pth']['F1_LIST']))
print("AVERAGE CV ACC net_best_cls.pth", sum(AVG_DICT['net_best_cls.pth']['ACC_LIST'])/len(AVG_DICT['net_best_cls.pth']['ACC_LIST']))

print("AVERAGE CV AUC net_best_cls.pth", AVG_DICT['net_best_cls.pth']['AUC_AVG_LIST'])
print("AVERAGE CV PR net_best_cls.pth", AVG_DICT['net_best_cls.pth']['PRAVG_LIST'])
print("AVERAGE CV F1 net_best_cls.pth", AVG_DICT['net_best_cls.pth']['F1_LIST'])
print("AVERAGE CV ACC net_best_cls.pth", AVG_DICT['net_best_cls.pth']['ACC_LIST'])
print("-"*10)
