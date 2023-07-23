











export CUDA_VISIBLE_DEVICES=0
{ nohup python3 run_RIDL.py --Ypredinv --gpu=0 --input_type_0=2 --input_type_1=4 --glb_ft="original_shape_Maximum3DDiameter,original_shape_Maximum2DDiameterSlice,original_firstorder_Maximum,original_glcm_Idn" --split_seed=0 --Ybxseg --fc_layer_num=2 --slice_len=96 --epoch=25 --w_cls=1 --w_rec=1 --w_recRad=0 --w_corr=2 --w_exp=0 --initweight=pretrained/CV0/net_best_cls.pth --experiment=experiment/ridl_s1 > logs/ridl_s1.log 2>&1 ; \
nohup python3 run_RIDL.py --Ypredinv --gpu=0 --input_type_0=2 --input_type_1=4 --glb_ft="original_shape_Maximum3DDiameter,original_shape_Maximum2DDiameterSlice,original_firstorder_Maximum,original_glcm_Idn" --split_seed=0 --Ybxseg --fc_layer_num=2 --slice_len=96 --epoch=25 --w_cls=1 --w_rec=1 --w_recRad=0 --w_corr=2 --w_exp=0 --initweight=pretrained/CV0/net_best_cls.pth --experiment=experiment/ridl_s2 > logs/ridl_s2.log 2>&1 ; \
nohup python3 run_RIDL.py --Ypredinv --gpu=0 --input_type_0=2 --input_type_1=4 --glb_ft="original_shape_Maximum3DDiameter,original_shape_Maximum2DDiameterSlice,original_firstorder_Maximum,original_glcm_Idn" --split_seed=0 --Ybxseg --fc_layer_num=2 --slice_len=96 --epoch=25 --w_cls=1 --w_rec=1 --w_recRad=0 --w_corr=2 --w_exp=0 --initweight=pretrained/CV0/net_best_cls.pth --experiment=experiment/ridl_s3 > logs/ridl_s3.log 2>&1 ;} &

