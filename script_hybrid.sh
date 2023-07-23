










export CUDA_VISIBLE_DEVICES=4
{ nohup python3 run_HYBRID.py --gpu=4 --input_type=2 --output_type=2 --glb_ft='original_shape_Maximum3DDiameter,original_shape_Maximum2DDiameterSlice,original_firstorder_Maximum,original_glcm_Idn' --split_seed=0 --Ybxseg --fc_layer_num=2 --slice_len=96 --epoch=25 --w_cls=1 --w_rec=1 --w_recRad=0 --experiment=experiment/hybrid_s0 --initweight=pretrained_/CV0/net_best_cls.pth > logs/hybrid_s0.log 2>&1 ; \
nohup python3 run_HYBRID.py --gpu=4 --input_type=2 --output_type=2 --glb_ft='original_shape_Maximum3DDiameter,original_shape_Maximum2DDiameterSlice,original_firstorder_Maximum,original_glcm_Idn' --split_seed=0 --Ybxseg --fc_layer_num=2 --slice_len=96 --epoch=25 --w_cls=1 --w_rec=1 --w_recRad=0 --experiment=experiment/hybrid_s4 --initweight=pretrained_/CV0/net_best_cls.pth > logs/hybrid_s4.log 2>&1 ; \
nohup python3 run_HYBRID.py --gpu=4 --input_type=2 --output_type=2 --glb_ft='original_shape_Maximum3DDiameter,original_shape_Maximum2DDiameterSlice,original_firstorder_Maximum,original_glcm_Idn' --split_seed=0 --Ybxseg --fc_layer_num=2 --slice_len=96 --epoch=25 --w_cls=1 --w_rec=1 --w_recRad=0 --experiment=experiment/hybrid_s8 --initweight=pretrained_/CV0/net_best_cls.pth > logs/hybrid_s8.log 2>&1 ; \
nohup python3 run_HYBRID.py --gpu=4 --input_type=2 --output_type=2 --glb_ft='original_shape_Maximum3DDiameter,original_shape_Maximum2DDiameterSlice,original_firstorder_Maximum,original_glcm_Idn' --split_seed=0 --Ybxseg --fc_layer_num=2 --slice_len=96 --epoch=25 --w_cls=1 --w_rec=1 --w_recRad=0 --experiment=experiment/hybrid_s12 --initweight=pretrained_/CV0/net_best_cls.pth > logs/hybrid_s12.log 2>&1 ;} &
