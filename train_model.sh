python train.py \
    --train_dataset="Omni6DPose(ROOT='../RGBGenPose/data/SOPE', resolution=224)" \
    --test_dataset="Omni6DPose(ROOT='../RGBGenPose/data/SOPE', resolution=224, mode='test')" \
    --train_criterion="Pos3rCriterion()" \
    --test_criterion="Regr3D(L21)" \
    --model="RayCroCoNet(pos_embed='RoPE100', img_size=(224, 224), head_type='linear', output_mode='pts3d', depth_mode=('exp', -inf, inf), conf_mode=None, enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12)" \
    --pretrained="checkpoints/CroCo_V2_ViTLarge_BaseDecoder.pth" \
    --lr=0.0001 --min_lr=1e-06 --warmup_epochs=1 --epochs=100 --batch_size=6 --accum_iter=1 \
    --save_freq=1 --keep_freq=1 --eval_freq=1 --print_freq=100 --num_workers=16 \
    --output_dir="checkpoints/pos3r"