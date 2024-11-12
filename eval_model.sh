python eval.py \
    --test_dataset="NOCS(ROOT='../RGBGenPose/data', resolution=224, source='Real', mode='test', all_objects=True)" \
    --model="RayCroCoNet(pos_embed='RoPE100', img_size=(224, 224), head_type='linear', output_mode='pts3d', depth_mode=('exp', -inf, inf), conf_mode=None, enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12)" \
    --pretrained="checkpoints/pos3r/checkpoint-best.pth" \
    --batch_size=6 --num_workers=16 \
    --output_dir="checkpoints/pos3r"