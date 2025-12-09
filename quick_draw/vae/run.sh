# uv run cvae.py --latent_dim 32 --epochs 150 --lr 0.0005 --batch_size 256 --max_filters 64 --save_dir ./model_weights_latent_32_filters_64
uv run cvae.py --latent_dim 32 --epochs 50 --load_dir ./model_weights_latent_32_filters_64/cvae_final.safetensors --lr 0.00005 --batch_size 256 --max_filters 64 --save_dir ./model_weights_latent_32_filters_64_ft1
