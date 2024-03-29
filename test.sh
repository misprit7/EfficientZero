set -ex
export CUDA_DEVICE_ORDER='PCI_BUS_ID'
export CUDA_VISIBLE_DEVICES=0

python main.py --env BreakoutNoFrameskip-v4 --case atari --opr test --seed 0 --num_gpus 1 --num_cpus 20 --force \
  --test_episodes 32 \
  --load_model \
  --amp_type 'torch_amp' \
  --model_path 'model.p' \
  --info 'Test'

# python main.py --env minetester-treechop_shaped-v0 --case minetest --opr train --amp_type torch_amp --num_gpus 1 --num_cpus 10 --cpu_actor 8 --gpu_actor 4 --force --object_store_memory 16000000000

