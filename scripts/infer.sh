export PYTHONPATH=$PYTHONPATH:$PWD


model_path=src/ckpt/model_200ms.safetensors
vocoder_path=src/ckpt/vocos.pt
bn_path=path/to/bn/
speaker_emb_path=path/to/speaker_emb.npy
prompt_path=path/to/prompt_mel.npy
output_dir=src/outputs

mkdir -p $output_dir

python3 src/infer/infer.py \
--model-config src/config/config_200ms.json \
--ckpt-path ${model_path} \
--vocoder-ckpt-path ${vocoder_path} \
--output-dir ${output_dir} \
--bn-path ${bn_path} \
--spk-emb-path ${speaker_emb_path} \
--prompt-path ${prompt_path} \
--chunk-size 20 \
--steps 2 