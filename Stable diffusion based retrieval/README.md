# Stable diffusion

Can stable diffusion be used to retrieve images based on text? Let's find it out.


LoRA: Low-Rank Adaptation of Large Language Models

# Installing
Please run the following lines of code: 

    git clone https://github.com/huggingface/diffusers
    cd diffusers
    pip install .
    pip install -r ../requirements.txt


# Runing the code

We decided to do a finetuning of the model "**CompVis/stable-diffusion-v1-4**" with the dataset "**lambdalabs/pokemon-blip-captions**" both provided by huggingface. 
Note that this step is ***Optional*** as stable diffusion models already work well with a great amount of datasets.

For this matter we have to set the accelerate and huggingface-cli libraries, create a wandb (if wanted) for training tracking and run the training with the code provided.

Setting up the environment: 

    accelerate config
    huggingface-cli login

Run in cmd: 
    
    export MODEL_NAME="CompVis/stable-diffusion-v1-4"
    export DATASET_NAME="lambdalabs/pokemon-blip-captions"
    
    accelerate launch --mixed_precision="fp16"  train_text_to_image.py \
      --pretrained_model_name_or_path=$MODEL_NAME \
      --dataset_name=$DATASET_NAME \
      --use_ema \
      --resolution=512 --center_crop --random_flip \
      --train_batch_size=1 \
      --gradient_accumulation_steps=4 \
      --gradient_checkpointing \
      --max_train_steps=15000 \
      --learning_rate=1e-05 \
      --max_grad_norm=1 \
      --lr_scheduler="constant" --lr_warmup_steps=0 \
      --output_dir="sd-pokemon-model" 

After this a model will be created in ```output_dir``` which we will use for inference. 


### Brief list of main used libraries
- [Transformers](https://huggingface.co/docs/transformers/index)
- [Datasets](https://huggingface.co/docs/datasets/index) -> We used the labeled Pokémon dataset: [lambdalabs/pokemon-blip-captions](https://huggingface.co/datasets/lambdalabs/pokemon-blip-captions)
- [Accelerate](https://huggingface.co/docs/accelerate/index)
- [Torch](https://pytorch.org/)

