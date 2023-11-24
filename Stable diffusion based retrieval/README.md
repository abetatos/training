# Stable diffusion

Can stable diffusion be used to retrieve images based on text? Let's find it out.

The general pipeline consists of a text to image generative model, specifically a diffuser, which will generate an image given a prompt. 
That image can be used to obtain a similarity score when compared to a dataset, thus obtaining an image retriever based on text.

LoRA: Low-Rank Adaptation of Large Language Models


> ⚠️ **Warning:** This is only for educational purposes, do not use in production, it is slow as a turtle 🐢

# Installing
You only need to install the requirements in order to run the code. 

    pip install -r requirements.txt

We added a ```freeze.txt``` file which contains all the libraries and their versions in case you encounter some version problems.

Once setted the environment we need to set the configuration for ```accelerate``` and ```huggingface-cli```

Setting up the environment: 

    accelerate config
    huggingface-cli login

For tracking we created an account in ```wandb``` but ```tensorboard``` can also be used.

# Runing the code

We decided to do a finetuning of the model "**CompVis/stable-diffusion-v1-4**" with the dataset "**lambdalabs/pokemon-blip-captions**" both provided by huggingface. 
Note that this step is ***Optional*** as stable diffusion models already work well with a great amount of datasets.



Run in cmd: 
    
    export MODEL_NAME="CompVis/stable-diffusion-v1-4"
    export DATASET_NAME="lambdalabs/pokemon-blip-captions"
    
    accelerate launch --mixed_precision="fp16" train_text_to_image_lora.py \
      --pretrained_model_name_or_path=$MODEL_NAME \
      --dataset_name=$DATASET_NAME --caption_column="text" \
      --resolution=512 --random_flip \
      --train_batch_size=1 \
      --num_train_epochs=100 --checkpointing_steps=5000 \
      --learning_rate=1e-04 --lr_scheduler="constant" --lr_warmup_steps=0 \
      --seed=42 \
      --output_dir="sd-pokemon-model-lora" \
      --validation_prompt="cute dragon creature" --report_to="wandb"

After this a model will be created in ```output_dir``` which we will use for inference.