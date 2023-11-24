# Stable diffusion

Can stable diffusion be used to retrieve images based on text? Let's find it out.


LoRA: Low-Rank Adaptation of Large Language Models

# Installing
Please run the following lines of code: 

    git clone https://github.com/huggingface/diffusers
    cd diffusers
    pip install .
    pip install -r ../requirements.txt


### Brief list of main used libraries
- [Transformers](https://huggingface.co/docs/transformers/index)
- [Datasets](https://huggingface.co/docs/datasets/index) -> We used the labeled Pokémon dataset: [lambdalabs/pokemon-blip-captions](https://huggingface.co/datasets/lambdalabs/pokemon-blip-captions)
- [Accelerate](https://huggingface.co/docs/accelerate/index)
- [Torch](https://pytorch.org/)

