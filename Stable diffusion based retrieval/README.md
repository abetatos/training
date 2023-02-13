# Stable diffusion

Can stable diffusion be used to retrive images based on text? Let's find it out fine-tuning one stable diffusion based model with some amazing open source libraries.

There has been a new release of a new way of training Large Language Models (LLMs) called Parameter Efficient Fine Tuning ([PEFT](https://huggingface.co/blog/peft)) which can be used to fine tune a transformers model in low resource equipments. Even in Google Colab! 

The pipeline here is: 

1) Fine-tuning a LLM with the use of accelerate
2) Predicting a completely new image from a text prompt. 
3) Measuring distances with the dataset to obtain which ones are similar and thus, which ones are more similar to the query.


# To run the code
Colab can be used to run the code as I made a jupyter notebook, altough a premium version is needed to give acces to the system shell and run some commands. 

With a GPU it can be run with no problem. 


### Brief list of main used libraries
- [Transformers](https://huggingface.co/docs/transformers/index)
- [Datasets](https://huggingface.co/docs/datasets/index) -> We used the labeled pokemon dataset: [lambdalabs/pokemon-blip-captions](https://huggingface.co/datasets/lambdalabs/pokemon-blip-captions)
- [Accelerate](https://huggingface.co/docs/accelerate/index)
- [Torch](https://pytorch.org/)

