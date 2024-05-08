# Pathology MLLM training code

## Installation

```Shell
conda create -n train_dev python=3.10 -y
conda activate train_dev
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
pip install flash-attn --no-build-isolation
```

Installation should take about 5-10 minutes depending on your internet connection.

## Training

Tested on 64-bit Ubuntu 20.04.3 LTS but any other OS that supports python and CUDA should work as well. In our testing, you would need at least 8 x 80GB GPUs to have sufficient VRAM to train the model (with flash attention 2 + deepspeed zero3 recipe).

You should first request access to the llama weights using the following link:
https://huggingface.co/meta-llama/Llama-2-13b-chat-hf. After you have access, you can download the llama weights and put them in a folder. This is your <LLAMA_PATH> for the following commands.

You also need your own vision encoder + adapter weights (pretrained using CoCa). This is your <VISION_ENCODER_PATH> and <MM_ADAPTER_Path> for the following commands.


### Prepare data
You should prepare the data as a .json file which consists of a list of dictionaries. Each dictionary corresponds to a single training example and should have the following keys: **image** (image filename), **conversations** (a list of question / answer turns with keys **from** and **value**). The **from** key should be either "human" or "assistant" and the **value** key should be the text of the conversation. For the first question asked by "human", the content of "value" should end with a "\n\<image\>" token. Here is an example json file with a single training example.

```json
[
    {
        "id": "0001",
        "image": "train1.jpg",
        "conversations": [
            {
                "from": "human",
                "value": "<QUESTION>\n<image>"
            },
            {
                "from": "assistant",
                "value": "<ANSWER>"
            }
        ]
    }
]
```

We will refer to the path of this .json file as $DATA_PATH for the following commands. You should also have a folder of images with the same filenames as the "image" keys in the .json file. We will refer to the path of this folder as <IMG_PATH> for the following commands.

### Train the model
Training outputs will be saved in <OUTPUT_ROOT>. 

#### Pretrain 
Output of first stage of training (updating just the multimodal adapter) will be saved in <OUTPUT_ROOT>/pretrain.

```shell
deepspeed llava/train/train_mem.py \
    --deepspeed zero3.json \
    --model_name_or_path <LLAMA_PATH> \
    --version plain \
    --data_path <DATA_PATH> \
    --image_folder <IMG_PATH> \
    --vision_tower coca_vit-l \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -1 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --mm_projector_type "attn_pool+mlp2x_gelu" \
    --pretrain_vision_backbone <VISION_ENCODER_PATH> \
    --pretrain_mm_mlp_adapter <MM_ADAPTER_PATH> \
    --image_aspect_ratio pad \
    --bf16 True \
    --output_dir <OUTPUT_ROOT>/pretrain \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_total_limit 1 \
    --learning_rate 2e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to tensorboard 
```

#### Finetune

```shell
deepspeed llava/train/train_mem.py \
    --deepspeed zero3.json \
    --model_name_or_path <LLAMA_PATH>  \
    --version llava_llama_2 \
    --data_path <DATA_PATH> \
    --image_folder <IMG_PATH> \
    --vision_tower coca_vit-l \
    --mm_vision_select_layer -1 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --pretrain_mm_mlp_adapter <OUTPUT_ROOT>/pretrain/mm_projector.bin \
    --pretrain_vision_backbone <VISION_ENCODER_PATH> \
    --mm_projector_type "attn_pool+mlp2x_gelu" \
    --bf16 True \
    --output_dir <OUTPUT_ROOT>/finetune \
    --image_aspect_ratio pad \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to tensorboard 
```

During training, a tensorboard log will be saved in <OUTPUT_ROOT>/finetune/runs, which you can use to monitor the training loss and throughput. The final trained model checkpoint will be saved in <OUTPUT_ROOT>/finetune.

### Try training with provided demo data

We provide an actual example of the data and image folder in the **test_data** folder. You can try training with this data (works for both pretraining and finetuning) by specifying the corresponding paths in the above commands, specifically `--data_path ./test_data/data.json` and `--image_folder ./test_data/imgs`. To simply test training, you can ignore the `--pretrain_vision_backbone` and `--pretrain_mm_mlp_adapter` flags in the pretrain command, which will randomly initialize the vision encoder and multimodal adapter weights. 

The training should take a couple minutes to initialize the model, and then a couple minutes to finish and save the checkpoints once training starts. You should except to see the training loss, throughput, etc. printed to the console, and the trained model checkpoints will be saved in the specified output directory.

### Acknowledgements

This code is adapted from [Llava 1.1.3](https://github.com/haotian-liu/LLaVA/releases/tag/v1.1.3). We thank the original authors for their valuable contributions.

# PathQABench-Public
The images and open-ended questions from PathQABench-Public can be accessed [here](https://drive.google.com/drive/folders/1RM58s6XrzpqyGC0osagt3_UGrj9JZvOj?usp=sharing).

# Terms of use
The code and data should only be used for academic research purposes. 
