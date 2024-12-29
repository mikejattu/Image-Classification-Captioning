from functools import partial

import torch
from PIL import Image
from torch.utils.data import Dataset
from nltk.translate.bleu_score import corpus_bleu
from transformers import Seq2SeqTrainer
from torchvision import transforms

# ##########
# TODO: Add more imports
from transformers import default_data_collator, \
                        ViTImageProcessor, \
                        VisionEncoderDecoderModel, \
                        GPT2Tokenizer, \
                        Seq2SeqTrainingArguments,\
                        DataCollatorForSeq2Seq,\
                        AutoTokenizer
# ##########

class Args:
    """Configuration.
    """
    # Encoder-Decoder for captioning
    encoder = 'google/vit-base-patch16-224'
    decoder = 'gpt2'

    # Dataset path
    root_dir = "../flickr8k"

    # Save your model as "cap-vlm-{YOUR_CCID}"
    YOUR_CCID = "sjattu"
    name = f"cap-vlm-{YOUR_CCID}"

    # Hyperparameters
    batch_size = 4
    lr = 5e-5
    epochs = 1

    # Generation cfgs
    # TODO: Add more as you see fit
    num_beams = 5
    max_length = 50     # TODO: Can play around


    # Train ops
    # TODO: Add more as you see fit
    logging_steps = 50
def load_text(filename):
    # open the file as read only
    file = open(filename, 'r')
    # read all text
    text = file.read()
    text = text.split('\n')
    # close the file
    file.close()
    return text
def get_images_paths(root_dir):
    # load image paths
    text = load_text(f"{root_dir}")
    img_paths = []
    for i in range(len(text)):
        if i == 0:
            continue
        if text[i] == '':
            continue
        image = text[i].split(';')[0]
        img_paths.append(image)
    return img_paths
def get_captions(root_dir):
    # load captions
    text = load_text(f"{root_dir}")
    captions = []
    for i in range(len(text)):
        if i == 0:
            continue
        if text[i] == '':
            continue
        caption = text[i].split(';')
        caption = caption[1]
        captions.append(caption)
    return captions
class FlickrDataset(Dataset):
    def __init__(
        self, 
        args, 
        processor, 
        tokenizer,
        mode: str = "train",
        ):
        assert mode in ["train", "val", "test"]
        self.args = args
        # ####################
        # TODO: Load Flickr8k dataset
        # TODO: Initialize vision encoder's processor
        # TODO: Initialize langauge decoder's tokenizer

        # getting the image paths with respect to the mode
        if processor is None:
            processor = ViTImageProcessor.from_pretrained(args.encoder)

        if tokenizer is None:
            tokenizer = GPT2Tokenizer.from_pretrained(args.decoder)
            #adding the special tokens
            tokenizer.add_special_tokens({
                "bos_token": "<|beginoftext|>",
                "pad_token": "<|pad|>",
                "eos_token": "<|endoftext|>"
            })
        self.processor = processor
        self.tokenizer = tokenizer

        self.img_paths, self.captions = None, None

        if mode == "train":
            path = f"{args.root_dir}/train.txt"
            self.img_paths = get_images_paths(path)
            self.captions = get_captions(path)
        elif mode == "val":
            path = f"{args.root_dir}/val.txt"
            self.img_paths = get_images_paths(path)
            self.captions = get_captions(path)
        # ####################

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
    # ####################
            # TODO: Load and process image-caption data
        args = self.args
        begin_token = "<|beginoftext|> "
        end_token = " <|endoftext|>"
        image = Image.open(args.root_dir + "/images/"+self.img_paths[idx]).convert("RGB")
        # transform the image
        image = transforms.Resize((224, 224))(image)
        pixel_values = self.processor(images=image, 
                                        return_tensors="pt").pixel_values.squeeze(0) # Preprocess image

        labels = self.tokenizer(begin_token + self.captions[idx] + end_token, 
                                padding="max_length",
                                max_length = args.max_length,
                                return_tensors="pt", 
                                truncation = True).input_ids.squeeze(0) # Tokenize caption
        encoding = {
            "pixel_values": pixel_values,      
            "labels": labels,             
            "path": args.root_dir + "/images/"+self.img_paths[idx],
            "captions": self.captions[idx],
        }
        # ####################

        return encoding

    
def train_cap_model(args):
    # Define your vision processor and language tokenizer
    # Modify the embedding lookup table in decoder model and the tokenizer
    # to include bos_token "<|beginoftext|>" and pad_token "<|pad|>"
    processor = ViTImageProcessor.from_pretrained(args.encoder)
    tokenizer = GPT2Tokenizer.from_pretrained(args.decoder)
    tokenizer.add_special_tokens({
    "bos_token": "<|beginoftext|>",
    "pad_token": "<|pad|>",
    "eos_token": "<|endoftext|>"
    })
    # Define your Image Captioning model using Vision-Encoder-Decoder model 
    # Reference: https://huggingface.co/docs/transformers/en/model_doc/vision-encoder-decoder
    model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(args.encoder, args.decoder)
    model.decoder.resize_token_embeddings(len(tokenizer))

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")

    model = model.to(device)    # NOTE: Send your model to GPU


    # NOTE: The format of GPT2 inputs:
    # <|endoftext|> + article + " TL;DR: " + summary + <|endoftext|>
    # For captoning, we want:
    # <|beginoftext|> + caption + <|endoftext|>
    # followed by a number of paddings "<|pad|>"



    # Load train/val dataset
    train_dataset = FlickrDataset(args, mode = "train", tokenizer=tokenizer, processor=processor)
    val_dataset = FlickrDataset(args, mode = "val", tokenizer=tokenizer, processor=processor)


    # Model configuration. 
    # Reference: https://huggingface.co/docs/transformers/en/model_doc/vision-encoder-decoder
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.decoder_start_token_id = tokenizer.bos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.generation_config.bos_token_id = tokenizer.bos_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    # TODO: Play around with some generation config parameters
    # e.g. For beam search, you can potentially have a larger beam size of 5
    # Add more as you see fit
    model.generation_config.max_length = args.max_length #None
    model.generation_config.num_beams = args.num_beams #None

    # TODO: Define training arguments for Seq2Seq model (Seq2SeqTrainingArguments)
    # Reference: https://huggingface.co/docs/transformers/en/main_classes/trainer
    

    training_args = Seq2SeqTrainingArguments(output_dir = "./",
                                            do_train= True,
                                            do_eval = True, 
                                            eval_strategy = "epoch", 
                                            eval_steps = args.logging_steps, 
                                            logging_dir = "./logs", 
                                            logging_steps = args.logging_steps, 
                                            save_steps = args.logging_steps, 
                                            per_device_eval_batch_size= args.batch_size, 
                                            per_device_train_batch_size = args.batch_size, 
                                            num_train_epochs = args.epochs, 
                                            learning_rate = args.lr,
                                            save_total_limit= 3,
                                            weight_decay= 1e-4,
                                            predict_with_generate=True)
    
    # Create a data collator for padding input sequences
    # Instantiate seq2seq model trainer
    compute_metrics = partial(compute_bleu_score, tokenizer=tokenizer)
    trainer = Seq2SeqTrainer(
        tokenizer = tokenizer,
        model = model,
        args = training_args,
        train_dataset = train_dataset,
        eval_dataset = val_dataset,
        compute_metrics = compute_metrics,
        data_collator = default_data_collator
    )

    # Start training
    # TODO: A good performing model should easily reach a BLEU score above 0.07
    trainer.train()
    trainer.save_model(args.name)
    

def load_trained_model(
    ckpt_dir: str,
    ):
    """TODO: Load your best trained model, processor and tokenizer.
    """
    
    PATH = ckpt_dir
    args = Args
    # TODO: Load encoder processor
    processor = ViTImageProcessor.from_pretrained(args.encoder)

    # TODO: Load decoder tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(PATH)

    # TODO: Load your best trained model
    model = VisionEncoderDecoderModel.from_pretrained(PATH)
    
    # TODO: Load your model configuration
    #config = model.get_config()

    model.eval()


    if torch.cuda.is_available():
        model = model.cuda()

    return model, processor, tokenizer

def inference(
    img_path,
    model, 
    processor,
    tokenizer,
    ):
    """TODO: Example inference function to predict a caption for an image.
    """
    # TODO: Load and process the image
    image = Image.open(img_path).convert("RGB")
    # TODO: Preproces the image
    img_tensor = processor(images=image, return_tensors="pt")

    # Ensure your img_tensor is on GPU
    if torch.cuda.is_available():
        img_tensor = img_tensor.cuda()


    # TODO: Generate the caption with VisionEncoderDecoderModel's generate API
    generated_ids = model.generate(pixel_values = img_tensor.pixel_values)

    # Tokens -> Str
    print(generated_ids)
    generated_caption = tokenizer.decode(generated_ids.squeeze(0), skip_special_tokens=True)

    return generated_caption

def compute_bleu_score(pred, tokenizer):
    """NOTE: DO NOT CHANGE.
    Compute BLEU score.
    NOTE: if you are interested in learning about the BLEU score, here are some interesting resources:
    https://www.geeksforgeeks.org/nlp-bleu-score-for-evaluating-neural-machine-translation-python/
    https://cloud.google.com/translate/automl/docs/evaluate#interpretation
    https://www.nltk.org/api/nltk.translate.bleu_score.html
    """

    pred_ids = pred.predictions
    labels_ids = pred.label_ids#.squeeze(1)

    # Decode predictions and labels while handling special tokens and padding
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == tokenizer.pad_token_id] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

    # Prepare data for BLEU score calculation
    pred_bleu = [line.split() for line in pred_str]
    label_bleu = [[line.split()] for line in label_str]

    # Calculate BLEU score
    bleu_output = corpus_bleu(label_bleu, pred_bleu)
    bleu_score = round(bleu_output, 4)
    print("BLEU:", bleu_score)

    return {
        "bleu_score": bleu_score
    }
