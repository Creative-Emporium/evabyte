import argparse
import torch
from transformers import AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset, load_dataset # Import datasets for potential data handling
from typing import Dict, List, Optional, Tuple, Union

from configuration_evabyte import EvaByteConfig
from modeling_evabyte import EvaByteForCausalLM, EvaByteModel
from tokenization_evabyte import EvaByteTokenizer
from image_processing_evabyte import EvaByteImageProcessor
from eva_cache import EvaCache, EvaStaticCacheForTriton
from eva import EvaAttention
from eva_agg_kernel import triton_eva_agg_fwd
from eva_prep_kv_kernel import triton_eva_prep_kv_fwd
from transformers.modeling_outputs import (
    BaseModelOutputWithPast, 
    CausalLMOutputWithPast,
)

# --- Define Custom Multimodal Model (Adapt EvaByteForCausalLM) ---
class MultimodalEvaByteForCausalLM(EvaByteForCausalLM):
    def __init__(self, config: EvaByteConfig):
        super().__init__(config)
        self.visual_embedding = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.audio_embedding = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.video_embedding = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.multimodal_fusion = torch.nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        visual_inputs: Optional[torch.FloatTensor] = None,
        audio_inputs: Optional[torch.FloatTensor] = None,
        video_inputs: Optional[torch.FloatTensor] = None,
        return_all_pred_logits: Optional[bool] = None,
        multibyte_decoding: Optional[bool] = None,
        token_type_ids: Optional[torch.LongTensor] = None,  # Add token_type_ids here
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        # --- Handle Multimodal Inputs ---
        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)

        if visual_inputs is not None:
            visual_embeds = self.visual_embedding(visual_inputs)
            inputs_embeds = self.multimodal_fusion(torch.cat((inputs_embeds, visual_embeds), dim=-1))

        if audio_inputs is not None:
            audio_embeds = self.audio_embedding(audio_inputs)
            inputs_embeds = self.multimodal_fusion(torch.cat((inputs_embeds, audio_embeds), dim=-1))

        if video_inputs is not None:
            video_embeds = self.video_embedding(video_inputs)
            inputs_embeds = self.multimodal_fusion(torch.cat((inputs_embeds, video_embeds), dim=-1))

        # --- Call the base model's forward method ---
        return super().forward(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            return_all_pred_logits=return_all_pred_logits,
            multibyte_decoding=multibyte_decoding,
        )

# --- Data Loading and Preprocessing (Placeholders - Customize based on data format) ---
def load_data(data_path, modality):
    print(f"Loading {modality} data from: {data_path}")
    # Replace with actual data loading logic for different modalities
    if modality == "text":
        # Example: Load text from files, datasets library, etc.
        dataset = load_dataset("text", data_files={"train": data_path}) # Example using datasets lib
        return dataset['train']
    elif modality == "visual":
        # Example: Load image paths, image features, etc.
        dataset = Dataset.from_dict({"image_paths": [data_path]}) # Placeholder
        return dataset
    elif modality == "audio":
        dataset = Dataset.from_dict({"audio_paths": [data_path]}) # Placeholder
        return dataset
    elif modality == "video":
        dataset = Dataset.from_dict({"video_paths": [data_path]}) # Placeholder
        return dataset
    else:
        raise ValueError(f"Unsupported modality: {modality}")

def load_data(data_path, modality, eval_data_path=None):
    print(f"Loading {modality} data from: {data_path}")
    if modality == "text":
        # Load the "wikitext" dataset
        train_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split='train')
        small_train_dataset = train_dataset.select(range(len(train_dataset) // 20)) # 5% for training

        print(f"Using a subset of Wikitext training set with {len(small_train_dataset)} examples.")

        print("Sample of dataset *immediately after loading* (small_train_dataset[0]):", small_train_dataset[0]) # Debug print

        eval_dataset = None # Initialize eval_dataset to None
        if eval_data_path is not None: # Check if eval_data_path is provided
            eval_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split='validation') # Load validation split for evaluation
            small_eval_dataset = eval_dataset.select(range(len(eval_dataset) // 20)) # Take a small subset for eval too, if desired
            print(f"Using a subset of Wikitext validation set for evaluation with {len(small_eval_dataset)} examples.")
            return small_train_dataset, small_eval_dataset # Return both train and eval datasets
        
        return small_train_dataset, eval_dataset
        
def preprocess_data(dataset, modality, tokenizer=None, image_processor=None):
    print(f"Preprocessing {modality} data...")
    # Replace with actual preprocessing logic for different modalities
    if modality == "text":
        def tokenize_function(examples):
            return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128) # Example
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        return tokenized_dataset
    elif modality == "visual":
        def process_images(examples):
            # Placeholder: Load and process images using image_processor
            # For now, just return dummy tensors
            dummy_visual_features = torch.randn(len(examples["image_paths"]), 4096) # Example feature size
            return {"visual_features": dummy_visual_features}
        processed_dataset = dataset.map(process_images, batched=True)
        return processed_dataset
    elif modality == "audio":
         def process_audio(examples):
            # Placeholder: Load and process audio
            dummy_audio_features = torch.randn(len(examples["audio_paths"]), 2048) # Example feature size
            return {"audio_features": dummy_audio_features}
         processed_dataset = dataset.map(process_audio, batched=True)
         return processed_dataset
    elif modality == "video":
         def process_video(examples):
            # Placeholder: Load and process video
            dummy_video_features = torch.randn(len(examples["video_paths"]), 8192) # Example feature size
            return {"video_features": dummy_video_features}
         processed_dataset = dataset.map(process_video, batched=True)
         return processed_dataset
    else:
        raise ValueError(f"Unsupported modality: {modality}")


# --- Training Function (Placeholder - Customize based on training type) ---
def train_model(model, train_dataset, eval_dataset, training_args, modality):

    class MultimodalTrainer(Trainer): # Custom Trainer if needed
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None): # Modified definition
            # Customize loss computation if needed for multimodal inputs/outputs
            labels = inputs.get("labels")
            outputs = model(**inputs) # Pass all relevant inputs to forward
            loss = outputs.loss
            return (loss, outputs) if return_outputs else loss

    trainer = MultimodalTrainer( # Use custom or standard Trainer
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        # data_collator=data_collator # Custom data collator if needed for multimodal data
    )

    print(f"Starting training for modality: {modality}")
    trainer.train()
    print(f"Finished training for modality: {modality}")
    return trainer


def main():
    parser = argparse.ArgumentParser(description="EvaByte Multimodal Training Script")
    parser.add_argument("--modality", type=str, required=True, choices=["text", "visual", "audio", "video", "text-visual", "audio-video", "text-audio", "visual-audio", "visual-video", "multimodal"], help="Training modality")
    parser.add_argument("--train_data_path", type=str, required=True, help="Path to training data (text file, image directory, etc.)")
    parser.add_argument("--eval_data_path", type=str, default=None, help="Optional path to evaluation data")
    parser.add_argument("--model_name_or_path", type=str, default="evabyte/EvaByte-SFT", help="Pretrained model name or path")
    parser.add_argument("--output_dir", type=str, default="./multimodal_evabyte_output", help="Directory to save training outputs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4, help="Batch size per device during training")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4, help="Batch size per device during evaluation")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--logging_steps", type=int, default=50, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")
    parser.add_argument("--evaluation_strategy", type=str, default="steps", help="Evaluation strategy to adopt during training", choices=["no", "steps", "epoch"])
    parser.add_argument("--eval_steps", type=int, default=100, help="Run evaluation every X steps when evaluation_strategy is set to 'steps'")

    args = parser.parse_args()

    # --- Load Tokenizer and Image Processor ---
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True) # Or use EvaByteTokenizer.from_pretrained(...) if you have custom tokenizer config
    image_processor = EvaByteImageProcessor.from_pretrained(args.model_name_or_path, trust_remote_code=True) # Or EvaByteImageProcessor(...) with custom config

    # --- Load Model Configuration ---
    config = EvaByteConfig.from_pretrained(args.model_name_or_path, trust_remote_code=True) # Or EvaByteConfig(...) with custom parameters
    # --- Instantiate Multimodal Model ---
    model = MultimodalEvaByteForCausalLM.from_pretrained(args.model_name_or_path, config=config, torch_dtype=torch.bfloat16, trust_remote_code=True).to("cpu") # Or MultimodalEvaByteForCausalLM(config).to("cuda")

    # --- Load and Preprocess Data ---
    train_dataset, eval_dataset = load_data(args.train_data_path, args.modality, args.eval_data_path) # Load eval dataset too
    train_dataset = preprocess_data(train_dataset, args.modality, tokenizer, image_processor)
    print("Sample of train_dataset after preprocessing:", train_dataset[0]) # Print the first example
    if eval_dataset is not None: # Only preprocess if eval_dataset is loaded
        eval_dataset = preprocess_data(eval_dataset, args.modality, tokenizer, image_processor)


    # --- Set up Training Arguments ---
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        evaluation_strategy=args.evaluation_strategy, # Keep evaluation strategy
        eval_steps=args.eval_steps,
        learning_rate=5e-5, 
        weight_decay=0.01,    
        warmup_steps=500,     
        fp16=False,           
        bf16=True,           
        gradient_checkpointing=False, 
        dataloader_num_workers=4,    
        save_total_limit=2,
        report_to=[],       
        remove_unused_columns=False
    )

    # --- Train the Model ---
    trainer = train_model(model, train_dataset, eval_dataset, training_args, args.modality) # Pass eval_dataset to train_model

    # --- Save Trained Model and Processor ---
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    image_processor.save_pretrained(args.output_dir)

    print(f"Training completed and model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
