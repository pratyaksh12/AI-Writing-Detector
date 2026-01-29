import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
import os
import logging

logging.getLogger("transformers").setLevel(logging.ERROR)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = "results/final_model"
BASE_MODEL_NAME = "bert-base-cased"

def predict(text, model, tokenizer):

    inputs = tokenizer(
        text, 
        return_tensors="pt", 
        truncation=True, 
        padding=True, 
        max_length=512
    )
    
    # Move validation input to the same device as model
    device = model.device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    
    # Calculate probabilities
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    
    # Get predicted class
    predicted_class_id = logits.argmax().item()
    confidence = probabilities[0][predicted_class_id].item()
    
    # Map ID to label
    labels = {0: "Human", 1: "AI"}
    predicted_label = labels.get(predicted_class_id, "Unknown")
    
    return predicted_label, confidence, probabilities

def main():
    print(f"Loading model from {MODEL_PATH}...")
    
    # Check if model path exists
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model path not found at {MODEL_PATH}")
        print("Please ensure you have run the training script first.")
        return

    try:
        # Load Tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        except:
            print("Local tokenizer not found, downloading base tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)

        # Load Base Model
        base_model = AutoModelForSequenceClassification.from_pretrained(
            BASE_MODEL_NAME, 
            num_labels=2,
            id2label={0: "Human", 1: "AI"},
            label2id={"Human": 0, "AI": 1}
        )
        
        # Load Adapters (LoRA)
        model = PeftModel.from_pretrained(base_model, MODEL_PATH)
        
        # Set device (MPS if available, else CPU)
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            print("Using MPS (Metal Performance Shaders) acceleration.")
        else:
            device = torch.device("cpu")
            print("Using CPU.")
            
        model.to(device)
        model.eval()
        
        print("\n" + "="*50)
        print("MODEL LOADED SUCCESSFULLY")
        print("="*50)
        print("Enter text to classify (type 'quit' or 'exit' to stop):")
        
        while True:
            user_input = input("\nText > ")
            
            if user_input.lower() in ['quit', 'exit']:
                print("Exiting...")
                break
                
            if not user_input.strip():
                continue
                
            label, score, probabilities = predict(user_input, model, tokenizer)
            
            # Color encoding for terminal output
            color = "\033[92m" if label == "Human" else "\033[91m" # Green for Human, Red for AI
            reset = "\033[0m"
            
            print(f"Prediction: {color}{label}{reset}")
            print(f"Confidence: {score:.2%}")
            print(f"Raw Probabilities -> Human: {probabilities[0][0].item():.4f}, AI: {probabilities[0][1].item():.4f}")

    except Exception as e:
        print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    main()
