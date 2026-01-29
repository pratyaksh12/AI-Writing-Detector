import os, json, time;
from typing import List, Dict, Optional;
from dotenv import load_dotenv;
from openai import OpenAI;


load_dotenv()
api_key: Optional[str] = os.getenv("OPEN_AI_KEY")

if(not api_key):
    api_key =  os.getenv("OPEN_API_KEY")
    
if(not api_key):
    raise ValueError("Missing API Key please set API key in your .env file")

CLIENT = OpenAI(api_key=api_key)

INPUT_FILE: str = "AI-Writing-Detector/scraping/human_text_pre2022.jsonl"
OUTPUT_FILE: str = "ai_generated_text.jsonl"
MODEL: str = "gpt-5-nano" #cheapest model right now

def generate_rewrite(text: str) -> Optional[str]:
    
    try:
        response_1 = CLIENT.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a professional summarizer who summarizes long text to small paragraphs of no more than 30 words without losing the original paragraphs meaning"},
                {"role": "user", "content":"Summarize this paragraph for me please make sure to retain important context and meaning. Make it like 2 sentence max. The response should only consist of the summarized text and nothing else."}
            ]
        )
        
        summary: Optional[str] = response_1.choices[0].message.content
        
        if not summary: return None
        
        response_2 = CLIENT.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content":"You are a professional writer and you are to write a paragraph out of a summarized text. Make sure to retain the context and message that is being conveyed."},
                {"role":"user", "content":"Plase expand this summarized text for me please. The response should only consist of the paragraph and nothing else."}
            ]
        )
        
        ai_text: Optional[str] = response_2.choices[0].message.content
        return ai_text
    
    except Exception as ex:
        print(f"Error generating the text: {ex}")
        
        
        return None
def main():
    print("Reading input files")
    
    if not os.path.exists(INPUT_FILE):
        print("INPUT FILE PATH Doesn't exist")
        return
    
    total_lines: int = sum(1 for _ in open(INPUT_FILE, "r", encoding="utf-8"))
    processed: int = 0
    
    
    with open(INPUT_FILE, "r", encoding="utf-8") as fin, open(OUTPUT_FILE, "w", encoding="utf-8") as fout:
        
        for line in fin:
            processed += 1
            record: Dict[str, str] = json.loads(line)
            human_text: str = record["text"]
            
            ai_text = generate_rewrite(human_text)
            
            if(ai_text):
                new_record: Dict[str, str] = {
                    "human_text": human_text,
                    "ai_text": ai_text,
                    "source": record["page"],
                    "model": MODEL
                }
                
                fout.write(json.dumps(new_record) + "\n")
                print(f"[{processed + 1}/{total_lines}] AI text generated")
                
            else:
                print(f"[{processed + 1}/{total_lines}] Failed to generate AI text for {record["page"]}")
                
                
            time.sleep(0.5)
            
            
if __name__ == "__main__":
    main()
    
        