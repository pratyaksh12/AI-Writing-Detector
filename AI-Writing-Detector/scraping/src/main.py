import json, logging, random;
import time
from typing import List, Optional, Set, Dict, Any;
from bs4 import BeautifulSoup

import wikipediaapi as wiki;
import requests;

from random_list import RandomList;

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("run.log"), logging.StreamHandler()]
)

N_DATA_TARGET: int = 100
SCRAPE_FILE: str = "human_text_pre2022.jsonl"
MAX_DATA_PER_PAGE: int = 2
MAX_LINKS_PER_PAGE: int = 2
# Format: YYYYMMDDHHMMSS
CUTOFF_DATE: str = "20220101000000"
SESSION = requests.Session()
# IMPORTANT: Change this email to something unique!
SESSION.headers.update({
    "User-Agent": "Pratyash_AIDetector_Student/1.0 (pratyash_student_project@gmail.com)"
})
API_URL: str = "https://en.wikipedia.org/w/api.php"

USER_AGENT: str = "AI_Detector_Student_Project/1.0 (contact@example.com)"


END_SECTIONS: Set[str] = {
     "See also", "References", "Further reading", "External links", "Notes", 
    "Bibliography", "Works"
}

TOP_PAGES: List[str] = [
    "United States", "Donald Trump", "Elizabeth II", "India", "Barack Obama",
    "Cristiano Ronaldo", "World War II", "United Kingdom", "Michael Jackson", "Elon Musk"
]

def get_pre_2022_revision_id(title: str) -> Optional[int]:
    params: Dict[str, Any] = {
        "action": "query",
        "format": "json",
        "prop": "revisions",
        "titles": title,
        "rvstart": CUTOFF_DATE,
        "rvdir": "older",
        "rvlimit": 1,
        "rvprop": "ids"
    }
    
    try:
        resp = SESSION.get(API_URL, params = params)
        data = resp.json()
        
        pages = data.get("query", {}).get("pages", {})
        for _, page_data in pages.items():
            if "revisions" in page_data:
                return page_data["revisions"][0]["parentid"]
            
    except Exception as ex:
        logging.error(f"failed to get revision for {title}: {ex}")
        
    return None


def get_clean_text_from_revision(rev_id: int) -> List[str]:
    params: Dict[str, Any] = {
        "action": "parse",
        "format": "json",
        "oldid": rev_id,
        "prop": "text",
        "disabletoc": 1,
        "disableditsection": 1
    }
    
    try:
        resp = SESSION.get(API_URL, params=params)
        data =  resp.json()
    
        raw_html = data.get("parse", {}).get("text", {}).get("*", "")
        if not raw_html:
            return []
        
        soup = BeautifulSoup(raw_html, "html.parser")
        
        paragraphs: List[str] = []
        
        for p_tag in soup.find_all("p"):
            for sup in p_tag.find_all("sup"):
                sup.decompose()
                
            text = p_tag.get_text().strip()
            if text:
                paragraphs.append(text)
        
        return paragraphs
            
            
    
    except Exception as ex:
        logging.error(f"failed to parse content for rev {rev_id}: {ex}")
        return []
        
        
def get_linked_articles(title: str) -> List[str]:
    params: Dict[str, Any] = {
        "action": "query",
        "format": "json",
        "titles": title,
        "plnamespace": 0, # Articles only
        "prop": "links",
        "pllimit": 50
    }
    
    try:
        resp = SESSION.get(API_URL, params=params)
        data = resp.json()
        
        pages = data.get("query", {}).get("pages", {})
        links = []
        for _, page_data in pages.items():
            if "links" in page_data:
                links = [l["title"] for l in page_data["links"]]
                
        return links
    except Exception as ex:
        logging.error(f"failed to get linked pages, {ex}")
        return []
        
def filter_paragraphs(paragraphs: List[str], min_length: int = 150) -> List[str]:
    result: List[str] = []
    for p in paragraphs:
        if p in END_SECTIONS: break
        if len(p) < min_length: continue
        if p and p[0].islower(): continue
        if "doi:10." in p or "ISBN " in p: continue
        result.append(p)
    return result

def main():
    crawl_stack: RandomList[str] = RandomList(TOP_PAGES)
    seen: Set[str] = set(TOP_PAGES)
    n_data = 0
    page_title: str = "Unknown Page"  
    
    with open(SCRAPE_FILE, "w") as f:
        pass
    
    print(f"Starting scaping Target: {N_DATA_TARGET} paragraphs")
    
    while(n_data < N_DATA_TARGET and crawl_stack):
        try:
            page_title: str = crawl_stack.pop()
            
            rev_id: Optional[int] = get_pre_2022_revision_id(page_title)
            
            if(not rev_id):
                logging.warning(f"No pre rev i davailable for topic: {page_title}")
                continue
            
            logging.info(f"Processing data for {page_title} and rev {rev_id}")
            
            new_links = get_linked_articles(page_title)
            random.shuffle(new_links)
            
            for link in new_links[:MAX_LINKS_PER_PAGE]:
                if(link not in seen):
                    crawl_stack.append(link)
                    seen.add(link)
                    
                    
            raw_paragraphs: List[str] = get_clean_text_from_revision(rev_id)
            print(f"Found {len(raw_paragraphs)} raw paragraphs for {page_title}")
            paragraphs: List[str] = filter_paragraphs(raw_paragraphs)
            print(f"After filtering: {len(paragraphs)} remain")
            
            if(paragraphs):
                sampled: List[str] = random.sample(
                    paragraphs, k=min(len(paragraphs), MAX_DATA_PER_PAGE)
                )

                with open(SCRAPE_FILE, "a", encoding="utf-8") as f:
                    for p in sampled:
                        record: Dict[str, str] = {
                            "page": page_title,
                            "text": p,
                            "label":"human"
                        }
                        
                        f.write(json.dumps(record) + "\n")
                        
                n_data += len(sampled)
                print(f"collected {n_data}/{N_DATA_TARGET} paragraphs")
                
        except Exception as ex:
            logging.error(f"Error processing:{page_title}, {ex}")
            time.sleep(1)
            
            
if __name__ == "__main__":
    main()
                