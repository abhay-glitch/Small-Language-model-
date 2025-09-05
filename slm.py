import os
import math
import re
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tokenizers import ByteLevelBPETokenizer
from datetime import datetime
from typing import Optional, List


torch.manual_seed(42)


SCHEMA = {
    "status": "1",
    "alert": "0",
    "message": "Successfully analyzed and parsed resume PDF",
    "token": "",
    "data": {
        "parsedData": {
            "Name": None,
            "Mobile_Number": None,
            "Address": None,
            "City": None,
            "Zip_Code": None,
            "State": None,
            "Country": None,
            "Email": None,
            "LinkedIn": None,
            "GitHub": None,
            "Experience": [],
            "Education": [],
            "Years_of_Experience": None,
            "Skills": [],
            "Languages": []
        }
    }
}


edu_pattern = re.compile(
    r'(.+?)\s*[-–]\s*(.+?)\s*\('
    r'(\d{4}(?:[-/]\d{2}(?:[-/]\d{2})?)?)'          
    r'\s*[-–]\s*'
    r'(\d{4}(?:[-/]\d{2}(?:[-/]\d{2})?)?|current)' 
    r'(?:\s*,\s*([A-Za-z\s.,]+))?'                 
    r'\)',
    re.IGNORECASE
)

exp_pattern = re.compile(
    r'(.+?)\s*[-–]\s*(.+?)\s*\('
    r'(\d{4}(?:[-/]\d{2}(?:[-/]\d{2})?)?)'
    r'\s*[-–]\s*'
    r'(\d{4}(?:[-/]\d{2}(?:[-/]\d{2})?)?|current)'
    r'(?:\s*/\s*([A-Za-z\s.,]+))?'                
    r'\)',
    re.IGNORECASE
)

def _try_parse_date(text: str) -> Optional[datetime]:
    if not text or text.lower() == "current":
        return None
    text = text.strip()
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y-%m", "%Y/%m", "%Y"):
        try:
            return datetime.strptime(text, fmt)
        except Exception:
            continue
 
    m = re.search(r'\d{4}', text)
    if m:
        return datetime.strptime(m.group(0), "%Y")
    return None

def calculate_experience_years(experiences: List[dict]) -> Optional[str]:
    try:
        if not experiences:
            return None
        starts = []
        ends = []
        for e in experiences:
            s = _try_parse_date(e.get("Start_Date"))
            e_ = e.get("End_Date")
            if e_ and isinstance(e_, str) and e_.lower() == "current":
                end_dt = datetime.today()
            else:
                end_dt = _try_parse_date(e_.strip() if e_ else None) or datetime.today()
            if s:
                starts.append(s)
                ends.append(end_dt)
        if not starts:
            return None
        min_start = min(starts)
        max_end = max(ends) if ends else datetime.today()
        years = round((max_end - min_start).days / 365.0, 1)
        return str(years)
    except Exception:
        return None


def extract_basic_fields(resume_text: str) -> dict:
    data = {}


    lines = [l.strip() for l in resume_text.splitlines() if l.strip()]
    data["Name"] = lines[0].strip('"') if lines else None


    email_match = re.search(r'[\w\.-]+@[\w\.-]+', resume_text)
    data["Email"] = email_match.group(0) if email_match else None


    phone_match = re.search(r'(\(?\d{2,3}\)?[\s-]?\d{3,}[\s-]?\d{3,})', resume_text)
    data["Mobile_Number"] = phone_match.group(0) if phone_match else None


    addr_match = re.search(r'([A-Za-z\s]+,\s*[A-Za-z]{2,}(?:,\s*[A-Za-z\s]+)?)', resume_text)
    data["Address"] = addr_match.group(0).strip() if addr_match else None
    data["City"] = None
    data["State"] = None
    data["Country"] = None
    if data["Address"]:
        parts = [p.strip() for p in data["Address"].split(",")]
        if len(parts) >= 1: data["City"] = parts[0]
        if len(parts) >= 2: data["State"] = parts[1]
        if len(parts) >= 3: data["Country"] = parts[2]

 
    linkedin_match = re.search(r'(https?://[^\s]*linkedin\.com[^\s]*|LinkedIn)', resume_text, re.IGNORECASE)
    data["LinkedIn"] = linkedin_match.group(0) if linkedin_match else None

    github_match = re.search(r'(https?://[^\s]*github\.com[^\s]*|GitHub)', resume_text, re.IGNORECASE)
    data["GitHub"] = github_match.group(0) if github_match else None


    skills_match = re.search(r'SKILLS([\s\S]*?)(EDUCATION|WORK EXPERIENCE|LANGUAGES|CERTIFICATIONS|$)', resume_text, re.IGNORECASE)
    if skills_match:
        skills_text = skills_match.group(1).replace("\n", " ")
        data["Skills"] = [s.strip(" -") for s in re.split(r'[,|;]', skills_text) if s.strip()]
    else:
        data["Skills"] = []


    langs_match = re.search(r'LANGUAGES([\s\S]*?)(EDUCATION|WORK EXPERIENCE|CERTIFICATIONS|$)', resume_text, re.IGNORECASE)
    if langs_match:
        langs_text = langs_match.group(1).replace("\n", " ")
        data["Languages"] = [l.strip(" -") for l in re.split(r'[,|;]', langs_text) if l.strip()]
    else:
        data["Languages"] = []

    education = []
    edu_section = re.search(r'EDUCATION([\s\S]*?)(SKILLS|WORK EXPERIENCE|CERTIFICATIONS|LANGUAGES|$)', resume_text, re.IGNORECASE)
    if edu_section:
        for raw in edu_section.group(1).splitlines():
            line = raw.strip()
            if not line:
                continue
            m = edu_pattern.match(line)
            if m:
                degree, institution, start, end, location = m.groups()
                education.append({
                    "Degree": degree.strip(),
                    "Institution": institution.strip(),
                    "Graduation_Start_Date": start.strip(),
                    "Graduation_End_Date": end.strip(),
                    "Location": location.strip() if location else None
                })
    data["Education"] = education


    experience = []
    exp_section = re.search(r'WORK EXPERIENCE([\s\S]*?)(EDUCATION|SKILLS|CERTIFICATIONS|LANGUAGES|$)', resume_text, re.IGNORECASE)
    if exp_section:
   
        blocks = [b for b in exp_section.group(1).split("\n\n") if b.strip()]
        for block in blocks:
            lines_b = [l for l in block.splitlines() if l.strip()]
            if not lines_b:
                continue
            header_line = lines_b[0].strip()
            m = exp_pattern.match(header_line)
            if m:
                left, right, start, end, location = m.groups()

                job_title, company = left.strip(), right.strip()
                bullets = [b.strip("- ").strip() for b in lines_b[1:] if b.strip().startswith("-")]
                experience.append({
                    "Job_Title": job_title,
                    "Company": company,
                    "Start_Date": start.strip(),
                    "End_Date": end.strip(),
                    "Location": location.strip() if location else None,
                    "Description": "\n".join(bullets) if bullets else None
                })
    data["Experience"] = experience

    data["Years_of_Experience"] = calculate_experience_years(experience)

    return data

def _pick_col(df: pd.DataFrame, candidates: List[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"None of the expected columns {candidates} found. Columns present: {list(df.columns)}")

def get_bpe_tokenizer(csv_path: str, vocab_size=8000) -> ByteLevelBPETokenizer:
    if not (os.path.exists("bpe_tokenizer/vocab.json") and os.path.exists("bpe_tokenizer/merges.txt")):
        print("Training new tokenizer...")
        df = pd.read_csv(csv_path, encoding="ISO-8859-1")
        resume_col = _pick_col(df, ["Resume", "Resume "])
        json_col   = _pick_col(df, ["JSON", "JSON "])
        texts = (df[resume_col].astype(str) + "\n" + df[json_col].astype(str)).tolist()
        os.makedirs("bpe_tokenizer", exist_ok=True)
        with open("train_text.txt", "w", encoding="utf-8") as f:
            for t in texts:
                f.write((t or "") + "\n")
        tokenizer = ByteLevelBPETokenizer()
        tokenizer.train(files="train_text.txt", vocab_size=vocab_size, min_frequency=2, special_tokens=["<pad>", "<resume>", "</resume>", "<json>", "</json>"])
        tokenizer.save_model("bpe_tokenizer")
    return ByteLevelBPETokenizer("bpe_tokenizer/vocab.json", "bpe_tokenizer/merges.txt")


class ResumeDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer: ByteLevelBPETokenizer, block_size=256, pad_id=0):
        self.tok = tokenizer
        self.block_size = block_size
        self.pad_id = pad_id
        resume_col = _pick_col(df, ["Resume", "Resume "])
        json_col   = _pick_col(df, ["JSON", "JSON "])
        self.samples = []
        for i in range(len(df)):
            resume = str(df[resume_col].iloc[i]) if pd.notna(df[resume_col].iloc[i]) else ""
            js     = str(df[json_col].iloc[i]) if pd.notna(df[json_col].iloc[i]) else ""
            text = f"<resume> {resume} </resume> <json> {js} </json>"
            ids = self.tok.encode(text).ids
           
            ids = ids[:self.block_size] if len(ids) > self.block_size else ids + [self.pad_id] * (self.block_size - len(ids))
            self.samples.append(ids)

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        data = self.samples[idx]
      
        x = torch.tensor(data[:-1], dtype=torch.long)  
        y = torch.tensor(data[1:], dtype=torch.long)   
        return x, y


class SelfAttention(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head

    def forward(self, x):
        B, T, C = x.size()
        H = self.n_head
        head_dim = C // H
        k = self.key(x).view(B, T, H, head_dim).transpose(1, 2)  
        q = self.query(x).view(B, T, H, head_dim).transpose(1, 2)
        v = self.value(x).view(B, T, H, head_dim).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(head_dim)     
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        att = att.masked_fill(mask, float('-inf'))
        att = F.softmax(att, dim=-1)
        out = att @ v                                          
        out = out.transpose(1, 2).contiguous().view(B, T, C)      
        return self.proj(out)

class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.attn = SelfAttention(n_embd, n_head)
        self.ff = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd)
        )
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class SmallLanguageModel(nn.Module):
    def __init__(self, vocab_size, n_embd=256, n_layer=4, n_head=4, block_size=256, pad_id=0):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[TransformerBlock(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)
        self.block_size = block_size
        self.pad_id = pad_id

    def forward(self, idx, targets=None):
        B, T = idx.size()
        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)
        x = self.token_emb(idx) + self.pos_emb(pos)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            # optional: ignore pad positions
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=self.pad_id
            )
        return logits, loss


def train_slm(csv_path, epochs=5, batch_size=2, block_size=256, lr=3e-4, device="cpu"):
    tokenizer = get_bpe_tokenizer(csv_path)
    df = pd.read_csv(csv_path, encoding="ISO-8859-1")
    pad_id = 0  
    dataset = ResumeDataset(df, tokenizer, block_size=block_size, pad_id=pad_id)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    vocab_size = tokenizer.get_vocab_size()
    model = SmallLanguageModel(vocab_size, block_size=block_size, pad_id=pad_id).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        last_loss = None
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits, loss = model(x, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            last_loss = loss.item()
        print(f"Epoch {epoch+1} Loss: {last_loss:.4f}")
    torch.save(model.state_dict(), "slm_json.pth")
    return model, tokenizer

@torch.no_grad()
def generate(model, tokenizer, start_text="<resume>", max_new_tokens=200, device="cpu"):
    model.eval()
    ids = tokenizer.encode(start_text).ids
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.block_size:]
        logits, _ = model(idx_cond, None)
        probs = F.softmax(logits[:, -1, :], dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_id], dim=1)
    out_ids = idx[0].tolist()
    return tokenizer.decode(out_ids)


def enforce_schema(resume_text: str, raw_json_text: str):
    try:
        parsed = json.loads(raw_json_text)
    except Exception:
        parsed = {}

    def merge(schema, data):
        if isinstance(schema, dict):
            result = {}
            for k in schema:
                v_schema = schema[k]
                v_data = data.get(k, v_schema) if isinstance(data, dict) else v_schema
                result[k] = merge(v_schema, v_data)
            return result
        elif isinstance(schema, list):
            return data if isinstance(data, list) and len(data) > 0 else schema
        else:
            return data if (data not in (None, "", "null")) else schema

    base = merge(SCHEMA, parsed)

  
    extracted = extract_basic_fields(resume_text)
    for k, v in extracted.items():
        if v is not None:
            base["data"]["parsedData"][k] = v

    return base


if __name__ == "__main__":
    # train on your CSV
    model, tokenizer = train_slm("newone.csv", epochs=20, batch_size=4, block_size=256, lr=3e-4, device="cpu")

    resume_text = """
IVANA REYNOLDS
Alteryx Data Analyst
Email: i.reynolds@email.com
Phone: (123) 456-7890
Address: San Diego, CA
LinkedIn: LinkedIn

EDUCATION
Bachelor of Science -Computer Science, University of California, San Diego (2009-01-01 - 2013-01-01)

SKILLS
Workflow canvas, Data preparation tools, Predictive modeling tools, Data visualization, Interactive dashboards, Database querying, Data manipulation, Data extraction

WORK EXPERIENCE
Intuit Inc. -Data Analytics Manager (2019-01-01 - current / San Diego, CA)
- Led a team of 8 data analysts, achieving a 27% productivity boost via strategic delegation.
- Consolidated disparate data sources, improving data availability by 41%.
- Introduced governance measures that reduced data errors by 32%.
- Delivered market insights that enhanced marketing campaign performance by 17%.

Illumina, Inc. - Senior Data Analyst (2016-01-01 - 2019-01-01 / San Diego, CA)
- Cleaned and transformed raw data, raising data quality by 29%.
- Built predictive models that boosted customer retention by 46%.
- Created real-time dashboards, cutting response times to business events.
- Applied visualization techniques that improved stakeholder decision-making by 57%.

Qualcomm Incorporated -Database Administrator (2013-01-01 - 2016-01-01 / San Diego, CA)
- Built workflow canvas reducing data processing time by 42%.
- Optimized database queries, reducing retrieval times by 36%.
- Consolidated redundant data, reducing storage needs by 51%.
- Automated backups, minimizing recovery time by 63%

"""

   
    raw_output = generate(model, tokenizer, start_text=f"<resume> {resume_text} </resume> <json>", max_new_tokens=300, device="cpu")
    final_json = enforce_schema(resume_text, raw_output)
    print(json.dumps(final_json, indent=4))
