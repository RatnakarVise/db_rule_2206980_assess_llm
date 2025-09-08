from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import os, json
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv

# --- Load environment ---
dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=dotenv_path)
langchain_api_key = os.getenv("LANGCHAIN_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")
if langchain_api_key:
    os.environ["LANGCHAIN_API_KEY"] = langchain_api_key
if openai_api_key:
    os.environ["OPENAI_API_KEY"] = openai_api_key
os.environ["LANGCHAIN_TRACING_V2"] = "true"

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-nano")

app = FastAPI(title="OSS Note 2206980 Assessment & Remediation Prompt (MM-IM Changes)")


# ---- Input models ----
class MMIMUsage(BaseModel):
    table: str
    target_type: str
    target_name: str
    used_fields: List[str]
    suggested_fields: List[str]
    suggested_statement: Optional[str] = None
    snippet: Optional[str] = None


class NoteContext(BaseModel):
    pgm_name: Optional[str] = None
    inc_name: Optional[str] = None
    type: Optional[str] = None
    name: Optional[str] = None
    mmim_usage: List[MMIMUsage] = Field(default_factory=list)


# ---- Summarizer ----
def summarize_context(ctx: NoteContext) -> dict:
    return {
        "unit_program": ctx.pgm_name,
        "unit_include": ctx.inc_name,
        "unit_type": ctx.type,
        "name": ctx.name,
        "mmim_usage": [item.model_dump() for item in ctx.mmim_usage]
    }


# ---- LangChain Prompt ----
SYSTEM_MSG = """You are a senior ABAP expert. Output ONLY JSON as response.
In llm_prompt: For every provided payload item,
write a bullet point that:
- Displays the exact offending code
- Explains the necessary action to fix the offset error using the provided .suggestion text (if available).
- Bullet points should contain both offending code snippet and the fix (no numbering or referencing like "snippet[1]": display the code inline).
- Do NOT omit any snippet; all must be covered, no matter how many there are.
- Only show actual ABAP code for each snippet with its specific action.
""".strip()

USER_TEMPLATE = """
You are evaluating a system context related to SAP OSS Note 2206980 (S/4HANA MM-IM Data Model Changes).
We provide:
- program/include/type metadata
- ABAP code with detected MM-IM table usage

Your job:
1) Provide a concise **assessment**:
   - If MKPF/MSEG is found, suggest MATDOC.
   - If MARD/MARC/MKPF aggregate/split/history tables are found, suggest corresponding CDS views (e.g., NSDM_DDL_MARD).
   - Merge all these recommendations into `suggested_statement`.
   - Ignore the place where we are doing UPDATE functionality of these MARC/MARD/MKPFetc tables.

2) Provide an actionable **LLM remediation prompt**:
   - Reference program/include/type/name.
   - Ask to locate all obsolete table reads/writes.
   - Replace them with MATDOC/CDS views ensuring functional equivalence.

Return ONLY strict JSON:
{{
  "assessment": "<concise 2206980 impact>",
  "llm_prompt": "<prompt for LLM code fixer>"
}}
   
Unit metadata:
- Program: {pgm_name}
- Include: {inc_name}
- Unit type: {type}
- Unit name: {name}

System context:
{context_json}
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_MSG),
    ("user", USER_TEMPLATE),
])

llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0)
parser = JsonOutputParser()
chain = prompt | llm | parser


def llm_assess(ctx: NoteContext):
    ctx_json = json.dumps(summarize_context(ctx), ensure_ascii=False, indent=2)
    return chain.invoke({
        "context_json": ctx_json,
        "pgm_name": ctx.pgm_name,
        "inc_name": ctx.inc_name,
        "type": ctx.type,
        "name": ctx.name
    })


@app.post("/assess-2206980")
async def assess_note_context(ctxs: List[NoteContext]):
    results = []
    for ctx in ctxs:
        try:
            llm_result = llm_assess(ctx)
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"LLM call failed: {e}")

        results.append({
            "pgm_name": ctx.pgm_name,
            "inc_name": ctx.inc_name,
            "type": ctx.type,
            "name": ctx.name,
            "code": "",  # keep ABAP code outside response
            "assessment": llm_result.get("assessment", ""),
            "llm_prompt": llm_result.get("llm_prompt", "")
        })

    return results


@app.get("/health")
def health():
    return {"ok": True, "model": OPENAI_MODEL}
