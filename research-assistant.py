# research_assistant.py
# Minimal free-first Multi-Step Research Assistant with LangGraph + local LLM (Ollama)

import os, re, time, math, logging
from typing import List, Dict, Any, Literal, Optional, TypedDict
from dataclasses import dataclass, asdict

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    # --- Orchestration ---
    from langgraph.graph import StateGraph, END

    # --- Search & scraping ---
    from ddgs import DDGS
    import trafilatura

    # --- Embeddings & clustering ---
    from sentence_transformers import SentenceTransformer
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    # --- LLM (local via Ollama) ---
    import ollama
except ImportError as e:
    logger.error(f"Missing required dependency: {e}")
    logger.error("Please install: pip install langgraph duckduckgo-search trafilatura sentence-transformers scikit-learn ollama")
    exit(1)

# --- Helpers ---
def clean_text(t: str) -> str:
    t = re.sub(r"\s+", " ", t or "").strip()
    return t

def validate_ollama_connection(model: str = "llama3") -> bool:
    """Check if Ollama is running and model is available"""
    try:
        ollama.chat(model=model, messages=[{"role": "user", "content": "test"}])
        return True
    except Exception as e:
        logger.error(f"Ollama connection failed: {e}")
        return False

# ---------- Shared State ----------
class Source(TypedDict):
    title: str
    url: str
    content: str
    summary: str
    cluster: Optional[int]

class RState(TypedDict):
    query: str
    sources: List[Source]
    clusters: Dict[int, Dict[str, Any]]  # {cluster_id: {"label": str, "members": [idxs]}}
    table_md: str
    report_md: str
    critique: str
    loop_count: int
    error: Optional[str]  # Track errors

# ---------- Local LLM wrapper ----------
def llm(prompt: str, model: str = "llama3") -> str:
    try:
        resp = ollama.chat(
            model=model, 
            messages=[{"role": "user", "content": prompt}], 
            options={"temperature": 0.2}
        )
        return resp["message"]["content"].strip()
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        return f"Error: Could not generate response - {str(e)}"

# ---------- Node 1: Collect sources ----------
def collect_node(state: RState) -> RState:
    query = state["query"]
    results = []
    
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, region="us-en", safesearch="moderate", timelimit=None, max_results=15, backend="auto"):
                results.append(r)
    except Exception as e:
        logger.error(f"Search failed: {e}")
        state["error"] = f"Search failed: {str(e)}"
        return state
    
    sources: List[Source] = []
    for r in results:
        url = r.get("href") or r.get("url")
        title = r.get("title") or r.get("body") or "Untitled"
        if not url:
            continue
        
        try:
            downloaded = trafilatura.fetch_url(url, no_ssl=True, timeout=10)
            if downloaded:
                extracted = trafilatura.extract(downloaded, include_comments=False, include_tables=False) or ""
            else:
                extracted = ""
        except Exception as e:
            logger.warning(f"Failed to extract from {url}: {e}")
            extracted = ""
            
        if len(extracted) < 200:  # Lowered threshold
            continue
            
        sources.append({
            "title": clean_text(title), 
            "url": url, 
            "content": clean_text(extracted), 
            "summary": "", 
            "cluster": None
        })
        
        if len(sources) >= 8:
            break
    
    if not sources:
        state["error"] = "No valid sources found"
        logger.warning("No sources collected")
    else:
        logger.info(f"Collected {len(sources)} sources")
    
    state["sources"] = sources
    return state

# ---------- Node 2: Summarize & normalize ----------
SUMMARY_PROMPT = """You are an analyst. Summarize the following article in ~8 bullet points.
Focus on: company names, products, business model, market segment, region, traction (funding/users/revenue), differentiators, risks.

Text:
---
{body}
---

Return only bullet points, no other text:"""

def summarize_node(state: RState) -> RState:
    if state.get("error"):
        return state
        
    new_sources = []
    for i, s in enumerate(state["sources"]):
        if s["summary"]:
            new_sources.append(s)
            continue
            
        body = s["content"][:5000]  # Reasonable limit for local models
        summary = llm(SUMMARY_PROMPT.format(body=body))
        
        if summary.startswith("Error:"):
            logger.warning(f"Failed to summarize source {i+1}: {summary}")
            s["summary"] = f"• {s['title']}\n• Content from: {s['url']}"
        else:
            s["summary"] = summary
            
        new_sources.append(s)
        time.sleep(0.5)  # Be gentle with local models
        
    state["sources"] = new_sources
    return state

# ---------- Node 3: Cluster findings ----------
def cluster_node(state: RState) -> RState:
    if state.get("error"):
        return state
        
    summaries = [s["summary"] for s in state["sources"] if s["summary"]]
    
    if len(summaries) < 2:
        # Create single cluster
        state["clusters"] = {0: {"label": "All Sources", "members": list(range(len(state["sources"])))}}
        for i in range(len(state["sources"])):
            state["sources"][i]["cluster"] = 0
        return state

    try:
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        X = model.encode(summaries, normalize_embeddings=True)
    except Exception as e:
        logger.error(f"Embedding failed: {e}")
        # Fallback to single cluster
        state["clusters"] = {0: {"label": "All Sources", "members": list(range(len(state["sources"])))}}
        for i in range(len(state["sources"])):
            state["sources"][i]["cluster"] = 0
        return state

    # Choose K with silhouette score
    best_k, best_score = 2, -1
    best_labels = None
    
    for k in range(2, min(6, len(summaries))):
        try:
            km = KMeans(n_clusters=k, n_init="auto", random_state=42)
            labels = km.fit_predict(X)
            
            if len(set(labels)) > 1:  # Ensure multiple clusters
                score = silhouette_score(X, labels)
                if score > best_score:
                    best_k, best_score = k, score
                    best_labels = labels
        except Exception as e:
            logger.warning(f"Clustering with k={k} failed: {e}")
            continue

    if best_labels is None:
        # Fallback to single cluster
        state["clusters"] = {0: {"label": "All Sources", "members": list(range(len(state["sources"])))}}
        for i in range(len(state["sources"])):
            state["sources"][i]["cluster"] = 0
        return state

    # Build clusters
    clusters: Dict[int, Dict[str, Any]] = {i: {"label": "", "members": []} for i in range(best_k)}
    for idx, lab in enumerate(best_labels):
        state["sources"][idx]["cluster"] = int(lab)
        clusters[int(lab)]["members"].append(idx)

    # Name clusters
    for cid, info in clusters.items():
        if not info["members"]:
            continue
            
        sample_summaries = [state["sources"][idx]["summary"] for idx in info["members"][:3]]
        label_prompt = f"Name this cluster theme in ≤4 words based on these summaries:\n\n" + "\n\n".join(sample_summaries)
        label = llm(label_prompt)
        
        if label.startswith("Error:"):
            clusters[cid]["label"] = f"Group {cid+1}"
        else:
            clusters[cid]["label"] = clean_text(label)

    state["clusters"] = clusters
    return state

# ---------- Node 4: Comparative table ----------
def table_node(state: RState) -> RState:
    if state.get("error"):
        return state
        
    rows = []
    header = "| Company | Offering | Segment | Region | Traction | Differentiators | Risks |\n|---|---|---|---|---|---|---|"
    
    for s in state["sources"]:
        cluster_label = state["clusters"].get(s.get("cluster", 0), {}).get("label", "Other")
        
        # More structured prompt for table generation
        cell_prompt = f"""Extract company info into this exact format (single markdown table row):
| CompanyName | What they offer | Market segment | Geographic region | Funding/users/revenue | Key differentiators | Main risks |

From this summary:
{s['summary'][:1000]}

Return ONLY the table row, starting with |"""
        
        cell = llm(cell_prompt)
        if not cell.startswith("|"):
            # Fallback row
            cell = f"| {s['title'][:30]} | See summary | Unknown | Unknown | Unknown | See source | Unknown |"
        
        rows.append((cluster_label, cell))

    # Sort by cluster
    rows.sort(key=lambda r: r[0])
    table_lines = [header] + [r[1] for r in rows]
    state["table_md"] = "\n".join(table_lines)
    return state

# ---------- Node 5: Report composer ----------
REPORT_PROMPT = """Write a competitive analysis report in Markdown format with these sections:

# Competitive Analysis Report

## Executive Summary
- [3-4 bullet points summarizing key findings]

## Market Landscape  
- [Describe the main segments/clusters found]

## Competitor Analysis
- [Reference the comparative table and highlight key patterns]

## Key Insights
- [Main differentiators, risks, and opportunities identified]

## Conclusion
- [Brief outlook and recommendations]

Use the following data:

Clusters Found:
{clusters}

Comparative Data:
{table}

Keep the report concise (500-700 words) and professional."""

def report_node(state: RState) -> RState:
    if state.get("error"):
        # Generate error report
        state["report_md"] = f"# Research Assistant Report\n\n**Error:** {state['error']}\n\nPlease try again with a different query or check your setup."
        return state
    
    cluster_overview = "\n".join([
        f"- **{info['label']}**: {len(info['members'])} companies" 
        for cid, info in state["clusters"].items() if info['members']
    ])
    
    report_md = llm(REPORT_PROMPT.format(clusters=cluster_overview, table=state["table_md"]))
    
    # Add sources appendix
    appendix = "\n\n---\n\n## Sources\n"
    for i, s in enumerate(state["sources"]):
        cluster_label = state["clusters"].get(s.get("cluster", 0), {}).get("label", "Other")
        appendix += f"{i+1}. **{cluster_label}**: [{s['title']}]({s['url']})\n"
    
    state["report_md"] = report_md + appendix
    return state

# ---------- Node 6: Critic ----------
CRITIC_PROMPT = """Review this research report quality:

CRITERIA:
- Are there at least 4 credible sources?
- Are different market segments represented?
- Is the analysis coherent and useful?

REPORT TO REVIEW:
{report}

RESPOND WITH:
OK - if the report meets quality standards
NEEDS_MORE - if significant gaps exist

Then explain your reasoning in 1-2 sentences."""

def critic_node(state: RState) -> RState:
    if state.get("error"):
        state["critique"] = "NEEDS_MORE - Error in data collection"
        return state
        
    critique = llm(CRITIC_PROMPT.format(report=state["report_md"][:2000]))  # Limit length
    state["critique"] = critique
    return state

# ---------- Conditional edge ----------
def gate(state: RState) -> Literal["Loop", "Ship"]:
    if state.get("error"):
        return "Ship"  # Don't loop on errors
        
    critique = state.get("critique", "")
    needs_more = "NEEDS_MORE" in critique.upper()
    
    # Allow one retry with broader search
    if needs_more and state["loop_count"] < 1:
        state["loop_count"] += 1
        # Modify query for broader results
        original_query = state["query"].split(" site:")[0]  # Remove previous site filters
        state["query"] = f"{original_query} news OR updates OR analysis 2024"
        logger.info("Critique suggests improvement needed, retrying with broader search...")
        return "Loop"
    
    return "Ship"

# ---------- Build the graph ----------
def create_app():
    graph = StateGraph(RState)
    graph.add_node("Collect", collect_node)
    graph.add_node("Summarize", summarize_node)
    graph.add_node("Cluster", cluster_node)
    graph.add_node("Table", table_node)
    graph.add_node("Report", report_node)
    graph.add_node("Critic", critic_node)

    graph.set_entry_point("Collect")
    graph.add_edge("Collect", "Summarize")
    graph.add_edge("Summarize", "Cluster")
    graph.add_edge("Cluster", "Table")
    graph.add_edge("Table", "Report")
    graph.add_edge("Report", "Critic")

    graph.add_conditional_edges("Critic", gate, {"Loop": "Collect", "Ship": END})
    return graph.compile()

def run(query: str, model: str = "llama3"):
    """Run the research assistant"""
    # Validate setup
    if not validate_ollama_connection(model):
        print("❌ Ollama connection failed. Please ensure Ollama is running and the model is available.")
        return
    
    print(f"🔍 Starting research on: {query}")
    
    state: RState = {
        "query": query,
        "sources": [],
        "clusters": {},
        "table_md": "",
        "report_md": "",
        "critique": "",
        "loop_count": 0,
        "error": None
    }
    
    app = create_app()
    
    try:
        final = app.invoke(state)
        
        # Write report
        output_file = "report.md"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(final["report_md"])
        
        print(f"✅ Analysis complete! Report saved to {output_file}")
        print(f"📊 Processed {len(final['sources'])} sources across {len(final['clusters'])} clusters")
        
        if final.get("error"):
            print(f"⚠️  Warning: {final['error']}")
            
    except Exception as e:
        logger.error(f"Application failed: {e}")
        print(f"❌ Research failed: {e}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        query = "competitive analysis fintech startups 2024"
        print(f"No query provided, using default: {query}")
    else:
        query = " ".join(sys.argv[1:])
    
    run(query)