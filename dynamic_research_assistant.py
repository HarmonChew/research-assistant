# dynamic_research_assistant.py
# Extensible Multi-Domain Research Assistant with Dynamic Prompt Generation

import os, re, time, math, logging
from typing import List, Dict, Any, Literal, Optional, TypedDict
from dataclasses import dataclass
from enum import Enum
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from langgraph.graph import StateGraph, END
    from ddgs import DDGS
    import trafilatura
    from sentence_transformers import SentenceTransformer
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    import ollama
except ImportError as e:
    logger.error(f"Missing required dependency: {e}")
    exit(1)

# --- Domain Classification System ---
class ResearchDomain(Enum):
    BUSINESS_COMPETITIVE = "business_competitive"
    TECHNOLOGY_TRENDS = "technology_trends"
    ACADEMIC_RESEARCH = "academic_research"
    MARKET_ANALYSIS = "market_analysis"
    POLICY_REGULATORY = "policy_regulatory"
    HEALTHCARE_MEDICAL = "healthcare_medical"
    FINANCIAL_INVESTMENT = "financial_investment"
    GENERAL_TOPIC = "general_topic"

@dataclass
class DomainConfig:
    """Configuration for domain-specific research behavior"""
    search_terms: List[str]
    summary_focus_areas: List[str]
    analysis_dimensions: List[str]
    report_sections: List[str]
    cluster_approach: str
    quality_criteria: List[str]

# --- Dynamic Prompt Templates System ---
class PromptTemplateManager:
    """Manages domain-specific prompt templates"""
    
    def __init__(self):
        self.domain_configs = {
            ResearchDomain.BUSINESS_COMPETITIVE: DomainConfig(
                search_terms=["competitors", "market share", "business model", "startups", "companies"],
                summary_focus_areas=["company names", "products", "business model", "market segment", "funding", "differentiators", "risks"],
                analysis_dimensions=["Company", "Offering", "Segment", "Region", "Traction", "Differentiators", "Risks"],
                report_sections=["Executive Summary", "Market Landscape", "Competitor Analysis", "Key Insights", "Strategic Recommendations"],
                cluster_approach="business_segments",
                quality_criteria=["diverse companies", "clear business models", "competitive differentiation"]
            ),
            
            ResearchDomain.TECHNOLOGY_TRENDS: DomainConfig(
                search_terms=["technology", "innovation", "emerging", "trends", "breakthrough", "advancement"],
                summary_focus_areas=["technology name", "applications", "maturity level", "key players", "market impact", "limitations", "future potential"],
                analysis_dimensions=["Technology", "Applications", "Maturity", "Key Players", "Market Impact", "Limitations", "Timeline"],
                report_sections=["Executive Summary", "Technology Landscape", "Adoption Patterns", "Key Players", "Future Outlook"],
                cluster_approach="technology_categories",
                quality_criteria=["diverse technologies", "clear applications", "credible sources"]
            ),
            
            ResearchDomain.ACADEMIC_RESEARCH: DomainConfig(
                search_terms=["research", "study", "academic", "journal", "findings", "methodology"],
                summary_focus_areas=["research question", "methodology", "key findings", "sample size", "limitations", "implications", "future research"],
                analysis_dimensions=["Study", "Methodology", "Sample Size", "Key Findings", "Limitations", "Significance", "Citations"],
                report_sections=["Literature Overview", "Methodological Approaches", "Key Findings", "Research Gaps", "Future Directions"],
                cluster_approach="research_methods",
                quality_criteria=["peer-reviewed sources", "diverse methodologies", "recent publications"]
            ),
            
            ResearchDomain.HEALTHCARE_MEDICAL: DomainConfig(
                search_terms=["health", "medical", "treatment", "clinical", "patients", "diagnosis"],
                summary_focus_areas=["condition/disease", "treatment approach", "efficacy", "side effects", "patient population", "regulatory status", "cost"],
                analysis_dimensions=["Condition", "Treatment", "Efficacy", "Safety", "Population", "Regulatory Status", "Cost"],
                report_sections=["Medical Overview", "Treatment Landscape", "Clinical Evidence", "Regulatory Environment", "Patient Impact"],
                cluster_approach="medical_specialties",
                quality_criteria=["clinical evidence", "peer-reviewed sources", "regulatory compliance"]
            ),
            
            ResearchDomain.GENERAL_TOPIC: DomainConfig(
                search_terms=["overview", "analysis", "information", "facts", "details"],
                summary_focus_areas=["key facts", "main concepts", "important details", "relationships", "implications", "context"],
                analysis_dimensions=["Topic", "Key Facts", "Context", "Implications", "Sources", "Credibility", "Relevance"],
                report_sections=["Topic Overview", "Key Information", "Analysis", "Implications", "Conclusions"],
                cluster_approach="thematic_grouping",
                quality_criteria=["comprehensive coverage", "credible sources", "factual accuracy"]
            )
        }
    
    def classify_research_domain(self, query: str) -> ResearchDomain:
        """Automatically classify research domain based on query"""
        query_lower = query.lower()
        
        # Business/competitive keywords
        business_keywords = ["competitor", "competitive", "market", "business", "startup", "company", "industry", "revenue", "funding"]
        if any(keyword in query_lower for keyword in business_keywords):
            return ResearchDomain.BUSINESS_COMPETITIVE
        
        # Technology keywords  
        tech_keywords = ["technology", "tech", "ai", "machine learning", "software", "innovation", "digital", "platform"]
        if any(keyword in query_lower for keyword in tech_keywords):
            return ResearchDomain.TECHNOLOGY_TRENDS
        
        # Academic keywords
        academic_keywords = ["research", "study", "academic", "literature", "theory", "methodology", "findings"]
        if any(keyword in query_lower for keyword in academic_keywords):
            return ResearchDomain.ACADEMIC_RESEARCH
        
        # Healthcare keywords
        health_keywords = ["health", "medical", "disease", "treatment", "clinical", "patient", "drug", "therapy"]
        if any(keyword in query_lower for keyword in health_keywords):
            return ResearchDomain.HEALTHCARE_MEDICAL
        
        # Policy keywords
        policy_keywords = ["policy", "regulation", "government", "law", "regulatory", "compliance", "legal"]
        if any(keyword in query_lower for keyword in policy_keywords):
            return ResearchDomain.POLICY_REGULATORY
        
        return ResearchDomain.GENERAL_TOPIC
    
    def generate_search_query(self, original_query: str, domain: ResearchDomain) -> str:
        """Enhance search query with domain-specific terms"""
        config = self.domain_configs[domain]
        enhanced_terms = " OR ".join(config.search_terms[:3])  # Add top 3 domain terms
        return f"{original_query} ({enhanced_terms})"
    
    def generate_summary_prompt(self, content: str, domain: ResearchDomain) -> str:
        """Generate domain-specific summarization prompt"""
        config = self.domain_configs[domain]
        focus_areas = ", ".join(config.summary_focus_areas)
        
        return f"""You are a domain expert analyst. Summarize the following content in ~8 bullet points.
Focus specifically on: {focus_areas}

Extract and highlight information relevant to these areas. If certain focus areas are not present in the content, note their absence.

Content:
---
{content[:5000]}
---

Return only bullet points, no other text:"""
    
    def generate_table_prompt(self, summary: str, domain: ResearchDomain) -> str:
        """Generate domain-specific table extraction prompt"""
        config = self.domain_configs[domain]
        columns = " | ".join(config.analysis_dimensions)
        
        return f"""Extract information into this exact markdown table format:
| {columns} |

From this summary:
{summary[:1000]}

Fill each column with relevant information from the summary. Use "Unknown" or "N/A" if information is not available.
Return ONLY the table row, starting with |"""
    
    def generate_cluster_prompt(self, summaries: List[str], domain: ResearchDomain) -> str:
        """Generate domain-specific cluster naming prompt"""
        config = self.domain_configs[domain]
        approach = config.cluster_approach.replace("_", " ")
        
        return f"""Name this cluster of related content using the {approach} approach.
Provide a descriptive label in ≤4 words that captures the common theme.

Content summaries:
{summaries[:3]}

Focus on identifying the main {approach} pattern. Return only the cluster name:"""
    
    def generate_report_prompt(self, clusters: str, table: str, domain: ResearchDomain, original_query: str) -> str:
        """Generate domain-specific report writing prompt"""
        config = self.domain_configs[domain]
        sections = "\n".join([f"## {section}" for section in config.report_sections])
        
        return f"""Write a comprehensive research report about: "{original_query}"

Use this structure:
# Research Report: {original_query}

{sections}

Guidelines:
- Write 600-800 words
- Use professional, analytical tone
- Reference the data provided below
- Focus on insights and patterns
- Include actionable conclusions

Data to analyze:
Clusters Found: {clusters}
Comparative Analysis: {table}

Ensure each section provides unique value and insights."""
    
    def generate_critique_prompt(self, report: str, domain: ResearchDomain) -> str:
        """Generate domain-specific quality critique prompt"""
        config = self.domain_configs[domain]
        criteria = ", ".join(config.quality_criteria)
        
        return f"""Review this research report for quality and completeness.

Quality criteria for this domain:
- {criteria}
- Sufficient source diversity
- Clear analytical insights
- Actionable conclusions

Report to review:
{report[:2000]}

RESPOND WITH:
OK - if the report meets quality standards
NEEDS_MORE - if significant gaps exist

Then explain your assessment in 1-2 sentences:"""

# --- Enhanced State System ---
class Source(TypedDict):
    title: str
    url: str
    content: str
    summary: str
    cluster: Optional[int]
    relevance_score: float

class RState(TypedDict):
    original_query: str
    enhanced_query: str
    domain: ResearchDomain
    sources: List[Source]
    clusters: Dict[int, Dict[str, Any]]
    table_md: str
    report_md: str
    critique: str
    loop_count: int
    error: Optional[str]
    config: DomainConfig

# --- Enhanced Agents with Dynamic Prompts ---
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
        return f"Error: {str(e)}"

def initialize_research_node(state: RState) -> RState:
    """New agent: Initialize research configuration based on query"""
    prompt_manager = PromptTemplateManager()
    
    # Classify domain
    domain = prompt_manager.classify_research_domain(state["original_query"])
    state["domain"] = domain
    state["config"] = prompt_manager.domain_configs[domain]
    
    # Enhance query
    state["enhanced_query"] = prompt_manager.generate_search_query(
        state["original_query"], domain
    )
    
    logger.info(f"Classified as {domain.value}, enhanced query: {state['enhanced_query']}")
    return state

def collect_node(state: RState) -> RState:
    """Enhanced collection with domain-specific search"""
    query = state["enhanced_query"]
    results = []
    
    try:
        with DDGS() as ddgs:
            # Use enhanced query for better domain-specific results
            for r in ddgs.text(query, region="us-en", safesearch="moderate", max_results=15):
                results.append(r)
    except Exception as e:
        logger.error(f"Search failed: {e}")
        state["error"] = f"Search failed: {str(e)}"
        return state
    
    sources: List[Source] = []
    for r in results:
        url = r.get("href") or r.get("url")
        title = r.get("title") or "Untitled"
        if not url:
            continue
            
        try:
            downloaded = trafilatura.fetch_url(url, timeout=10)
            if downloaded:
                extracted = trafilatura.extract(downloaded) or ""
            else:
                extracted = ""
        except Exception as e:
            logger.warning(f"Failed to extract from {url}: {e}")
            extracted = ""
            
        if len(extracted) < 300:
            continue
            
        # Calculate relevance score based on domain keywords
        relevance = calculate_relevance_score(extracted, state["config"].search_terms)
        
        sources.append({
            "title": title[:100],
            "url": url,
            "content": extracted,
            "summary": "",
            "cluster": None,
            "relevance_score": relevance
        })
        
        if len(sources) >= 10:
            break
    
    # Sort by relevance and keep top sources
    sources.sort(key=lambda x: x["relevance_score"], reverse=True)
    state["sources"] = sources[:8]
    
    logger.info(f"Collected {len(state['sources'])} sources")
    return state

def calculate_relevance_score(content: str, domain_keywords: List[str]) -> float:
    """Calculate relevance score based on domain keywords"""
    content_lower = content.lower()
    score = 0
    for keyword in domain_keywords:
        score += content_lower.count(keyword.lower())
    return score / len(content) * 1000  # Normalize by content length

def summarize_node(state: RState) -> RState:
    """Enhanced summarization with domain-specific prompts"""
    if state.get("error"):
        return state
    
    prompt_manager = PromptTemplateManager()
    new_sources = []
    
    for i, s in enumerate(state["sources"]):
        if s["summary"]:
            new_sources.append(s)
            continue
            
        # Generate domain-specific summary prompt
        summary_prompt = prompt_manager.generate_summary_prompt(
            s["content"], state["domain"]
        )
        
        summary = llm(summary_prompt)
        
        if summary.startswith("Error:"):
            logger.warning(f"Failed to summarize source {i+1}")
            s["summary"] = f"• {s['title']}\n• Source: {s['url']}\n• Content available for analysis"
        else:
            s["summary"] = summary
            
        new_sources.append(s)
        time.sleep(0.3)
        
    state["sources"] = new_sources
    return state

def cluster_node(state: RState) -> RState:
    """Enhanced clustering with domain-specific approaches"""
    if state.get("error"):
        return state
        
    summaries = [s["summary"] for s in state["sources"] if s["summary"]]
    
    if len(summaries) < 2:
        state["clusters"] = {0: {"label": "All Sources", "members": list(range(len(state["sources"])))}}
        for i in range(len(state["sources"])):
            state["sources"][i]["cluster"] = 0
        return state

    try:
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        X = model.encode(summaries, normalize_embeddings=True)
    except Exception as e:
        logger.error(f"Embedding failed: {e}")
        state["clusters"] = {0: {"label": "All Sources", "members": list(range(len(state["sources"])))}}
        return state

    # Domain-specific clustering approach
    best_k = determine_optimal_clusters(X, state["domain"])
    
    try:
        km = KMeans(n_clusters=best_k, n_init="auto", random_state=42)
        labels = km.fit_predict(X)
    except Exception as e:
        logger.error(f"Clustering failed: {e}")
        labels = [0] * len(summaries)
        best_k = 1

    # Build clusters
    clusters: Dict[int, Dict[str, Any]] = {i: {"label": "", "members": []} for i in range(best_k)}
    for idx, lab in enumerate(labels):
        state["sources"][idx]["cluster"] = int(lab)
        clusters[int(lab)]["members"].append(idx)

    # Generate domain-specific cluster names
    prompt_manager = PromptTemplateManager()
    for cid, info in clusters.items():
        if not info["members"]:
            continue
            
        sample_summaries = [state["sources"][idx]["summary"] for idx in info["members"][:3]]
        cluster_prompt = prompt_manager.generate_cluster_prompt(
            sample_summaries, state["domain"]
        )
        
        label = llm(cluster_prompt)
        clusters[cid]["label"] = label if not label.startswith("Error:") else f"Group {cid+1}"

    state["clusters"] = clusters
    return state

def determine_optimal_clusters(X, domain: ResearchDomain) -> int:
    """Determine optimal cluster count based on domain and data"""
    n_samples = len(X)
    
    # Domain-specific cluster preferences
    domain_preferences = {
        ResearchDomain.BUSINESS_COMPETITIVE: min(4, max(2, n_samples // 2)),
        ResearchDomain.TECHNOLOGY_TRENDS: min(5, max(2, n_samples // 2)),
        ResearchDomain.ACADEMIC_RESEARCH: min(3, max(2, n_samples // 3)),
        ResearchDomain.HEALTHCARE_MEDICAL: min(4, max(2, n_samples // 2)),
        ResearchDomain.GENERAL_TOPIC: min(3, max(2, n_samples // 3))
    }
    
    preferred_k = domain_preferences.get(domain, min(3, max(2, n_samples // 3)))
    
    # Validate with silhouette score if possible
    if n_samples >= preferred_k:
        try:
            km = KMeans(n_clusters=preferred_k, n_init="auto", random_state=42)
            labels = km.fit_predict(X)
            if len(set(labels)) > 1:
                score = silhouette_score(X, labels)
                if score > 0.2:  # Reasonable clustering quality
                    return preferred_k
        except:
            pass
    
    return max(2, min(preferred_k, n_samples - 1))

def table_node(state: RState) -> RState:
    """Enhanced table generation with domain-specific columns"""
    if state.get("error"):
        return state
        
    prompt_manager = PromptTemplateManager()
    config = state["config"]
    
    # Generate domain-specific table header
    header = "| " + " | ".join(config.analysis_dimensions) + " |\n"
    header += "|" + "---|" * len(config.analysis_dimensions) + "\n"
    
    rows = []
    for s in state["sources"]:
        table_prompt = prompt_manager.generate_table_prompt(
            s["summary"], state["domain"]
        )
        
        cell = llm(table_prompt)
        if not cell.startswith("|"):
            # Fallback row
            fallback_values = ["Unknown"] * len(config.analysis_dimensions)
            fallback_values[0] = s["title"][:30]  # First column gets title
            cell = "| " + " | ".join(fallback_values) + " |"
        
        rows.append(cell)

    state["table_md"] = header + "\n".join(rows)
    return state

def report_node(state: RState) -> RState:
    """Enhanced report generation with domain-specific structure"""
    if state.get("error"):
        state["report_md"] = f"# Research Report Error\n\n**Error:** {state['error']}\n\nPlease try again."
        return state
    
    prompt_manager = PromptTemplateManager()
    
    # Generate cluster overview
    cluster_overview = "\n".join([
        f"- **{info['label']}**: {len(info['members'])} sources" 
        for cid, info in state["clusters"].items() if info['members']
    ])
    
    # Generate domain-specific report
    report_prompt = prompt_manager.generate_report_prompt(
        cluster_overview, state["table_md"], state["domain"], state["original_query"]
    )
    
    report_md = llm(report_prompt)
    
    # Add methodology and sources appendix
    methodology = f"""
---

## Methodology
- **Research Domain**: {state['domain'].value.replace('_', ' ').title()}
- **Sources Analyzed**: {len(state['sources'])}
- **Clustering Approach**: {state['config'].cluster_approach.replace('_', ' ').title()}
- **Search Strategy**: Enhanced query with domain-specific terms

## Sources"""
    
    for i, s in enumerate(state["sources"]):
        cluster_label = state["clusters"].get(s.get("cluster", 0), {}).get("label", "Other")
        methodology += f"\n{i+1}. **{cluster_label}**: [{s['title']}]({s['url']}) (Relevance: {s['relevance_score']:.2f})"
    
    state["report_md"] = report_md + methodology
    return state

def critic_node(state: RState) -> RState:
    """Enhanced critique with domain-specific quality criteria"""
    if state.get("error"):
        state["critique"] = "NEEDS_MORE - Error in data collection"
        return state
    
    prompt_manager = PromptTemplateManager()
    critique_prompt = prompt_manager.generate_critique_prompt(
        state["report_md"], state["domain"]
    )
    
    critique = llm(critique_prompt)
    state["critique"] = critique
    return state

def gate(state: RState) -> Literal["Loop", "Ship"]:
    """Enhanced gate with domain-aware retry logic"""
    if state.get("error"):
        return "Ship"
        
    critique = state.get("critique", "")
    needs_more = "NEEDS_MORE" in critique.upper()
    
    if needs_more and state["loop_count"] < 1:
        state["loop_count"] += 1
        # Domain-specific query enhancement for retry
        domain_specific_terms = {
            ResearchDomain.BUSINESS_COMPETITIVE: "market analysis OR industry report OR competitive intelligence",
            ResearchDomain.TECHNOLOGY_TRENDS: "technology review OR innovation report OR tech analysis",
            ResearchDomain.ACADEMIC_RESEARCH: "literature review OR academic study OR research paper",
            ResearchDomain.HEALTHCARE_MEDICAL: "clinical study OR medical research OR health analysis",
            ResearchDomain.GENERAL_TOPIC: "comprehensive analysis OR detailed report"
        }
        
        enhancement = domain_specific_terms.get(state["domain"], "analysis OR report")
        state["enhanced_query"] = f"{state['original_query']} {enhancement}"
        
        logger.info("Quality insufficient, retrying with enhanced search...")
        return "Loop"
    
    return "Ship"

# --- Build Enhanced Graph ---
def create_dynamic_app():
    graph = StateGraph(RState)
    
    # Add all agents including the new initialization agent
    graph.add_node("Initialize", initialize_research_node)
    graph.add_node("Collect", collect_node)
    graph.add_node("Summarize", summarize_node)
    graph.add_node("Cluster", cluster_node)
    graph.add_node("Table", table_node)
    graph.add_node("Report", report_node)
    graph.add_node("Critic", critic_node)

    # Enhanced workflow
    graph.set_entry_point("Initialize")
    graph.add_edge("Initialize", "Collect")
    graph.add_edge("Collect", "Summarize")
    graph.add_edge("Summarize", "Cluster")
    graph.add_edge("Cluster", "Table")
    graph.add_edge("Table", "Report")
    graph.add_edge("Report", "Critic")

    graph.add_conditional_edges("Critic", gate, {"Loop": "Collect", "Ship": END})
    return graph.compile()

# --- Enhanced Runner ---
def run_dynamic_research(query: str, model: str = "llama3"):
    """Run the enhanced dynamic research assistant"""
    print(f"🔍 Starting dynamic research on: {query}")
    
    state: RState = {
        "original_query": query,
        "enhanced_query": "",
        "domain": ResearchDomain.GENERAL_TOPIC,
        "sources": [],
        "clusters": {},
        "table_md": "",
        "report_md": "",
        "critique": "",
        "loop_count": 0,
        "error": None,
        "config": None
    }
    
    app = create_dynamic_app()
    
    try:
        final = app.invoke(state)
        
        # Generate domain-specific filename
        domain_name = final["domain"].value
        safe_query = re.sub(r'[^\w\s-]', '', query)[:50]
        safe_query = re.sub(r'[-\s]+', '_', safe_query)
        output_file = f"report_{domain_name}_{safe_query}.md"
        
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(final["report_md"])
        
        print(f"✅ Dynamic analysis complete! Report saved to {output_file}")
        print(f"📊 Domain: {final['domain'].value}")
        print(f"📚 Processed {len(final['sources'])} sources across {len(final['clusters'])} clusters")
        
        if final.get("error"):
            print(f"⚠️  Warning: {final['error']}")
            
    except Exception as e:
        logger.error(f"Application failed: {e}")
        print(f"❌ Research failed: {e}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python dynamic_research_assistant.py 'your research query here'")
        print("\nExample queries:")
        print("- 'competitive analysis of AI image generation tools'")
        print("- 'recent advances in quantum computing technology'") 
        print("- 'literature review on remote work productivity'")
        print("- 'market analysis of electric vehicle charging infrastructure'")
        exit(1)
    
    query = " ".join(sys.argv[1:])
    run_dynamic_research(query)