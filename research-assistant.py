# research-assistant.py
# Fixed Streamlit UI for the Dynamic Research Assistant

import streamlit as st
import asyncio
import threading
import time
import io
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import base64
from dynamic_research_assistant import get_installed_ollama_models

# Import your research assistant
from dynamic_research_assistant import (
    run_dynamic_research, 
    ResearchDomain, 
    PromptTemplateManager,
    create_dynamic_app,
    RState
)

# Configure Streamlit page
st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #667eea;
        margin: 0.5rem 0;
    }
    
    .domain-tag {
        background: #667eea;
        color: white;
        padding: 0.2rem 0.8rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: bold;
        display: inline-block;
        margin: 0.2rem;
    }
    
    .source-card {
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    .cluster-header {
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        font-weight: bold;
        margin: 1rem 0 0.5rem 0;
    }
    
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    .research-stats {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class StreamlitResearchApp:
    def __init__(self):
        self.prompt_manager = PromptTemplateManager()
        
        # Initialize session state
        if 'research_results' not in st.session_state:
            st.session_state.research_results = None
        if 'research_history' not in st.session_state:
            st.session_state.research_history = []
        if 'is_researching' not in st.session_state:
            st.session_state.is_researching = False
        if 'current_progress' not in st.session_state:
            st.session_state.current_progress = 0
        if 'current_status' not in st.session_state:
            st.session_state.current_status = ""
            
    def render_header(self):
        """Render the main header"""
        st.markdown('<h1 class="main-header">🔍 AI Research Assistant</h1>', unsafe_allow_html=True)
        st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem; color: #666;">
            Generate comprehensive research reports on any topic using AI-powered analysis
        </div>
        """, unsafe_allow_html=True)
        
    def render_sidebar(self):
        """Render the sidebar with controls and settings"""
        with st.sidebar:
            
            st.markdown("## 🎯 Research Settings")
            
            # Model selection
            st.markdown("#### 🤖 Model Selection")
            model_type = st.radio(
                "Provider:",
                ["Ollama (Local)", "OpenAI (API)"],
                help="Choose between local Ollama or OpenAI API models"
            )
            
            api_key = None  # Initialize api_key variable
            
            if model_type == "Ollama (Local)":
                # Get installed Ollama models dynamically
                installed_models = get_installed_ollama_models()
                
                if not installed_models:
                    st.error("No Ollama models found. Please install Ollama and download models.")
                    st.code("ollama pull llama3", language="bash")
                    model = "llama3"  # Default fallback
                else:
                    model = st.selectbox(
                        "Ollama Model:",
                        installed_models,
                        help=f"Found {len(installed_models)} installed Ollama models"
                    )
                    
                    # Show model info
                    with st.expander("ℹ️ Model Info"):
                        st.write(f"**Selected Model:** {model}")
                        st.write("**Available Models:**")
                        for m in installed_models:
                            st.write(f"• {m}")
            else:
                # OpenAI models
                model = st.selectbox(
                    "OpenAI Model:",
                    ["gpt-5-nano-2025-08-07", "gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
                    help="Choose OpenAI model (requires API key)"
                )
                
                # API key input for OpenAI
                api_key = st.text_input(
                    "OpenAI API Key:",
                    type="password",
                    help="Enter your OpenAI API key or set OPENAI_API_KEY environment variable",
                    value=st.session_state.get('openai_api_key', '')
                )
                
                # Store API key in session state
                if api_key:
                    st.session_state.openai_api_key = api_key
                
                # Validate OpenAI setup
                import os
                if not api_key and not os.getenv("OPENAI_API_KEY"):
                    st.warning("⚠️ OpenAI API key required. Please enter your API key above or set OPENAI_API_KEY environment variable.")
                    
                    # Show instructions
                    with st.expander("📚 How to get an API key"):
                        st.markdown("""
                        1. Go to [OpenAI Platform](https://platform.openai.com/)
                        2. Sign up or log in
                        3. Navigate to API Keys section
                        4. Create a new API key
                        5. Copy and paste it above
                        """)
            
            # Advanced settings expander
            with st.expander("⚙️ Advanced Settings"):
                max_sources = st.slider("Maximum Sources", 5, 20, 8)
                enable_clustering = st.checkbox("Enable Smart Clustering", True)
                quality_threshold = st.slider("Quality Threshold", 0.1, 1.0, 0.3)
                
            st.markdown("---")
            
            # Domain examples (keep your existing code)
            st.markdown("## 💡 Example Queries")
            example_queries = {
                "🏢 Business": [
                    "competitive analysis of fintech startups 2024",
                    "market analysis of electric vehicle companies"
                ],
                "🚀 Technology": [
                    "recent advances in quantum computing",
                    "AI image generation tools comparison"
                ],
                "📚 Academic": [
                    "literature review on remote work productivity",
                    "research on climate change adaptation"
                ],
                "🏥 Healthcare": [
                    "telemedicine adoption trends",
                    "personalized medicine developments"
                ]
            }
            
            for category, queries in example_queries.items():
                with st.expander(category):
                    for i, query in enumerate(queries):
                        button_key = f"example_{category}_{i}"
                        if st.button(query, key=button_key, width='stretch'):
                            st.session_state.current_query = query
                            st.session_state.example_query = query
                            if 'research_results' in st.session_state:
                                st.session_state.research_results = None
                            st.rerun()
                            
            st.markdown("---")
            
            # Research history (keep your existing code)
            if st.session_state.research_history:
                st.markdown("## 📚 Recent Research")
                for i, item in enumerate(st.session_state.research_history[-5:]):
                    with st.expander(f"🔍 {item['query'][:30]}..."):
                        st.write(f"**Domain:** {item['domain']}")
                        st.write(f"**Date:** {item['timestamp']}")
                        if st.button(f"View Report", key=f"history_{i}"):
                            st.session_state.research_results = item
                            st.rerun()
                            
        return model, max_sources, enable_clustering, quality_threshold, api_key

    
    def render_input_section(self):
        """Render the query input section"""
        st.markdown("## 🔍 Research Query")
        
        col1, col2 = st.columns([4, 1])
        
        with col1:
            # Initialize query input value from session state
            if 'current_query' not in st.session_state:
                st.session_state.current_query = ""
            
            # Check if example query was selected
            if hasattr(st.session_state, 'example_query'):
                st.session_state.current_query = st.session_state.example_query
                del st.session_state.example_query
                
            query = st.text_input(
                "Enter your research topic:",
                value=st.session_state.current_query,
                placeholder="e.g., competitive analysis of AI image generation startups",
                help="Be specific about what you want to research",
                key="query_input"
            )
            
            # Update session state with current input
            st.session_state.current_query = query
            
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            research_button = st.button(
                "🚀 Start Research", 
                type="primary",
                disabled=st.session_state.is_researching or not query.strip(),
                width='stretch'
            )
            
        # Auto-classify domain preview
        if query:
            predicted_domain = self.prompt_manager.classify_research_domain(query)
            st.markdown(f"""
            <div style="margin-top: 1rem;">
                <span style="color: #666;">Detected Domain:</span>
                <span class="domain-tag">{predicted_domain.value.replace('_', ' ').title()}</span>
            </div>
            """, unsafe_allow_html=True)
            
        return query, research_button
    
    def update_progress(self, progress, status):
        """Update progress bar and status"""
        st.session_state.current_progress = progress
        st.session_state.current_status = status
    
    def run_research_async(self, query, model, settings, api_key=None):
        """Run research with detailed progress tracking"""
        try:
            st.session_state.is_researching = True
            
            # Create progress tracking containers
            progress_container = st.container()
            
            with progress_container:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Step 1: Initialize
                self.update_progress(5, "🔧 Initializing research pipeline...")
                progress_bar.progress(5)
                status_text.info("🔧 Initializing research pipeline...")
                time.sleep(1)
                
                # Initialize state for research pipeline
                state = {
                "original_query": query,
                "enhanced_query": "",
                "domain": None,
                "sources": [],
                "clusters": {},
                "table_md": "",
                "report_md": "",
                "critique": "",
                "loop_count": 0,
                "error": None,
                "config": None,
                "api_key": api_key  # Add API key to initial state
            }
                
                # Import and create the research app
                from dynamic_research_assistant import create_dynamic_app
                app = create_dynamic_app()
                
                # Step 2: Domain Classification
                self.update_progress(10, "🎯 Classifying research domain...")
                progress_bar.progress(10)
                status_text.info("🎯 Classifying research domain...")
                time.sleep(0.5)
                
                # Step 3: Enhanced Query Generation
                self.update_progress(15, "🔍 Enhancing search query...")
                progress_bar.progress(15)
                status_text.info("🔍 Enhancing search query...")
                time.sleep(0.5)
                
                # Step 4: Web Search & Collection
                self.update_progress(25, "🌐 Searching and collecting sources...")
                progress_bar.progress(25)
                status_text.info("🌐 Searching and collecting sources...")
                
                # Run the actual research with progress updates
                try:
                    final_state = self.run_research_with_progress(
                        app, state, progress_bar, status_text, settings, model, api_key
                    )
                except Exception as e:
                    st.error(f"Research pipeline failed: {str(e)}")
                    return
                
                # Final processing
                self.update_progress(95, "📊 Finalizing results...")
                progress_bar.progress(95)
                status_text.info("📊 Finalizing results...")
                time.sleep(0.5)
                
                # Store results in the expected format
                result = {
                    'query': query,
                    'domain': final_state.get('domain', 'unknown').value if hasattr(final_state.get('domain', 'unknown'), 'value') else str(final_state.get('domain', 'unknown')),
                    'sources': final_state.get('sources', []),
                    'clusters': final_state.get('clusters', {}),
                    'table_md': final_state.get('table_md', ''),
                    'report_md': final_state.get('report_md', ''),
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'model_used': model,
                    'critique': final_state.get('critique', ''),
                    'settings_used': settings  # Store the settings that were used
                }
                
                # Store results BEFORE clearing progress
                st.session_state.research_results = result
                st.session_state.research_history.append(result)
                
                # Complete
                self.update_progress(100, "✅ Research completed successfully!")
                progress_bar.progress(100)
                status_text.success("✅ Research completed successfully!")
                
                # Keep success message visible for a moment
                time.sleep(2)
            
            # Clear progress indicators
            progress_container.empty()
            
            # Force a rerun to show results
            st.rerun()
                
        except Exception as e:
            st.error(f"Research failed: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
        finally:
            st.session_state.is_researching = False

    
    def run_research_with_progress(self, app, state, progress_bar, status_text, settings, model, api_key=None):
        """Run research pipeline with detailed progress updates"""
        
        # Pass settings and model to state
        state.update({
        'number_of_sources': settings['max_sources'],
        'quality_threshold': settings['quality_threshold'],
        'enable_clustering': settings['enable_clustering'],
        'model_name': model,
        'api_key': api_key  # Add API key to state
    })

        
        # Initialize node
        progress_bar.progress(20)
        status_text.info("🔧 Initializing research configuration...")
        from dynamic_research_assistant import initialize_research_node
        state = initialize_research_node(state)
        time.sleep(0.5)
        
        # Collect node
        progress_bar.progress(35)
        status_text.info("🌐 Collecting web sources...")
        from dynamic_research_assistant import collect_node
        state = collect_node(state)
        time.sleep(1)
        
        # Summarize node
        progress_bar.progress(50)
        status_text.info("📝 Generating AI summaries...")
        from dynamic_research_assistant import summarize_node
        state = summarize_node(state)
        time.sleep(1)
        
        # Cluster node (conditionally skip if disabled)
        if settings['enable_clustering']:
            progress_bar.progress(65)
            status_text.info("🧠 Clustering related content...")
            from dynamic_research_assistant import cluster_node
            state = cluster_node(state)
        else:
            progress_bar.progress(65)
            status_text.info("⏭️ Skipping clustering (disabled)...")
            # Create single cluster for all sources
            state["clusters"] = {0: {"label": "All Sources", "members": list(range(len(state["sources"])))}}
            for i in range(len(state["sources"])):
                state["sources"][i]["cluster"] = 0
        time.sleep(0.5)
        
        # Table node
        progress_bar.progress(75)
        status_text.info("📊 Creating analysis table...")
        from dynamic_research_assistant import table_node
        state = table_node(state)
        time.sleep(0.5)
        
        # Report node
        progress_bar.progress(85)
        status_text.info("📄 Writing research report...")
        from dynamic_research_assistant import report_node
        state = report_node(state)
        time.sleep(1)
        
        # Critic node
        progress_bar.progress(90)
        status_text.info("🔍 Quality assessment...")
        from dynamic_research_assistant import critic_node
        state = critic_node(state)
        time.sleep(0.5)
        
        return state
    
    def render_results_dashboard(self, results):
        """Render comprehensive results dashboard"""
        st.markdown("## 📊 Research Results Dashboard")
        
        # Key metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="margin:0; color:#667eea;">📚 Sources</h3>
                <h2 style="margin:0;">{len(results['sources'])}</h2>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="margin:0; color:#667eea;">🧠 Clusters</h3>
                <h2 style="margin:0;">{len(results['clusters'])}</h2>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="margin:0; color:#667eea;">🎯 Domain</h3>
                <h2 style="margin:0; font-size:1rem;">{results['domain'].replace('_', ' ').title()}</h2>
            </div>
            """, unsafe_allow_html=True)
            
        with col4:
            # Show settings if available
            if 'settings_used' in results:
                settings = results['settings_used']
                st.markdown(f"""
                <div class="metric-card">
                    <h3 style="margin:0; color:#667eea;">⚙️ Settings</h3>
                    <p style="margin:0; font-size:0.8rem;">
                        Sources: {settings['max_sources']}<br>
                        Clustering: {'On' if settings['enable_clustering'] else 'Off'}<br>
                        Quality: {settings['quality_threshold']}
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="metric-card">
                    <h3 style="margin:0; color:#667eea;">⏰ Generated</h3>
                    <h2 style="margin:0; font-size:1rem;">{results['timestamp']}</h2>
                </div>
                """, unsafe_allow_html=True)
        
        # Create tabs for different views
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📄 Report", "📊 Analysis", "🔗 Sources", "📈 Visualizations", "💾 Export"])
        
        with tab1:
            self.render_report_tab(results)
            
        with tab2:
            self.render_analysis_tab(results)
            
        with tab3:
            self.render_sources_tab(results)
            
        with tab4:
            self.render_visualizations_tab(results)
            
        with tab5:
            self.render_export_tab(results)
    
    def render_report_tab(self, results):
        """Render the main report"""
        st.markdown("### 📄 Research Report")
        
        # Add report quality indicators
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown(results['report_md'])
            
        with col2:
            st.markdown("#### 📊 Report Stats")
            report_text = results['report_md']
            word_count = len(report_text.split())
            char_count = len(report_text)
            
            st.metric("Word Count", word_count)
            st.metric("Character Count", char_count)
            st.metric("Estimated Read Time", f"{word_count // 200 + 1} min")
            
            # Quality indicators
            st.markdown("#### ✨ Quality Indicators")
            quality_score = min(100, (len(results['sources']) * 10) + (len(results['clusters']) * 15) + min(50, word_count // 10))
            st.progress(quality_score / 100)
            st.write(f"Quality Score: {quality_score}/100")
    
    def render_analysis_tab(self, results):
        """Render detailed analysis"""
        st.markdown("### 📊 Comparative Analysis")
        
        if results['table_md']:
            st.markdown(results['table_md'])
        else:
            st.info("No comparative table generated for this research.")
            
        # Cluster analysis
        if results['clusters']:
            st.markdown("### 🧠 Cluster Analysis")
            
            for cluster_id, cluster_info in results['clusters'].items():
                st.markdown(f"""
                <div class="cluster-header">
                    📁 {cluster_info['label']} ({len(cluster_info['members'])} sources)
                </div>
                """, unsafe_allow_html=True)
                
                # Show cluster members
                for member_idx in cluster_info['members']:
                    if member_idx < len(results['sources']):
                        source = results['sources'][member_idx]
                        with st.expander(f"🔗 {source['title'][:60]}..."):
                            st.write(f"**URL:** {source['url']}")
                            st.write(f"**Relevance Score:** {source.get('relevance_score', 0):.2f}")
                            if source.get('summary'):
                                st.write("**Summary:**")
                                st.write(source['summary'])
    
    def render_sources_tab(self, results):
        """Render sources information"""
        st.markdown("### 🔗 Source Analysis")
        
        if not results['sources']:
            st.warning("No sources found for this research.")
            return
            
        # Sources overview
        sources_df = pd.DataFrame([
            {
                'Title': s['title'][:50] + '...' if len(s['title']) > 50 else s['title'],
                'URL': s['url'],
                'Cluster': results['clusters'].get(s.get('cluster', 0), {}).get('label', 'Unknown'),
                'Relevance': s.get('relevance_score', 0),
                'Content Length': len(s.get('content', ''))
            }
            for s in results['sources']
        ])
        
        st.dataframe(
            sources_df,
            width='stretch',
            column_config={
                "URL": st.column_config.LinkColumn("URL"),
                "Relevance": st.column_config.ProgressColumn("Relevance Score", min_value=0, max_value=10),
            }
        )
        
        # Detailed source cards
        st.markdown("### 📚 Detailed Sources")
        for i, source in enumerate(results['sources']):
            with st.expander(f"📄 Source {i+1}: {source['title']}"):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.write(f"**URL:** {source['url']}")
                    if source.get('summary'):
                        st.write("**AI Summary:**")
                        st.write(source['summary'])
                        
                with col2:
                    cluster_name = results['clusters'].get(source.get('cluster', 0), {}).get('label', 'Unknown')
                    st.write(f"**Cluster:** {cluster_name}")
                    st.write(f"**Relevance:** {source.get('relevance_score', 0):.2f}")
                    st.write(f"**Content Length:** {len(source.get('content', ''))} chars")
    
    def render_visualizations_tab(self, results):
        """Render data visualizations"""
        st.markdown("### 📈 Research Visualizations")
        
        if not results['sources']:
            st.warning("No data available for visualization.")
            return
            
        col1, col2 = st.columns(2)
        
        # Cluster distribution pie chart
        with col1:
            cluster_data = []
            for cluster_id, cluster_info in results['clusters'].items():
                cluster_data.append({
                    'Cluster': cluster_info['label'],
                    'Count': len(cluster_info['members'])
                })
                
            if cluster_data:
                df_clusters = pd.DataFrame(cluster_data)
                fig_pie = px.pie(
                    df_clusters, 
                    values='Count', 
                    names='Cluster', 
                    title='Source Distribution by Cluster'
                )
                st.plotly_chart(fig_pie, width='stretch')
        
        # Relevance score distribution
        with col2:
            relevance_scores = [s.get('relevance_score', 0) for s in results['sources']]
            fig_hist = px.histogram(
                x=relevance_scores,
                title='Source Relevance Score Distribution',
                labels={'x': 'Relevance Score', 'y': 'Count'}
            )
            st.plotly_chart(fig_hist, width='stretch')
        
        # Content length analysis
        st.markdown("#### 📏 Content Analysis")
        content_lengths = [len(s.get('content', '')) for s in results['sources']]
        titles = [s['title'][:30] + '...' for s in results['sources']]
        
        fig_bar = px.bar(
            x=titles,
            y=content_lengths,
            title='Content Length by Source',
            labels={'x': 'Sources', 'y': 'Characters'}
        )
        fig_bar.update_xaxes(tickangle=45)
        st.plotly_chart(fig_bar, width='stretch')
    
    def render_export_tab(self, results):
        """Render export options"""
        st.markdown("### 💾 Export Research Results")
        
        col1, col2, col3 = st.columns(3)
        
        # Export as Markdown
        with col1:
            st.markdown("#### 📝 Markdown Report")
            st.download_button(
                label="📥 Download Markdown",
                data=results['report_md'],
                file_name=f"research_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown"
            )
        
        # Export as JSON
        with col2:
            st.markdown("#### 🔧 Raw Data (JSON)")
            import json
            json_data = json.dumps(results, indent=2, default=str)
            st.download_button(
                label="📥 Download JSON",
                data=json_data,
                file_name=f"research_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        # Export sources as CSV
        with col3:
            st.markdown("#### 📊 Sources (CSV)")
            sources_df = pd.DataFrame(results['sources'])
            csv_data = sources_df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv_data,
                file_name=f"research_sources_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        # Share options
        st.markdown("---")
        st.markdown("#### 🔗 Share Research")
        
        # Generate shareable link (in real app, you'd store this in a database)
        share_code = base64.b64encode(f"{results['query']}_{results['timestamp']}".encode()).decode()[:10]
        share_url = f"https://yourapp.com/shared/{share_code}"
        
        st.code(share_url, language="text")
        st.info("💡 Share this link to let others view your research results")

def main():
    """Main Streamlit application"""
    app = StreamlitResearchApp()
    
    # Render components
    app.render_header()
    
    # Get sidebar values including api_key
    try:
        model, max_sources, enable_clustering, quality_threshold, api_key = app.render_sidebar()
    except Exception as e:
        st.error(f"Error in sidebar configuration: {str(e)}")
        # Use default values if sidebar fails
        model, max_sources, enable_clustering, quality_threshold, api_key = "llama3", 8, True, 0.3, None
    
    # Main content area
    query, research_button = app.render_input_section()
    
    # Handle research execution
    if research_button and query and not st.session_state.is_researching:
        # Validate model/API key combination
        if model.startswith("gpt-") and not api_key:
            import os
            if not os.getenv("OPENAI_API_KEY"):
                st.error("❌ OpenAI API key required for GPT models. Please enter your API key in the sidebar.")
                return
        
        settings = {
            'max_sources': max_sources,
            'enable_clustering': enable_clustering,
            'quality_threshold': quality_threshold
        }
        
        st.markdown("## 🔄 Research in Progress")
        app.run_research_async(query, model, settings, api_key)
    
    # Always display results if available
    if st.session_state.research_results and not st.session_state.is_researching:
        app.render_results_dashboard(st.session_state.research_results)
    
    # Footer
    if not st.session_state.is_researching:
        st.markdown("---")
        st.markdown("""
        <div>
            🔍 
            <span  style="text-align: center; color: #666; padding: 2rem;">
            AI Research Assistant | Powered by Streamlit & Local LLMs
            </span>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()