# streamlit_research_ui.py
# Beautiful Streamlit UI for the Dynamic Research Assistant

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
            st.image("https://via.placeholder.com/200x100/667eea/ffffff?text=Research+AI", width=200)
            
            st.markdown("## 🎯 Research Settings")
            
            # Model selection
            model = st.selectbox(
                "🤖 Select LLM Model",
                ["llama3", "llama3:8b", "llama3:70b", "mistral", "codellama"],
                help="Choose your local Ollama model"
            )
            
            # Advanced settings expander
            with st.expander("⚙️ Advanced Settings"):
                max_sources = st.slider("Maximum Sources", 5, 20, 8)
                enable_clustering = st.checkbox("Enable Smart Clustering", True)
                quality_threshold = st.slider("Quality Threshold", 0.1, 1.0, 0.3)
                
            st.markdown("---")
            
            # Domain examples
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
                    for query in queries:
                        if st.button(f"📝 {query}", key=f"example_{query}"):
                            st.session_state.example_query = query
                            st.rerun()
                            
            st.markdown("---")
            
            # Research history
            if st.session_state.research_history:
                st.markdown("## 📚 Recent Research")
                for i, item in enumerate(st.session_state.research_history[-5:]):
                    with st.expander(f"🔍 {item['query'][:30]}..."):
                        st.write(f"**Domain:** {item['domain']}")
                        st.write(f"**Date:** {item['timestamp']}")
                        if st.button(f"View Report", key=f"history_{i}"):
                            st.session_state.research_results = item
                            st.rerun()
                            
        return model, max_sources, enable_clustering, quality_threshold
    
    def render_input_section(self):
        """Render the query input section"""
        st.markdown("## 🔍 Research Query")
        
        col1, col2 = st.columns([4, 1])
        
        with col1:
            # Check if example query was selected
            default_query = ""
            if hasattr(st.session_state, 'example_query'):
                default_query = st.session_state.example_query
                del st.session_state.example_query
                
            query = st.text_input(
                "Enter your research topic:",
                value=default_query,
                placeholder="e.g., competitive analysis of AI image generation startups",
                help="Be specific about what you want to research"
            )
            
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            research_button = st.button(
                "🚀 Start Research", 
                type="primary",
                disabled=st.session_state.is_researching,
                use_container_width=True
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
    
    def run_research_async(self, query, model, settings):
        """Run research in background"""
        try:
            st.session_state.is_researching = True
            
            # Create progress tracking
            progress_placeholder = st.empty()
            status_placeholder = st.empty()
            
            # Initialize state
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
                "config": None
            }
            
            # Create app and run research
            app = create_dynamic_app()
            
            # Simulate progress updates (in real implementation, you'd hook into the actual pipeline)
            progress_steps = [
                ("🔍 Initializing research...", 10),
                ("🌐 Collecting sources...", 30),
                ("📝 Summarizing content...", 50),
                ("🧠 Clustering insights...", 70),
                ("📊 Creating analysis...", 85),
                ("📄 Generating report...", 95),
                ("✅ Finalizing results...", 100)
            ]
            
            for step, progress in progress_steps:
                progress_placeholder.progress(progress)
                status_placeholder.info(step)
                time.sleep(1)  # Simulate work
                
            # Run the actual research (this would be your real function call)
            final_state = app.invoke(state)
            
            # Store results
            result = {
                'query': query,
                'domain': final_state.get('domain', 'unknown'),
                'sources': final_state.get('sources', []),
                'clusters': final_state.get('clusters', {}),
                'table_md': final_state.get('table_md', ''),
                'report_md': final_state.get('report_md', ''),
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'model_used': model
            }
            
            st.session_state.research_results = result
            st.session_state.research_history.append(result)
            
            progress_placeholder.progress(100)
            status_placeholder.success("✅ Research completed successfully!")
            
        except Exception as e:
            st.error(f"Research failed: {str(e)}")
        finally:
            st.session_state.is_researching = False
            st.rerun()
    
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
            use_container_width=True,
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
                st.plotly_chart(fig_pie, use_container_width=True)
        
        # Relevance score distribution
        with col2:
            relevance_scores = [s.get('relevance_score', 0) for s in results['sources']]
            fig_hist = px.histogram(
                x=relevance_scores,
                title='Source Relevance Score Distribution',
                labels={'x': 'Relevance Score', 'y': 'Count'}
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        
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
        fig_bar.update_xaxis(tickangle=45)
        st.plotly_chart(fig_bar, use_container_width=True)
    
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
    model, max_sources, enable_clustering, quality_threshold = app.render_sidebar()
    
    # Main content area
    query, research_button = app.render_input_section()
    
    # Handle research execution
    if research_button and query and not st.session_state.is_researching:
        settings = {
            'max_sources': max_sources,
            'enable_clustering': enable_clustering,
            'quality_threshold': quality_threshold
        }
        
        with st.container():
            st.markdown("## 🔄 Research in Progress")
            app.run_research_async(query, model, settings)
    
    # Display results if available
    if st.session_state.research_results:
        app.render_results_dashboard(st.session_state.research_results)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
        🔍 AI Research Assistant | Powered by Streamlit & Local LLMs
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()