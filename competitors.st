import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def load_and_process_data(uploaded_file):
    """Load CSV and calculate AI metrics"""
    df = pd.read_csv(uploaded_file)
    
    # Fill NaN values with 0 for calculation
    df = df.fillna(0)
    
    # Calculate AI total (sum of SGE, Perplexity, and GPT queries covered)
    df['ai_total'] = (
        df['sge_queries_covered'].astype(float) + 
        df['perplexity_queries_covered'].astype(float) + 
        df['gpt_queries_covered'].astype(float)
    )
    
    # Convert serp_queries_covered to numeric for comparison
    df['serp_queries_covered'] = pd.to_numeric(df['serp_queries_covered'], errors='coerce').fillna(0)
    
    return df

def get_top_ai_domains(df, top_n=20):
    """Get top domains by AI total"""
    return df.nlargest(top_n, 'ai_total')[['domain', 'ai_total', 'sge_queries_covered', 
                                           'perplexity_queries_covered', 'gpt_queries_covered', 
                                           'serp_queries_covered']]

def get_ai_popular_not_serp(df, ai_threshold=1, serp_threshold=1):
    """Get domains popular in AI but not in SERP"""
    ai_popular = df[df['ai_total'] >= ai_threshold].copy()
    ai_not_serp = ai_popular[ai_popular['serp_queries_covered'] < serp_threshold]
    return ai_not_serp[['domain', 'ai_total', 'serp_queries_covered', 
                        'sge_queries_covered', 'perplexity_queries_covered', 'gpt_queries_covered']]

def get_serp_popular_not_ai(df, ai_threshold=1, serp_threshold=1):
    """Get domains popular in SERP but not in AI"""
    serp_popular = df[df['serp_queries_covered'] >= serp_threshold].copy()
    serp_not_ai = serp_popular[serp_popular['ai_total'] < ai_threshold]
    return serp_not_ai[['domain', 'serp_queries_covered', 'ai_total', 
                        'sge_queries_covered', 'perplexity_queries_covered', 'gpt_queries_covered']]

def create_ai_breakdown_chart(df_top):
    """Create a stacked bar chart showing AI breakdown"""
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='SGE Queries',
        x=df_top['domain'],
        y=df_top['sge_queries_covered'],
        marker_color='#1f77b4'
    ))
    
    fig.add_trace(go.Bar(
        name='Perplexity Queries',
        x=df_top['domain'],
        y=df_top['perplexity_queries_covered'],
        marker_color='#ff7f0e'
    ))
    
    fig.add_trace(go.Bar(
        name='GPT Queries',
        x=df_top['domain'],
        y=df_top['gpt_queries_covered'],
        marker_color='#2ca02c'
    ))
    
    fig.update_layout(
        title='AI Queries Breakdown by Domain (Top Domains)',
        xaxis_title='Domain',
        yaxis_title='Number of Queries',
        barmode='stack',
        xaxis_tickangle=-45,
        height=600
    )
    
    return fig

def create_ai_vs_serp_scatter(df):
    """Create scatter plot of AI total vs SERP queries"""
    fig = px.scatter(
        df, 
        x='serp_queries_covered', 
        y='ai_total',
        hover_data=['domain'],
        title='AI Total vs SERP Queries Coverage',
        labels={
            'serp_queries_covered': 'SERP Queries Covered',
            'ai_total': 'AI Total Queries'
        }
    )
    
    # Add diagonal line for reference
    max_val = max(df['serp_queries_covered'].max(), df['ai_total'].max())
    fig.add_shape(
        type="line",
        x0=0, y0=0, x1=max_val, y1=max_val,
        line=dict(color="red", width=2, dash="dash"),
    )
    
    fig.update_layout(height=500)
    return fig

def main():
    st.set_page_config(page_title="AI Domain Influence Analyzer", layout="wide")
    
    st.title("🤖 AI Domain Influence Analyzer")
    st.markdown("Analyze domain performance across AI platforms (SGE, Perplexity, GPT) vs traditional SERP")
    
    # File upload
    uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            # Load and process data
            df = load_and_process_data(uploaded_file)
            
            # Sidebar for parameters
            st.sidebar.header("Analysis Parameters")
            top_n = st.sidebar.slider("Number of top domains to show", 5, 50, 20)
            ai_threshold = st.sidebar.number_input("AI popularity threshold", min_value=0, value=1)
            serp_threshold = st.sidebar.number_input("SERP popularity threshold", min_value=0, value=1)
            
            # Data overview
            st.header("📊 Data Overview")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Domains", len(df))
            with col2:
                st.metric("Domains in AI", len(df[df['ai_total'] > 0]))
            with col3:
                st.metric("Domains in SERP", len(df[df['serp_queries_covered'] > 0]))
            with col4:
                st.metric("Max AI Total", int(df['ai_total'].max()))
            
            # Top AI domains
            st.header("🏆 Top Influential Domains by AI Total")
            top_ai_domains = get_top_ai_domains(df, top_n)
            
            # Display table
            st.dataframe(
                top_ai_domains.round(0),
                use_container_width=True,
                hide_index=True
            )
            
            # AI breakdown chart
            st.subheader("AI Queries Breakdown")
            ai_chart = create_ai_breakdown_chart(top_ai_domains.head(15))  # Limit to top 15 for readability
            st.plotly_chart(ai_chart, use_container_width=True)
            
            # Two-column layout for comparisons
            col1, col2 = st.columns(2)
            
            with col1:
                st.header("🔥 Popular in AI, Not in SERP")
                ai_not_serp = get_ai_popular_not_serp(df, ai_threshold, serp_threshold)
                
                if len(ai_not_serp) > 0:
                    st.write(f"Found {len(ai_not_serp)} domains popular in AI but not in SERP:")
                    st.dataframe(
                        ai_not_serp.round(0),
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    # Download button
                    csv_ai_not_serp = ai_not_serp.to_csv(index=False)
                    st.download_button(
                        label="Download AI-only popular domains",
                        data=csv_ai_not_serp,
                        file_name="ai_popular_not_serp.csv",
                        mime="text/csv"
                    )
                else:
                    st.info("No domains found that are popular in AI but not in SERP with current thresholds.")
            
            with col2:
                st.header("📈 Popular in SERP, Not in AI")
                serp_not_ai = get_serp_popular_not_ai(df, ai_threshold, serp_threshold)
                
                if len(serp_not_ai) > 0:
                    st.write(f"Found {len(serp_not_ai)} domains popular in SERP but not in AI:")
                    st.dataframe(
                        serp_not_ai.round(0),
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    # Download button
                    csv_serp_not_ai = serp_not_ai.to_csv(index=False)
                    st.download_button(
                        label="Download SERP-only popular domains",
                        data=csv_serp_not_ai,
                        file_name="serp_popular_not_ai.csv",
                        mime="text/csv"
                    )
                else:
                    st.info("No domains found that are popular in SERP but not in AI with current thresholds.")
            
            # Scatter plot comparison
            st.header("🎯 AI vs SERP Performance Comparison")
            scatter_chart = create_ai_vs_serp_scatter(df)
            st.plotly_chart(scatter_chart, use_container_width=True)
            
            # Summary statistics
            st.header("📋 Summary Statistics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("AI Platform Performance")
                ai_stats = pd.DataFrame({
                    'Platform': ['SGE', 'Perplexity', 'GPT'],
                    'Total Queries': [
                        df['sge_queries_covered'].sum(),
                        df['perplexity_queries_covered'].sum(),
                        df['gpt_queries_covered'].sum()
                    ],
                    'Domains Covered': [
                        len(df[df['sge_queries_covered'] > 0]),
                        len(df[df['perplexity_queries_covered'] > 0]),
                        len(df[df['gpt_queries_covered'] > 0])
                    ]
                })
                st.dataframe(ai_stats, hide_index=True)
            
            with col2:
                st.subheader("Platform Overlap Analysis")
                only_ai = len(df[(df['ai_total'] > 0) & (df['serp_queries_covered'] == 0)])
                only_serp = len(df[(df['serp_queries_covered'] > 0) & (df['ai_total'] == 0)])
                both = len(df[(df['ai_total'] > 0) & (df['serp_queries_covered'] > 0)])
                
                overlap_stats = pd.DataFrame({
                    'Category': ['AI Only', 'SERP Only', 'Both AI & SERP'],
                    'Count': [only_ai, only_serp, both]
                })
                st.dataframe(overlap_stats, hide_index=True)
            
            # Download full processed data
            st.header("💾 Download Processed Data")
            csv_full = df.to_csv(index=False)
            st.download_button(
                label="Download full processed dataset",
                data=csv_full,
                file_name="processed_domain_analysis.csv",
                mime="text/csv"
            )
            
        except Exception as e:
            st.error(f"Error processing the file: {str(e)}")
            st.info("Please make sure your CSV has the required columns: domain, sge_queries_covered, perplexity_queries_covered, gpt_queries_covered, serp_queries_covered")
    
    else:
        st.info("👆 Please upload a CSV file to begin analysis")
        
        # Show expected format
        st.subheader("Expected CSV Format")
        sample_data = {
            'domain': ['example.com', 'another.com'],
            'sge_queries_covered': [5, 2],
            'serp_queries_covered': [10, 0],
            'perplexity_queries_covered': [3, 1],
            'gpt_queries_covered': [2, 4]
        }
        sample_df = pd.DataFrame(sample_data)
        st.dataframe(sample_df, hide_index=True)

if __name__ == "__main__":
    main()
