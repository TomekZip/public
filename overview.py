import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter
import re
from urllib.parse import urlparse
import plotly.express as px
import plotly.graph_objects as go

def normalize_url(url):
    """Normalize URL by removing parameters and fragments"""
    try:
        parsed = urlparse(url.strip())
        normalized = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        if normalized.endswith('/') and len(parsed.path) > 1:
            normalized = normalized[:-1]
        return normalized.lower()
    except:
        return url.strip().lower()

def get_domain_from_url(url):
    """Extract domain from URL"""
    try:
        parsed = urlparse(url)
        return parsed.netloc.lower()
    except:
        return url.lower()

def extract_urls_from_text(text):
    """Extract URLs from text using regex"""
    if pd.isna(text) or text == '':
        return []
    
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    urls = re.findall(url_pattern, str(text))
    return [normalize_url(url) for url in urls]

def calculate_coverage_metrics(df):
    """Calculate AI total and SERP coverage metrics"""
    
    # Check required columns exist
    ai_columns = ['sge_queries_covered', 'perplexity_queries_covered', 'gpt_queries_covered']
    serp_column = 'serp_queries_covered'
    source_columns = ['gpt_sources', 'perplexity_sources', 'aio_response_sources']
    
    missing_cols = []
    for col in ai_columns + [serp_column] + source_columns:
        if col not in df.columns:
            missing_cols.append(col)
    
    if missing_cols:
        return None, f"Missing columns: {', '.join(missing_cols)}"
    
    # Calculate AI total for each row
    df['ai_total'] = df[ai_columns].sum(axis=1)
    
    # Extract domains from source columns and calculate weighted scores
    domain_ai_scores = {}
    domain_serp_scores = {}
    
    for idx, row in df.iterrows():
        ai_total = row['ai_total']
        serp_total = row['serp_queries_covered'] if not pd.isna(row['serp_queries_covered']) else 0
        
        # Get all domains from this row's sources
        all_domains = set()
        for col in source_columns:
            if not pd.isna(row[col]):
                urls = extract_urls_from_text(row[col])
                domains = [get_domain_from_url(url) for url in urls]
                all_domains.update(domains)
        
        # Add scores for each domain found in this row
        for domain in all_domains:
            if domain not in domain_ai_scores:
                domain_ai_scores[domain] = 0
            if domain not in domain_serp_scores:
                domain_serp_scores[domain] = 0
            
            domain_ai_scores[domain] += ai_total
            domain_serp_scores[domain] += serp_total
    
    return {
        'domain_ai_scores': domain_ai_scores,
        'domain_serp_scores': domain_serp_scores,
        'ai_columns': ai_columns,
        'total_queries': len(df)
    }, None

def main():
    st.set_page_config(
        page_title="AI vs SERP Domain Analysis",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 AI vs SERP Domain Influence Analysis")
    st.markdown("Analyze domain influence across AI platforms vs traditional SERP results")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Successfully loaded CSV with {len(df)} rows and {len(df.columns)} columns")
            
            # Calculate metrics
            result, error = calculate_coverage_metrics(df)
            
            if error:
                st.error(f"❌ {error}")
                st.write("Available columns:", list(df.columns))
                return
            
            domain_ai_scores = result['domain_ai_scores']
            domain_serp_scores = result['domain_serp_scores']
            ai_columns = result['ai_columns']
            total_queries = result['total_queries']
            
            if not domain_ai_scores:
                st.warning("⚠️ No domains found in source columns")
                return
            
            # Calculate summary stats
            total_ai_coverage = df[ai_columns].sum().sum()
            total_serp_coverage = df['serp_queries_covered'].sum()
            
            st.header("📊 Coverage Summary")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Queries", total_queries)
            with col2:
                st.metric("Total AI Coverage", f"{total_ai_coverage:,}")
            with col3:
                st.metric("Total SERP Coverage", f"{total_serp_coverage:,}")
            with col4:
                st.metric("Unique Domains", len(domain_ai_scores))
            
            # Create tabs for different analyses
            tab1, tab2, tab3, tab4 = st.tabs(["🏆 Top AI Domains", "🤖 AI Popular vs SERP", "🔍 SERP Popular vs AI", "📈 Comparison Analysis"])
            
            with tab1:
                st.header("Top Influential Domains by AI Total")
                
                # Sort domains by AI total score
                top_ai_domains = sorted(domain_ai_scores.items(), key=lambda x: x[1], reverse=True)[:20]
                
                if top_ai_domains:
                    domains = [item[0] for item in top_ai_domains]
                    scores = [item[1] for item in top_ai_domains]
                    
                    fig = px.bar(
                        x=scores,
                        y=domains,
                        orientation='h',
                        title="Top 20 Domains by AI Coverage Score",
                        labels={'x': 'AI Coverage Score', 'y': 'Domain'},
                        color=scores,
                        color_continuous_scale='viridis'
                    )
                    fig.update_layout(height=600, yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Table with AI and SERP scores
                    st.subheader("📋 Top AI Domains Detailed Breakdown")
                    table_data = []
                    for domain, ai_score in top_ai_domains:
                        serp_score = domain_serp_scores.get(domain, 0)
                        table_data.append({
                            'Domain': domain,
                            'AI Total Score': ai_score,
                            'SERP Total Score': serp_score,
                            'AI/SERP Ratio': round(ai_score / serp_score, 2) if serp_score > 0 else '∞'
                        })
                    
                    st.dataframe(pd.DataFrame(table_data), use_container_width=True)
            
            with tab2:
                st.header("Domains Popular in AI but Not in SERP")
                st.markdown("Domains with high AI coverage but low/no SERP presence")
                
                # Find domains popular in AI but not in SERP
                ai_not_serp = []
                for domain, ai_score in domain_ai_scores.items():
                    serp_score = domain_serp_scores.get(domain, 0)
                    
                    # Consider "popular in AI" if score > median, "not popular in SERP" if score < 25% of AI score
                    ai_median = np.median(list(domain_ai_scores.values()))
                    
                    if ai_score >= ai_median and (serp_score == 0 or serp_score < ai_score * 0.25):
                        ratio = ai_score / serp_score if serp_score > 0 else float('inf')
                        ai_not_serp.append({
                            'Domain': domain,
                            'AI Score': ai_score,
                            'SERP Score': serp_score,
                            'AI/SERP Ratio': ratio
                        })
                
                ai_not_serp.sort(key=lambda x: x['AI Score'], reverse=True)
                
                if ai_not_serp:
                    # Visualization
                    top_15 = ai_not_serp[:15]
                    domains = [item['Domain'] for item in top_15]
                    ai_scores = [item['AI Score'] for item in top_15]
                    
                    fig = px.bar(
                        x=ai_scores,
                        y=domains,
                        orientation='h',
                        title="Top 15 Domains: High AI Coverage, Low SERP Coverage",
                        labels={'x': 'AI Coverage Score', 'y': 'Domain'},
                        color=ai_scores,
                        color_continuous_scale='reds'
                    )
                    fig.update_layout(height=500, yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.subheader(f"📋 All {len(ai_not_serp)} Domains Popular in AI but Not SERP")
                    st.dataframe(pd.DataFrame(ai_not_serp), use_container_width=True)
                else:
                    st.info("No domains found that are popular in AI but not in SERP")
            
            with tab3:
                st.header("Domains Popular in SERP but Not in AI")
                st.markdown("Domains with high SERP coverage but low/no AI presence")
                
                # Find domains popular in SERP but not in AI
                serp_not_ai = []
                for domain, serp_score in domain_serp_scores.items():
                    ai_score = domain_ai_scores.get(domain, 0)
                    
                    # Consider "popular in SERP" if score > median, "not popular in AI" if score < 25% of SERP score
                    serp_median = np.median(list(domain_serp_scores.values()))
                    
                    if serp_score >= serp_median and (ai_score == 0 or ai_score < serp_score * 0.25):
                        ratio = serp_score / ai_score if ai_score > 0 else float('inf')
                        serp_not_ai.append({
                            'Domain': domain,
                            'SERP Score': serp_score,
                            'AI Score': ai_score,
                            'SERP/AI Ratio': ratio
                        })
                
                serp_not_ai.sort(key=lambda x: x['SERP Score'], reverse=True)
                
                if serp_not_ai:
                    # Visualization
                    top_15 = serp_not_ai[:15]
                    domains = [item['Domain'] for item in top_15]
                    serp_scores = [item['SERP Score'] for item in top_15]
                    
                    fig = px.bar(
                        x=serp_scores,
                        y=domains,
                        orientation='h',
                        title="Top 15 Domains: High SERP Coverage, Low AI Coverage",
                        labels={'x': 'SERP Coverage Score', 'y': 'Domain'},
                        color=serp_scores,
                        color_continuous_scale='blues'
                    )
                    fig.update_layout(height=500, yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.subheader(f"📋 All {len(serp_not_ai)} Domains Popular in SERP but Not AI")
                    st.dataframe(pd.DataFrame(serp_not_ai), use_container_width=True)
                else:
                    st.info("No domains found that are popular in SERP but not in AI")
            
            with tab4:
                st.header("AI vs SERP Comparison Analysis")
                
                # Scatter plot comparing AI vs SERP scores
                all_domains = set(list(domain_ai_scores.keys()) + list(domain_serp_scores.keys()))
                scatter_data = []
                
                for domain in all_domains:
                    ai_score = domain_ai_scores.get(domain, 0)
                    serp_score = domain_serp_scores.get(domain, 0)
                    scatter_data.append({
                        'Domain': domain,
                        'AI Score': ai_score,
                        'SERP Score': serp_score
                    })
                
                scatter_df = pd.DataFrame(scatter_data)
                
                fig_scatter = px.scatter(
                    scatter_df,
                    x='SERP Score',
                    y='AI Score',
                    hover_data=['Domain'],
                    title='AI Score vs SERP Score by Domain',
                    labels={'x': 'SERP Coverage Score', 'y': 'AI Coverage Score'}
                )
                
                # Add diagonal line for reference
                max_val = max(scatter_df['AI Score'].max(), scatter_df['SERP Score'].max())
                fig_scatter.add_trace(
                    go.Scatter(
                        x=[0, max_val],
                        y=[0, max_val],
                        mode='lines',
                        name='Equal Coverage Line',
                        line=dict(dash='dash', color='red')
                    )
                )
                
                st.plotly_chart(fig_scatter, use_container_width=True)
                
                # Correlation analysis
                correlation = scatter_df['AI Score'].corr(scatter_df['SERP Score'])
                st.metric("AI vs SERP Correlation", f"{correlation:.3f}")
                
                if correlation > 0.7:
                    st.success("Strong positive correlation - AI and SERP tend to favor similar domains")
                elif correlation > 0.3:
                    st.info("Moderate positive correlation - Some alignment between AI and SERP preferences")
                else:
                    st.warning("Weak correlation - AI and SERP show different domain preferences")
            
            # Download section
            st.header("💾 Download Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # All domains analysis
                all_analysis = []
                all_domains = set(list(domain_ai_scores.keys()) + list(domain_serp_scores.keys()))
                
                for domain in all_domains:
                    ai_score = domain_ai_scores.get(domain, 0)
                    serp_score = domain_serp_scores.get(domain, 0)
                    all_analysis.append({
                        'Domain': domain,
                        'AI_Total_Score': ai_score,
                        'SERP_Total_Score': serp_score,
                        'AI_SERP_Ratio': ai_score / serp_score if serp_score > 0 else float('inf')
                    })
                
                all_df = pd.DataFrame(all_analysis)
                all_df = all_df.sort_values('AI_Total_Score', ascending=False)
                
                st.download_button(
                    label="📥 Download Complete Analysis",
                    data=all_df.to_csv(index=False),
                    file_name="complete_domain_analysis.csv",
                    mime="text/csv"
                )
            
            with col2:
                if 'ai_not_serp' in locals() and ai_not_serp:
                    ai_not_serp_df = pd.DataFrame(ai_not_serp)
                    st.download_button(
                        label="📥 Download AI-Popular Domains",
                        data=ai_not_serp_df.to_csv(index=False),
                        file_name="ai_popular_domains.csv",
                        mime="text/csv"
                    )
            
            with col3:
                if 'serp_not_ai' in locals() and serp_not_ai:
                    serp_not_ai_df = pd.DataFrame(serp_not_ai)
                    st.download_button(
                        label="📥 Download SERP-Popular Domains",
                        data=serp_not_ai_df.to_csv(index=False),
                        file_name="serp_popular_domains.csv",
                        mime="text/csv"
                    )
            
        except Exception as e:
            st.error(f"Error processing the CSV file: {str(e)}")
            st.info("Please make sure your CSV file contains all required columns.")
    
    else:
        st.info("👆 Please upload a CSV file to begin the analysis")
        
        st.subheader("📋 Required Columns")
        st.markdown("""
        **Coverage Columns:**
        - `sge_queries_covered` - SGE query coverage
        - `perplexity_queries_covered` - Perplexity query coverage  
        - `gpt_queries_covered` - GPT query coverage
        - `serp_queries_covered` - SERP query coverage
        
        **Source Columns:**
        - `gpt_sources` - URLs cited by GPT
        - `perplexity_sources` - URLs cited by Perplexity  
        - `aio_response_sources` - URLs cited by AIO
        
        **Analysis:**
        - **AI Total** = Sum of SGE + Perplexity + GPT coverage per row
        - Domains weighted by their associated query coverage scores
        - Comparison of AI-popular vs SERP-popular domains
        """)

if __name__ == "__main__":
    main()
