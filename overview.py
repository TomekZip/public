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
    if pd.isna(text) or text == '' or str(text).strip() == '':
        return []
    
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    urls = re.findall(url_pattern, str(text))
    return [normalize_url(url) for url in urls]

def has_content(value):
    """Check if a field has meaningful content"""
    if pd.isna(value) or value == '' or str(value).strip() == '':
        return False
    return True

def calculate_ai_metrics(df, use_aio=True, use_perplexity=True, use_gpt=True):
    """Calculate AI coverage based on actual response presence and selected platforms"""
    
    # Required columns for AI analysis
    required_cols = [
        'aio_response_sources', 'perplexity_sources', 'gpt_sources',
        'serp_results', 'aio_response_status', 'perplexity_response_text', 'gpt_response_text'
    ]
    
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        return None, f"Missing required columns: {', '.join(missing_cols)}"
    
    # Calculate coverage for each platform (1 if has response/sources, 0 if not)
    df['aio_covered'] = df.apply(lambda row: 1 if (
        use_aio and (has_content(row['aio_response_sources']) or 
        (has_content(row['aio_response_status']) and str(row['aio_response_status']).lower() != 'no response'))
    ) else 0, axis=1)
    
    df['perplexity_covered'] = df.apply(lambda row: 1 if (
        use_perplexity and (has_content(row['perplexity_sources']) or has_content(row['perplexity_response_text']))
    ) else 0, axis=1)
    
    df['gpt_covered'] = df.apply(lambda row: 1 if (
        use_gpt and (has_content(row['gpt_sources']) or has_content(row['gpt_response_text']))
    ) else 0, axis=1)
    
    df['serp_covered'] = df.apply(lambda row: 1 if has_content(row['serp_results']) else 0, axis=1)
    
    # Calculate AI total (sum of selected AI platforms)
    df['ai_total'] = df['aio_covered'] + df['perplexity_covered'] + df['gpt_covered']
    
    # Extract domains and calculate weighted scores
    domain_ai_scores = {}
    domain_serp_scores = {}
    domain_breakdown = {}  # Track breakdown by AI platform
    
    # Only include source columns for selected platforms
    source_columns = []
    if use_aio:
        source_columns.append('aio_response_sources')
    if use_perplexity:
        source_columns.append('perplexity_sources')
    if use_gpt:
        source_columns.append('gpt_sources')
    
    for idx, row in df.iterrows():
        ai_total = row['ai_total']
        serp_total = row['serp_covered']
        
        # Get all domains from AI sources in this row
        all_domains = set()
        row_domain_sources = {}  # Track which AI platforms mention each domain in this row
        
        for col in source_columns:
            if has_content(row[col]):
                urls = extract_urls_from_text(row[col])
                domains = [get_domain_from_url(url) for url in urls]
                
                for domain in domains:
                    all_domains.add(domain)
                    if domain not in row_domain_sources:
                        row_domain_sources[domain] = []
                    row_domain_sources[domain].append(col)
        
        # Add scores for each domain found in this row
        for domain in all_domains:
            if domain not in domain_ai_scores:
                domain_ai_scores[domain] = 0
                domain_serp_scores[domain] = 0
                domain_breakdown[domain] = {'aio_response_sources': 0, 'perplexity_sources': 0, 'gpt_sources': 0}
            
            domain_ai_scores[domain] += ai_total
            domain_serp_scores[domain] += serp_total
            
            # Track breakdown by AI platform
            for source_col in row_domain_sources[domain]:
                domain_breakdown[domain][source_col] += 1
    
    return {
        'domain_ai_scores': domain_ai_scores,
        'domain_serp_scores': domain_serp_scores,
        'domain_breakdown': domain_breakdown,
        'df': df,
        'total_queries': len(df),
        'selected_platforms': {'aio': use_aio, 'perplexity': use_perplexity, 'gpt': use_gpt}
    }, None

def main():
    st.set_page_config(
        page_title="AI vs SERP Domain Analysis",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 AI vs SERP Domain Influence Analysis")
    st.markdown("Analyze domain influence across AI platforms (AIO, Perplexity, GPT) vs traditional SERP results")
    
    # AI Platform Selector
    st.header("🔧 AI Platform Selection")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        use_aio = st.checkbox("AI Overviews", value=True)
    with col2:
        use_perplexity = st.checkbox("Perplexity", value=True)
    with col3:
        use_gpt = st.checkbox("ChatGPT", value=True)
    
    if not any([use_aio, use_perplexity, use_gpt]):
        st.error("❌ Please select at least one AI platform for analysis")
        return
    
    selected_platforms = []
    if use_aio:
        selected_platforms.append("AI Overviews")
    if use_perplexity:
        selected_platforms.append("Perplexity")
    if use_gpt:
        selected_platforms.append("ChatGPT")
    
    st.success(f"✅ Selected platforms: {', '.join(selected_platforms)}")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Successfully loaded CSV with {len(df)} rows and {len(df.columns)} columns")
            
            # Calculate metrics
            result, error = calculate_ai_metrics(df, use_aio, use_perplexity, use_gpt)
            
            if error:
                st.error(f"❌ {error}")
                st.write("Available columns:", list(df.columns))
                return
            
            domain_ai_scores = result['domain_ai_scores']
            domain_serp_scores = result['domain_serp_scores']
            domain_breakdown = result['domain_breakdown']
            processed_df = result['df']
            total_queries = result['total_queries']
            selected_platforms = result['selected_platforms']
            
            if not domain_ai_scores:
                st.warning("⚠️ No domains found in AI source columns")
                return
            
            # Calculate summary stats
            total_aio_coverage = processed_df['aio_covered'].sum()
            total_perplexity_coverage = processed_df['perplexity_covered'].sum()
            total_gpt_coverage = processed_df['gpt_covered'].sum()
            total_ai_coverage = processed_df['ai_total'].sum()
            total_serp_coverage = processed_df['serp_covered'].sum()
            
            st.header("📊 Coverage Summary")
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("Total Queries", total_queries)
            with col2:
                st.metric("AIO Coverage", f"{total_aio_coverage}")
            with col3:
                st.metric("Perplexity Coverage", f"{total_perplexity_coverage}")
            with col4:
                st.metric("GPT Coverage", f"{total_gpt_coverage}")
            with col5:
                st.metric("SERP Coverage", f"{total_serp_coverage}")
            
            # Platform coverage rates
            st.subheader("📈 Platform Coverage Rates")
            
            # Only show selected platforms
            platform_names = []
            platform_coverages = []
            platform_rates = []
            
            if use_aio:
                platform_names.append('AI Overviews')
                platform_coverages.append(total_aio_coverage)
                platform_rates.append(f"{total_aio_coverage/total_queries*100:.1f}%")
            
            if use_perplexity:
                platform_names.append('Perplexity')
                platform_coverages.append(total_perplexity_coverage)
                platform_rates.append(f"{total_perplexity_coverage/total_queries*100:.1f}%")
            
            if use_gpt:
                platform_names.append('ChatGPT')
                platform_coverages.append(total_gpt_coverage)
                platform_rates.append(f"{total_gpt_coverage/total_queries*100:.1f}%")
            
            # Always show SERP
            platform_names.append('SERP')
            platform_coverages.append(total_serp_coverage)
            platform_rates.append(f"{total_serp_coverage/total_queries*100:.1f}%")
            
            coverage_data = {
                'Platform': platform_names,
                'Coverage': platform_coverages,
                'Rate': platform_rates
            }
            
            fig_coverage = px.bar(
                x=coverage_data['Platform'],
                y=coverage_data['Coverage'],
                title="Query Coverage by Platform",
                labels={'x': 'Platform', 'y': 'Queries Covered'},
                color=coverage_data['Coverage'],
                color_continuous_scale='viridis',
                text=coverage_data['Rate']
            )
            fig_coverage.update_traces(textposition='outside')
            st.plotly_chart(fig_coverage, use_container_width=True)
            
            # Create tabs for different analyses
            tab1, tab2, tab3, tab4, tab5 = st.tabs(["🏆 Top AI Domains", "🤖 AI Popular vs SERP", "🔍 SERP Popular vs AI", "📊 Platform Breakdown", "📈 Comparison Analysis"])
            
            with tab1:
                st.header("Top Influential Domains by AI Total Score")
                
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
                    
                    # Table with detailed breakdown
                    st.subheader("📋 Top AI Domains Detailed Breakdown")
                    table_data = []
                    for domain, ai_score in top_ai_domains:
                        serp_score = domain_serp_scores.get(domain, 0)
                        breakdown = domain_breakdown.get(domain, {})
                        table_data.append({
                            'Domain': domain,
                            'AI Total Score': ai_score,
                            'SERP Score': serp_score,
                            'AIO Citations': breakdown.get('aio_response_sources', 0) if use_aio else 'N/A',
                            'Perplexity Citations': breakdown.get('perplexity_sources', 0) if use_perplexity else 'N/A',
                            'GPT Citations': breakdown.get('gpt_sources', 0) if use_gpt else 'N/A',
                            'AI/SERP Ratio': round(ai_score / serp_score, 2) if serp_score > 0 else '∞'
                        })
                    
                    st.dataframe(pd.DataFrame(table_data), use_container_width=True)
            
            with tab2:
                st.header("Domains Popular in AI but Not in SERP")
                st.markdown("Domains frequently cited by AI platforms but rarely appearing in SERP results")
                
                # Show data distribution first
                ai_scores_list = list(domain_ai_scores.values())
                serp_scores_list = list(domain_serp_scores.values())
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Max AI Score", max(ai_scores_list) if ai_scores_list else 0)
                with col2:
                    st.metric("Avg AI Score", f"{np.mean(ai_scores_list):.1f}" if ai_scores_list else 0)
                with col3:
                    st.metric("Domains with AI Score > 0", sum(1 for s in ai_scores_list if s > 0))
                
                # Find domains popular in AI but not in SERP with more flexible thresholds
                ai_not_serp = []
                ai_threshold = max(1, np.percentile(ai_scores_list, 60) if ai_scores_list else 1)  # Top 40% of AI scores
                
                st.info(f"Using AI threshold: {ai_threshold:.1f} (top 40% of AI scores)")
                
                for domain, ai_score in domain_ai_scores.items():
                    serp_score = domain_serp_scores.get(domain, 0)
                    
                    # More flexible criteria: AI score above threshold AND (no SERP presence OR AI score > 2x SERP score)
                    if ai_score >= ai_threshold and (serp_score == 0 or ai_score > serp_score * 2):
                        breakdown = domain_breakdown.get(domain, {})
                        ai_not_serp.append({
                            'Domain': domain,
                            'AI Score': ai_score,
                            'SERP Score': serp_score,
                            'AIO Citations': breakdown.get('aio_response_sources', 0),
                            'Perplexity Citations': breakdown.get('perplexity_sources', 0),
                            'GPT Citations': breakdown.get('gpt_sources', 0),
                            'AI/SERP Ratio': ai_score / serp_score if serp_score > 0 else float('inf'),
                            'Difference': ai_score - serp_score
                        })
                
                ai_not_serp.sort(key=lambda x: x['Difference'], reverse=True)
                
                if ai_not_serp:
                    # Visualization
                    top_15 = ai_not_serp[:15]
                    domains = [item['Domain'] for item in top_15]
                    ai_scores = [item['AI Score'] for item in top_15]
                    
                    fig = px.bar(
                        x=ai_scores,
                        y=domains,
                        orientation='h',
                        title="Top 15 Domains: Higher AI Coverage than SERP Coverage",
                        labels={'x': 'AI Coverage Score', 'y': 'Domain'},
                        color=ai_scores,
                        color_continuous_scale='reds'
                    )
                    fig.update_layout(height=500, yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.subheader(f"📋 All {len(ai_not_serp)} Domains More Popular in AI than SERP")
                    st.dataframe(pd.DataFrame(ai_not_serp), use_container_width=True)
                else:
                    st.warning("No domains found that are more popular in AI than SERP with current thresholds")
                    
                    # Show some examples anyway
                    st.subheader("🔍 Top AI Domains Regardless of SERP Comparison")
                    top_ai_only = sorted(domain_ai_scores.items(), key=lambda x: x[1], reverse=True)[:10]
                    ai_only_data = []
                    for domain, ai_score in top_ai_only:
                        serp_score = domain_serp_scores.get(domain, 0)
                        breakdown = domain_breakdown.get(domain, {})
                        ai_only_data.append({
                            'Domain': domain,
                            'AI Score': ai_score,
                            'SERP Score': serp_score,
                            'AIO Citations': breakdown.get('aio_response_sources', 0),
                            'Perplexity Citations': breakdown.get('perplexity_sources', 0),
                            'GPT Citations': breakdown.get('gpt_sources', 0)
                        })
                    st.dataframe(pd.DataFrame(ai_only_data), use_container_width=True)
            
            with tab3:
                st.header("Domains Popular in SERP but Not in AI")
                st.markdown("Domains frequently appearing in SERP results but rarely cited by AI platforms")
                
                # Show data distribution first
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Max SERP Score", max(serp_scores_list) if serp_scores_list else 0)
                with col2:
                    st.metric("Avg SERP Score", f"{np.mean(serp_scores_list):.1f}" if serp_scores_list else 0)
                with col3:
                    st.metric("Domains with SERP Score > 0", sum(1 for s in serp_scores_list if s > 0))
                
                # Find domains popular in SERP but not in AI with more flexible thresholds
                serp_not_ai = []
                serp_threshold = max(1, np.percentile(serp_scores_list, 60) if serp_scores_list else 1)  # Top 40% of SERP scores
                
                st.info(f"Using SERP threshold: {serp_threshold:.1f} (top 40% of SERP scores)")
                
                for domain, serp_score in domain_serp_scores.items():
                    ai_score = domain_ai_scores.get(domain, 0)
                    
                    # More flexible criteria: SERP score above threshold AND (no AI presence OR SERP score > 2x AI score)
                    if serp_score >= serp_threshold and (ai_score == 0 or serp_score > ai_score * 2):
                        breakdown = domain_breakdown.get(domain, {})
                        serp_not_ai.append({
                            'Domain': domain,
                            'SERP Score': serp_score,
                            'AI Score': ai_score,
                            'AIO Citations': breakdown.get('aio_response_sources', 0),
                            'Perplexity Citations': breakdown.get('perplexity_sources', 0),
                            'GPT Citations': breakdown.get('gpt_sources', 0),
                            'SERP/AI Ratio': serp_score / ai_score if ai_score > 0 else float('inf'),
                            'Difference': serp_score - ai_score
                        })
                
                serp_not_ai.sort(key=lambda x: x['Difference'], reverse=True)
                
                if serp_not_ai:
                    # Visualization
                    top_15 = serp_not_ai[:15]
                    domains = [item['Domain'] for item in top_15]
                    serp_scores = [item['SERP Score'] for item in top_15]
                    
                    fig = px.bar(
                        x=serp_scores,
                        y=domains,
                        orientation='h',
                        title="Top 15 Domains: Higher SERP Coverage than AI Coverage",
                        labels={'x': 'SERP Coverage Score', 'y': 'Domain'},
                        color=serp_scores,
                        color_continuous_scale='blues'
                    )
                    fig.update_layout(height=500, yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.subheader(f"📋 All {len(serp_not_ai)} Domains More Popular in SERP than AI")
                    st.dataframe(pd.DataFrame(serp_not_ai), use_container_width=True)
                else:
                    st.warning("No domains found that are more popular in SERP than AI with current thresholds")
                    
                    # Show some examples anyway
                    st.subheader("🔍 Top SERP Domains Regardless of AI Comparison")
                    top_serp_only = sorted(domain_serp_scores.items(), key=lambda x: x[1], reverse=True)[:10]
                    serp_only_data = []
                    for domain, serp_score in top_serp_only:
                        ai_score = domain_ai_scores.get(domain, 0)
                        breakdown = domain_breakdown.get(domain, {})
                        serp_only_data.append({
                            'Domain': domain,
                            'SERP Score': serp_score,
                            'AI Score': ai_score,
                            'AIO Citations': breakdown.get('aio_response_sources', 0),
                            'Perplexity Citations': breakdown.get('perplexity_sources', 0),
                            'GPT Citations': breakdown.get('gpt_sources', 0)
                        })
                    st.dataframe(pd.DataFrame(serp_only_data), use_container_width=True)
            
            with tab4:
                st.header("AI Platform Breakdown Analysis")
                
                # Create breakdown by AI platform
                platform_totals = {
                    'AIO': sum(breakdown.get('aio_response_sources', 0) for breakdown in domain_breakdown.values()),
                    'Perplexity': sum(breakdown.get('perplexity_sources', 0) for breakdown in domain_breakdown.values()),
                    'GPT': sum(breakdown.get('gpt_sources', 0) for breakdown in domain_breakdown.values())
                }
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Pie chart of platform distribution
                    fig_pie = px.pie(
                        values=list(platform_totals.values()),
                        names=list(platform_totals.keys()),
                        title="Domain Citations by AI Platform"
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                with col2:
                    # Bar chart of platform totals
                    fig_bar = px.bar(
                        x=list(platform_totals.keys()),
                        y=list(platform_totals.values()),
                        title="Total Domain Citations by Platform",
                        labels={'x': 'AI Platform', 'y': 'Total Citations'},
                        color=list(platform_totals.values()),
                        color_continuous_scale='viridis'
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)
                
                # Top domains by platform
                st.subheader("📊 Top Domains by AI Platform")
                
                for platform, column in [('AIO', 'aio_response_sources'), ('Perplexity', 'perplexity_sources'), ('GPT', 'gpt_sources')]:
                    platform_domains = [(domain, breakdown.get(column, 0)) 
                                      for domain, breakdown in domain_breakdown.items() 
                                      if breakdown.get(column, 0) > 0]
                    platform_domains.sort(key=lambda x: x[1], reverse=True)
                    
                    if platform_domains:
                        st.write(f"**Top 10 {platform} Domains:**")
                        top_10 = platform_domains[:10]
                        platform_df = pd.DataFrame(top_10, columns=['Domain', f'{platform} Citations'])
                        st.dataframe(platform_df, use_container_width=True)
            
            with tab5:
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
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("AI vs SERP Correlation", f"{correlation:.3f}")
                with col2:
                    st.metric("Unique Domains", len(all_domains))
                with col3:
                    st.metric("Total AI Citations", sum(domain_ai_scores.values()))
                
                if correlation > 0.7:
                    st.success("🟢 Strong positive correlation - AI and SERP tend to favor similar domains")
                elif correlation > 0.3:
                    st.info("🟡 Moderate positive correlation - Some alignment between AI and SERP preferences")
                else:
                    st.warning("🔴 Weak correlation - AI and SERP show different domain preferences")
            
            # Download section
            st.header("💾 Download Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Complete analysis
                all_analysis = []
                all_domains = set(list(domain_ai_scores.keys()) + list(domain_serp_scores.keys()))
                
                for domain in all_domains:
                    ai_score = domain_ai_scores.get(domain, 0)
                    serp_score = domain_serp_scores.get(domain, 0)
                    breakdown = domain_breakdown.get(domain, {})
                    all_analysis.append({
                        'Domain': domain,
                        'AI_Total_Score': ai_score,
                        'SERP_Total_Score': serp_score,
                        'AIO_Citations': breakdown.get('aio_response_sources', 0),
                        'Perplexity_Citations': breakdown.get('perplexity_sources', 0),
                        'GPT_Citations': breakdown.get('gpt_sources', 0),
                        'AI_SERP_Ratio': ai_score / serp_score if serp_score > 0 else float('inf')
                    })
                
                all_df = pd.DataFrame(all_analysis)
                all_df = all_df.sort_values('AI_Total_Score', ascending=False)
                
                st.download_button(
                    label="📥 Complete Analysis",
                    data=all_df.to_csv(index=False),
                    file_name="complete_domain_analysis.csv",
                    mime="text/csv"
                )
            
            with col2:
                if 'ai_not_serp' in locals() and ai_not_serp:
                    ai_not_serp_df = pd.DataFrame(ai_not_serp)
                    st.download_button(
                        label="📥 AI-Popular Domains",
                        data=ai_not_serp_df.to_csv(index=False),
                        file_name="ai_popular_domains.csv",
                        mime="text/csv"
                    )
            
            with col3:
                if 'serp_not_ai' in locals() and serp_not_ai:
                    serp_not_ai_df = pd.DataFrame(serp_not_ai)
                    st.download_button(
                        label="📥 SERP-Popular Domains",
                        data=serp_not_ai_df.to_csv(index=False),
                        file_name="serp_popular_domains.csv",
                        mime="text/csv"
                    )
            
        except Exception as e:
            st.error(f"Error processing the CSV file: {str(e)}")
            st.info("Please make sure your CSV file contains all required columns.")
    
    else:
        st.info("👆 Please upload a CSV file to begin the analysis")
        
        st.subheader("📋 How It Works")
        st.markdown("""
        **Coverage Calculation:**
        - **AIO Covered**: 1 if has `aio_response_sources` or valid `aio_response_status`
        - **Perplexity Covered**: 1 if has `perplexity_sources` or `perplexity_response_text`  
        - **GPT Covered**: 1 if has `gpt_sources` or `gpt_response_text`
        - **SERP Covered**: 1 if has `serp_results`
        - **AI Total**: Sum of AIO + Perplexity + GPT coverage per query
        
        **Domain Scoring:**
        - Each domain gets points equal to the AI total score for queries where it appears
        - Compares AI-favored domains vs SERP-popular domains
        - Shows platform-specific breakdowns (which AI cited which domains)
        """)

if __name__ == "__main__":
    main()
