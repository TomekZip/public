
import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter
import re
from urllib.parse import urlparse, urljoin
import plotly.express as px
import plotly.graph_objects as go

def normalize_url(url):
    """Normalize URL by removing parameters and fragments"""
    try:
        parsed = urlparse(url.strip())
        # Reconstruct URL without query parameters and fragments
        normalized = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        # Remove trailing slash for consistency
        if normalized.endswith('/') and len(parsed.path) > 1:
            normalized = normalized[:-1]
        return normalized.lower()
    except:
        return url.strip().lower()

def extract_urls_from_text(text):
    """Extract URLs from text using regex"""
    if pd.isna(text) or text == '':
        return []
    
    # Pattern to match URLs
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    urls = re.findall(url_pattern, str(text))
    return [normalize_url(url) for url in urls]

def get_domain_from_url(url):
    """Extract domain from URL"""
    try:
        parsed = urlparse(url)
        return parsed.netloc.lower()
    except:
        return url

def analyze_source_urls(df):
    """Analyze URLs from the three specific source columns"""
    source_columns = ['gpt_sources', 'perplexity_sources', 'aio_response_sources']
    
    # Check which columns exist in the dataframe
    existing_columns = [col for col in source_columns if col in df.columns]
    
    if not existing_columns:
        return None, None, existing_columns
    
    all_urls = []
    url_sources = {}  # Track which columns each URL comes from
    domain_sources = {}  # Track which columns each domain comes from
    
    for col in existing_columns:
        for idx, cell in enumerate(df[col]):
            urls = extract_urls_from_text(cell)
            for url in urls:
                domain = get_domain_from_url(url)
                
                # Track full URLs
                all_urls.append(url)
                if url not in url_sources:
                    url_sources[url] = {}
                if col not in url_sources[url]:
                    url_sources[url][col] = 0
                url_sources[url][col] += 1
                
                # Track domains
                if domain not in domain_sources:
                    domain_sources[domain] = {}
                if col not in domain_sources[domain]:
                    domain_sources[domain][col] = 0
                domain_sources[domain][col] += 1
    
    url_counter = Counter(all_urls)
    domain_counter = Counter([get_domain_from_url(url) for url in all_urls])
    
    return (url_counter, domain_counter, url_sources, domain_sources, existing_columns)

def main():
    st.set_page_config(
        page_title="AI Sources URL Analysis",
        page_icon="🔗",
        layout="wide"
    )
    
    st.title("🔗 AI Sources URL Analysis")
    st.markdown("Upload your CSV file to analyze the most influential URLs from AI response sources")
    
    # File upload
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            # Load the CSV
            df = pd.read_csv(uploaded_file)
            
            st.success(f"✅ Successfully loaded CSV with {len(df)} rows and {len(df.columns)} columns")
            
            # Analyze source URLs
            result = analyze_source_urls(df)
            
            if result[0] is None:
                st.error("❌ None of the required columns found in the CSV!")
                st.write("Looking for columns: `gpt_sources`, `perplexity_sources`, `aio_response_sources`")
                st.write("Available columns:", list(df.columns))
                return
            
            url_counter, domain_counter, url_sources, domain_sources, existing_columns = result
            
            if not url_counter:
                st.warning("⚠️ No URLs found in the source columns")
                return
            
            st.success(f"📊 Found {len(url_counter)} unique URLs from {len(domain_counter)} domains across {len(existing_columns)} columns")
            st.write(f"**Analyzed columns:** {', '.join(existing_columns)}")
            
            # Create tabs for different views
            tab1, tab2, tab3 = st.tabs(["🏆 Top URLs", "🌐 Top Domains", "📊 Detailed Breakdown"])
            
            with tab1:
                st.header("Most Influential URLs (Normalized)")
                
                top_urls = url_counter.most_common(20)
                
                if top_urls:
                    # Create visualization for URLs
                    urls = [item[0] for item in top_urls]
                    counts = [item[1] for item in top_urls]
                    
                    # Truncate long URLs for display
                    display_urls = [url[:60] + "..." if len(url) > 60 else url for url in urls]
                    
                    fig_urls = px.bar(
                        x=counts,
                        y=display_urls,
                        orientation='h',
                        title="Top 20 Most Frequently Cited URLs",
                        labels={'x': 'Citation Count', 'y': 'URL'},
                        color=counts,
                        color_continuous_scale='viridis'
                    )
                    fig_urls.update_layout(height=600, yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig_urls, use_container_width=True)
                    
                    # URL table
                    st.subheader("📋 Top URLs Table")
                    url_data = []
                    for url, count in top_urls:
                        row = {'URL': url, 'Total Citations': count}
                        
                        # Add counts for each source column
                        for col in existing_columns:
                            row[col] = url_sources.get(url, {}).get(col, 0)
                        
                        url_data.append(row)
                    
                    url_df = pd.DataFrame(url_data)
                    st.dataframe(url_df, use_container_width=True)
            
            with tab2:
                st.header("Most Influential Domains")
                
                top_domains = domain_counter.most_common(15)
                
                if top_domains:
                    # Create visualization for domains
                    domains = [item[0] for item in top_domains]
                    counts = [item[1] for item in top_domains]
                    
                    fig_domains = px.bar(
                        x=counts,
                        y=domains,
                        orientation='h',
                        title="Top 15 Most Frequently Cited Domains",
                        labels={'x': 'Citation Count', 'y': 'Domain'},
                        color=counts,
                        color_continuous_scale='plasma'
                    )
                    fig_domains.update_layout(height=500, yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig_domains, use_container_width=True)
                    
                    # Domain table
                    st.subheader("📋 Top Domains Table")
                    domain_data = []
                    for domain, count in top_domains:
                        row = {'Domain': domain, 'Total Citations': count}
                        
                        # Add counts for each source column
                        for col in existing_columns:
                            row[col] = domain_sources.get(domain, {}).get(col, 0)
                        
                        domain_data.append(row)
                    
                    domain_df = pd.DataFrame(domain_data)
                    st.dataframe(domain_df, use_container_width=True)
            
            with tab3:
                st.header("Source Distribution Analysis")
                
                # Citations by source column
                col_totals = {}
                for col in existing_columns:
                    total = sum(url_sources.get(url, {}).get(col, 0) for url in url_sources)
                    col_totals[col] = total
                
                if col_totals:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig_col_dist = px.pie(
                            values=list(col_totals.values()),
                            names=list(col_totals.keys()),
                            title="URL Citations by AI Source"
                        )
                        st.plotly_chart(fig_col_dist, use_container_width=True)
                    
                    with col2:
                        # Bar chart for source totals
                        fig_col_bar = px.bar(
                            x=list(col_totals.keys()),
                            y=list(col_totals.values()),
                            title="Total Citations by AI Source",
                            labels={'x': 'AI Source', 'y': 'Total Citations'},
                            color=list(col_totals.values()),
                            color_continuous_scale='blues'
                        )
                        st.plotly_chart(fig_col_bar, use_container_width=True)
                
                # Summary statistics
                st.subheader("📈 Summary Statistics")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Unique URLs", len(url_counter))
                
                with col2:
                    st.metric("Total Unique Domains", len(domain_counter))
                
                with col3:
                    st.metric("Total Citations", sum(url_counter.values()))
                
                with col4:
                    avg_citations = sum(url_counter.values()) / len(url_counter) if url_counter else 0
                    st.metric("Avg Citations per URL", f"{avg_citations:.1f}")
            
            # Download section
            st.header("💾 Download Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # URL analysis CSV
                url_summary_data = []
                for url, count in url_counter.most_common():
                    row = {'URL': url, 'Total_Citations': count, 'Domain': get_domain_from_url(url)}
                    
                    for col in existing_columns:
                        row[f'{col}_citations'] = url_sources.get(url, {}).get(col, 0)
                    
                    url_summary_data.append(row)
                
                url_summary_df = pd.DataFrame(url_summary_data)
                csv_urls = url_summary_df.to_csv(index=False)
                
                st.download_button(
                    label="📥 Download URL Analysis",
                    data=csv_urls,
                    file_name="url_analysis.csv",
                    mime="text/csv"
                )
            
            with col2:
                # Domain analysis CSV
                domain_summary_data = []
                for domain, count in domain_counter.most_common():
                    row = {'Domain': domain, 'Total_Citations': count}
                    
                    for col in existing_columns:
                        row[f'{col}_citations'] = domain_sources.get(domain, {}).get(col, 0)
                    
                    domain_summary_data.append(row)
                
                domain_summary_df = pd.DataFrame(domain_summary_data)
                csv_domains = domain_summary_df.to_csv(index=False)
                
                st.download_button(
                    label="📥 Download Domain Analysis",
                    data=csv_domains,
                    file_name="domain_analysis.csv",
                    mime="text/csv"
                )
            
        except Exception as e:
            st.error(f"Error processing the CSV file: {str(e)}")
            st.info("Please make sure your CSV file is properly formatted and contains the required columns.")
    
    else:
        st.info("👆 Please upload a CSV file to begin the analysis")
        
        st.subheader("📋 Required Columns")
        st.markdown("""
        The program will analyze URLs from these columns:
        - **gpt_sources** - URLs cited by GPT
        - **perplexity_sources** - URLs cited by Perplexity
        - **aio_response_sources** - URLs cited by AIO responses
        
        **URL Normalization:**
        - Removes URL parameters (everything after `?`)
        - Removes URL fragments (everything after `#`)
        - Converts to lowercase for consistency
        - Removes trailing slashes
        """)

if __name__ == "__main__":
    main()
