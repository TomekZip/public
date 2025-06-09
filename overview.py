import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter
import re
from urllib.parse import urlparse
import plotly.express as px
import plotly.graph_objects as go

def extract_urls_from_text(text):
    """Extract URLs from text using regex"""
    if pd.isna(text) or text == '':
        return []
    
    # Pattern to match URLs
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    urls = re.findall(url_pattern, str(text))
    return urls

def get_domain_from_url(url):
    """Extract domain from URL"""
    try:
        parsed = urlparse(url)
        return parsed.netloc.lower()
    except:
        return url

def analyze_url_influence(df):
    """Analyze URL influence across different columns"""
    url_columns = []
    
    # Identify columns that likely contain URLs
    for col in df.columns:
        if any(keyword in col.lower() for keyword in ['source', 'url', 'link', 'citation']):
            url_columns.append(col)
    
    all_urls = []
    url_sources = {}  # Track which columns each URL comes from
    
    for col in url_columns:
        for idx, cell in enumerate(df[col]):
            urls = extract_urls_from_text(cell)
            for url in urls:
                domain = get_domain_from_url(url)
                all_urls.append(domain)
                
                if domain not in url_sources:
                    url_sources[domain] = {}
                if col not in url_sources[domain]:
                    url_sources[domain][col] = 0
                url_sources[domain][col] += 1
    
    return Counter(all_urls), url_sources, url_columns

def main():
    st.set_page_config(
        page_title="CSV Analysis Dashboard",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("📊 CSV Analysis Dashboard")
    st.markdown("Upload your CSV file to analyze basic statistics and find the most influential URLs")
    
    # File upload
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            # Load the CSV
            df = pd.read_csv(uploaded_file)
            
            st.success(f"✅ Successfully loaded CSV with {len(df)} rows and {len(df.columns)} columns")
            
            # Basic Statistics Section
            st.header("📈 Basic Statistics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Rows", len(df))
            
            with col2:
                st.metric("Total Columns", len(df.columns))
            
            with col3:
                st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")
            
            with col4:
                st.metric("Missing Values", df.isnull().sum().sum())
            
            # Data types breakdown
            st.subheader("Data Types Overview")
            dtype_counts = df.dtypes.value_counts()
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_dtype = px.pie(
                    values=dtype_counts.values, 
                    names=dtype_counts.index,
                    title="Distribution of Data Types"
                )
                st.plotly_chart(fig_dtype, use_container_width=True)
            
            with col2:
                # Missing values by column
                missing_data = df.isnull().sum().sort_values(ascending=False)
                missing_data = missing_data[missing_data > 0]
                
                if len(missing_data) > 0:
                    fig_missing = px.bar(
                        x=missing_data.index,
                        y=missing_data.values,
                        title="Missing Values by Column",
                        labels={'x': 'Columns', 'y': 'Missing Count'}
                    )
                    fig_missing.update_xaxes(tickangle=45)
                    st.plotly_chart(fig_missing, use_container_width=True)
                else:
                    st.info("No missing values found in the dataset! 🎉")
            
            # Column Information
            st.subheader("Column Information")
            
            col_info = []
            for col in df.columns:
                col_data = {
                    'Column': col,
                    'Data Type': str(df[col].dtype),
                    'Non-Null Count': df[col].count(),
                    'Null Count': df[col].isnull().sum(),
                    'Unique Values': df[col].nunique(),
                    'Sample Values': ', '.join(str(x) for x in df[col].dropna().head(3).tolist())
                }
                col_info.append(col_data)
            
            col_info_df = pd.DataFrame(col_info)
            st.dataframe(col_info_df, use_container_width=True)
            
            # URL Analysis Section
            st.header("🔗 Most Influential URLs Analysis")
            
            url_counter, url_sources, url_columns = analyze_url_influence(df)
            
            if url_counter:
                st.success(f"Found {len(url_counter)} unique domains across {len(url_columns)} URL-containing columns")
                
                # Show which columns contain URLs
                st.subheader("URL-containing columns detected:")
                st.write(", ".join(url_columns))
                
                # Top influential URLs
                st.subheader("🏆 Top 20 Most Influential Domains")
                
                top_urls = url_counter.most_common(20)
                
                if top_urls:
                    # Create visualization
                    domains = [item[0] for item in top_urls]
                    counts = [item[1] for item in top_urls]
                    
                    fig_urls = px.bar(
                        x=counts,
                        y=domains,
                        orientation='h',
                        title="Most Frequently Mentioned Domains",
                        labels={'x': 'Mention Count', 'y': 'Domain'},
                        color=counts,
                        color_continuous_scale='viridis'
                    )
                    fig_urls.update_layout(height=600, yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig_urls, use_container_width=True)
                    
                    # Detailed breakdown table
                    st.subheader("📊 Detailed Breakdown by Column")
                    
                    breakdown_data = []
                    for domain, count in top_urls[:10]:  # Top 10 for detailed breakdown
                        row = {'Domain': domain, 'Total Count': count}
                        
                        # Add counts for each URL column
                        for col in url_columns:
                            row[f'{col}'] = url_sources.get(domain, {}).get(col, 0)
                        
                        breakdown_data.append(row)
                    
                    breakdown_df = pd.DataFrame(breakdown_data)
                    st.dataframe(breakdown_df, use_container_width=True)
                    
                    # URL column distribution
                    st.subheader("📈 URL Distribution Across Columns")
                    
                    col_totals = {}
                    for col in url_columns:
                        total = sum(url_sources.get(domain, {}).get(col, 0) for domain in url_sources)
                        col_totals[col] = total
                    
                    if col_totals:
                        fig_col_dist = px.pie(
                            values=list(col_totals.values()),
                            names=list(col_totals.keys()),
                            title="URL Mentions by Column"
                        )
                        st.plotly_chart(fig_col_dist, use_container_width=True)
                
            else:
                st.warning("No URLs found in the dataset. The analysis looked for columns containing 'source', 'url', 'link', or 'citation' in their names.")
            
            # Sample Data Preview
            st.header("👀 Data Preview")
            st.subheader("First 10 rows:")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Download processed data
            st.header("💾 Download Results")
            
            if url_counter:
                # Create summary report
                summary_data = {
                    'Domain': [domain for domain, count in url_counter.most_common()],
                    'Total_Mentions': [count for domain, count in url_counter.most_common()]
                }
                
                # Add column-specific counts
                for col in url_columns:
                    summary_data[f'{col}_mentions'] = [
                        url_sources.get(domain, {}).get(col, 0) 
                        for domain, count in url_counter.most_common()
                    ]
                
                summary_df = pd.DataFrame(summary_data)
                
                csv_summary = summary_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download URL Analysis Summary",
                    data=csv_summary,
                    file_name="url_analysis_summary.csv",
                    mime="text/csv"
                )
            
        except Exception as e:
            st.error(f"Error processing the CSV file: {str(e)}")
            st.info("Please make sure your CSV file is properly formatted and not corrupted.")
    
    else:
        st.info("👆 Please upload a CSV file to begin the analysis")
        
        # Show example of expected format
        st.subheader("📋 Expected CSV Format")
        st.markdown("""
        Your CSV can contain any columns, but for URL analysis, columns with these keywords in their names will be analyzed:
        - **source** (e.g., 'aio_response_sources', 'data_sources')
        - **url** (e.g., 'source_urls', 'reference_urls')  
        - **link** (e.g., 'external_links', 'citation_links')
        - **citation** (e.g., 'citations', 'citation_sources')
        
        The program will automatically detect and extract URLs from these columns.
        """)

if __name__ == "__main__":
    main()
