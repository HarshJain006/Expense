import streamlit as st
import pandas as pd
import json
import os
from groq import Groq
import re
from datetime import datetime, timedelta
import altair as alt  # For interactive charts
import httpx  # For proxy bypass

# Page configuration
st.set_page_config(
    page_title="Transaction AI Assistant",
    page_icon="💰",
    layout="wide"
)

# Initialize Groq client
@st.cache_resource
def get_groq_client(api_key):
    return Groq(
        api_key=api_key,
        http_client=httpx.Client(proxies=None)
    )

def get_api_key():
    if 'groq_api_key' in st.session_state and st.session_state.groq_api_key:
        return st.session_state.groq_api_key
    
    api_key = os.environ.get("GROQ_API_KEY")
    if api_key:
        return api_key
    
    return None

def validate_api_key(api_key):
    try:
        client = Groq(
            api_key=api_key,
            http_client=httpx.Client(proxies=None)
        )
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": "Test"}],
            max_tokens=1,
            temperature=0
        )
        return True, None
    except Exception as e:
        return False, str(e)

# Load and preprocess data
@st.cache_data
def load_transaction_data(file_path):
    try:
        if file_path.endswith('.json'):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            df = pd.DataFrame(data)
        elif file_path.endswith('.csv'):
            df = pd.read_csv(file_path, encoding='utf-8')
            if df.columns[0] != 'Date':
                df.columns = ['Date', 'Merchant', 'Transaction ID', 'UTR No.', 'Paid by', 'Type', 'Amount']
        else:
            st.error("Unsupported file format. Use JSON or CSV.")
            return None
        
        df = preprocess_dataframe(df)
        return df
    
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def preprocess_dataframe(df):
    df['Amount'] = df['Amount'].astype(str).str.replace('₹', '').str.replace(',', '').str.strip()
    df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce').fillna(0)
    
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%y', errors='coerce')
    
    df['Type'] = df['Type'].str.upper().str.strip()
    
    df['Merchant'] = df['Merchant'].fillna('Unknown')
    
    df['Month'] = df['Date'].dt.month_name()
    df['Year'] = df['Date'].dt.year
    df['Day'] = df['Date'].dt.day
    df['Weekday'] = df['Date'].dt.day_name()
    
    return df

def get_dataframe_schema(df):
    schema = f"""
DataFrame Schema:
- Shape: {df.shape[0]} rows, {df.shape[1]} columns
- Columns: {', '.join(df.columns.tolist())}
- Date range: {df['Date'].min()} to {df['Date'].max()}
- Data types:
"""
    for col in df.columns:
        schema += f"  • {col}: {df[col].dtype}\n"
    
    schema += f"""
- Transaction Types: {df['Type'].unique().tolist()}
- Sample merchants: {df['Merchant'].value_counts().head(5).index.tolist()}

Important Notes:
- Amount is numeric (cleaned)
- Date is datetime
- Use df['Type'] == 'DEBIT' for expenses, 'CREDIT' for income
- Interactive Altair charts can be created and assigned to 'chart' when helpful
"""
    return schema

def clean_generated_code(code):
    code = re.sub(r'^import\s+.*$', '', code, flags=re.MULTILINE)
    code = re.sub(r'^from\s+.*$', '', code, flags=re.MULTILINE)
    code = re.sub(r'print\([^)]*\)', '', code)
    
    lines = [line.strip() for line in code.split('\n') if line.strip() and not line.strip().startswith('#')]
    return '\n'.join(lines).strip()

def generate_pandas_query(user_question, df, client):
    schema = get_dataframe_schema(df)
    
    system_prompt = f"""You are an expert Python pandas + Altair data analyst. Generate SAFE and EFFICIENT code.

{schema}

AVAILABLE: pd (pandas), datetime, timedelta, alt (altair). NO IMPORTS REQUIRED!

CRITICAL RULES:
1. Return ONLY executable Python code, no explanations or markdown
2. Use 'df' as the DataFrame variable name
3. ALWAYS assign the primary result to 'result' (DataFrame for listings/tables, numeric/str for summaries/totals)
4. OPTIONALLY, if a chart would add insight (time trends, monthly totals, top merchants, merchant-specific spending, distributions), assign an interactive Altair chart to 'chart'
5. NO try-except blocks or print statements
6. Use vectorized operations, avoid loops
7. Be case-insensitive for merchant names (use .str.contains with case=False)
8. For date queries, use datetime.now().date() - timedelta(days=1) for yesterday
9. Consider DEBIT for expenses unless specified otherwise
10. NO IMPORT STATEMENTS!

Chart Guidelines:
- Use tooltips, titles, proper axis labels
- For time series → mark_point() or mark_line(), x='Date:T'
- For categories/top lists → mark_bar(), sort='-y'
- For individual transactions → mark_circle() with tooltip
- Use .properties(title=..., width="container", height=400)

EXAMPLES:
User: "How much did I spend on 26th October?"
Code:
result = df[(df['Date'].dt.day == 26) & (df['Date'].dt.month == 10) & (df['Type'] == 'DEBIT')]['Amount'].sum()

User: "Total expenses at Daily Dose cafe"
Code:
result = df[(df['Merchant'].str.contains('Daily Dose', case=False, na=False)) & (df['Type'] == 'DEBIT')]['Amount'].sum()

User: "Show all transactions above 100"
Code:
result = df[df['Amount'] > 100][['Date', 'Merchant', 'Amount', 'Type']].sort_values('Amount', ascending=False)

User: "Show all transactions with cafe"
Code:
cafe_df = df[df['Merchant'].str.contains('cafe', case=False, na=False) & (df['Type'] == 'DEBIT')]
result = cafe_df[['Date', 'Merchant', 'Amount']].sort_values('Date')
chart = alt.Chart(result).mark_circle(size=80).encode(
    x=alt.X('Date:T', title='Date'),
    y=alt.Y('Amount:Q', title='Amount (₹)'),
    tooltip=['Date', 'Merchant', 'Amount']
).properties(
    title='Cafe Transactions Over Time',
    width="container",
    height=400
)

User: "Monthly spending trend"
Code:
monthly_df = df[df['Type'] == 'DEBIT'].groupby(df['Date'].dt.to_period('M'))['Amount'].sum().reset_index()
monthly_df['Month'] = monthly_df['Date'].astype(str)
result = monthly_df
chart = alt.Chart(monthly_df).mark_line(point=True).encode(
    x=alt.X('Month:T', title='Month'),
    y=alt.Y('Amount:Q', title='Total Spending (₹)'),
    tooltip=['Month', 'Amount']
).properties(title='Monthly Spending Trend', width="container", height=400)

User: "Top 5 merchants by spending"
Code:
top_df = df[df['Type'] == 'DEBIT'].groupby('Merchant')['Amount'].sum().sort_values(ascending=False).head(5).reset_index()
result = top_df
chart = alt.Chart(top_df).mark_bar().encode(
    x=alt.X('Merchant:N', sort='-y', title='Merchant'),
    y=alt.Y('Amount:Q', title='Total Spent (₹)'),
    tooltip=['Merchant', 'Amount']
).properties(title='Top 5 Merchants by Spending', width="container", height=400)
"""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Generate code for: {user_question}"}
            ],
            temperature=0.1,
            max_tokens=600
        )
        
        code = response.choices[0].message.content.strip()
        code = re.sub(r'^```python\n?', '', code)
        code = re.sub(r'^```\n?', '', code)
        code = re.sub(r'\n?```$', '', code)
        
        return code.strip()
    
    except Exception as e:
        st.error(f"Error generating query: {str(e)}")
        return None

def execute_query_safely(code, df):
    forbidden_patterns = [
        'import', 'exec', 'eval', '__', 'open', 'file', 
        'os.', 'sys.', 'subprocess', 'rm ', 'del '
    ]
    
    code_lower = code.lower()
    for pattern in forbidden_patterns:
        if pattern in code_lower:
            return None, f"Security Error: Forbidden operation '{pattern}' detected"
    
    try:
        local_vars = {
            'df': df.copy(),
            'pd': pd,
            'datetime': datetime,
            'timedelta': timedelta,
            'alt': alt
        }
        exec(code, {"__builtins__": {}}, local_vars)
        
        outputs = {
            'result': local_vars.get('result'),
            'chart': local_vars.get('chart')
        }
        return outputs, None
    
    except Exception as e:
        return None, f"Execution Error: {str(e)}"

def generate_natural_response(user_question, main_result, code, client):
    if isinstance(main_result, pd.DataFrame):
        result_str = main_result.to_string(index=False)
    else:
        result_str = str(main_result)
    
    prompt = f"""Given this user question and query result, provide a clear, concise natural language answer.

User Question: {user_question}
Query Result: {result_str}

Provide a helpful, conversational response. Include numbers with currency symbol (₹) where appropriate.
If a chart is displayed below, mention "See the interactive chart below for a visual view."
Keep it brief and direct."""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a helpful financial assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=300
        )
        
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        if isinstance(main_result, (int, float)):
            return f"The result is ₹{main_result:,.2f}"
        return f"Result: {main_result}"

# Streamlit UI
st.title("💰 Transaction AI Assistant")
st.markdown("Ask questions about your expenses in natural language! Now with interactive charts where helpful.")

# Full Sidebar (restored to fix file_path not defined issue)
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # API Key Input
    st.subheader("🔑 Groq API Key")
    api_key_input = st.text_input(
        "Enter your Groq API Key:",
        type="password",
        value=st.session_state.get('groq_api_key', ''),
        help="Get your free API key from https://console.groq.com"
    )
    
    if api_key_input and st.button("Test & Save API Key"):
        is_valid, error_msg = validate_api_key(api_key_input)
        if is_valid:
            st.session_state.groq_api_key = api_key_input
            st.session_state.api_key_valid = True
            st.success("✅ API Key validated and saved!")
            st.rerun()
        else:
            st.session_state.api_key_valid = False
            st.error(f"❌ Invalid API Key: {error_msg}")
    
    if st.session_state.get('api_key_valid', False):
        st.success("✅ API Key is valid and ready!")
    elif api_key_input:
        st.warning("⚠️ Please test your API key using the button above.")
    else:
        env_key = os.environ.get("GROQ_API_KEY")
        if env_key:
            st.session_state.groq_api_key = env_key
            is_valid, error_msg = validate_api_key(env_key)
            if is_valid:
                st.session_state.api_key_valid = True
                st.success("✅ API Key from environment is valid!")
            else:
                st.session_state.api_key_valid = False
                st.error(f"❌ Invalid environment API Key: {error_msg}")
        else:
            st.warning("⚠️ Please enter your Groq API key above")
    
    st.markdown("---")
    
    file_format = st.radio("Select file format:", ["CSV", "JSON"])
    
    if file_format == "CSV":
        default_path = r"C:\Users\harshjain\Desktop\Harsh\Expense\output\expense.csv"
    else:
        default_path = r"C:\Users\harshjain\Desktop\Harsh\Expense\output\expense.json"
    
    file_path = st.text_input("File path:", value=default_path)
    
    if st.button("🔄 Load Data"):
        st.cache_data.clear()
        st.rerun()
    
    st.markdown("---")
    st.markdown("### 💡 Example Questions")
    st.markdown("""
    - How much did I spend yesterday?
    - Total expenses at BMTC?
    - Show all transactions above ₹100
    - Average daily spending in November
    - Top 5 merchants by spending
    - What's my total spending this month?
    - Show all transactions with cafe
    """)

# Fallback if file_path somehow not defined (prevents yellow warning in some editors)
if 'file_path' not in locals() and 'file_path' not in globals():
    file_path = default_path if 'default_path' in locals() else ""

# Load data
if not file_path or not os.path.exists(file_path):
    st.warning(f"⚠️ File not found: {file_path}")
    st.info("Please check and update the file path in the sidebar, then click 'Load Data' if needed.")
    st.stop()

df = load_transaction_data(file_path)

if df is not None and not df.empty:
    api_key = get_api_key()
    if not api_key:
        st.error("🔑 Please enter your Groq API Key in the sidebar.")
        st.stop()
    
    if not st.session_state.get('api_key_valid', False):
        st.error("🔑 Please validate your API Key using the button.")
        st.stop()
    
    try:
        client = get_groq_client(api_key)
    except Exception as e:
        st.error(f"❌ Failed to initialize Groq client: {str(e)}")
        st.stop()
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        total_debit = df[df['Type'] == 'DEBIT']['Amount'].sum()
        st.metric("Total Expenses", f"₹{total_debit:,.2f}")
    with col2:
        total_credit = df[df['Type'] == 'CREDIT']['Amount'].sum()
        st.metric("Total Income", f"₹{total_credit:,.2f}")
    with col3:
        st.metric("Transactions", len(df))
    with col4:
        avg_transaction = df[df['Type'] == 'DEBIT']['Amount'].mean()
        st.metric("Avg Expense", f"₹{avg_transaction:,.2f}" if avg_transaction > 0 else "₹0")
    
    st.markdown("---")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "code" in message:
                with st.expander("📊 View Query"):
                    st.code(message["code"], language="python")
            if "result_data" in message and isinstance(message["result_data"], pd.DataFrame):
                with st.expander("📋 View Data"):
                    st.dataframe(message["result_data"])
    
    if prompt := st.chat_input("Ask about your transactions..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("🤔 Analyzing..."):
                full_code = generate_pandas_query(prompt, df, client)
                
                if full_code:
                    display_code = clean_generated_code(full_code)
                    
                    outputs, error = execute_query_safely(full_code, df)
                    
                    if error:
                        response = f"⚠️ {error}"
                        st.error(response)
                    else:
                        main_result = outputs.get('result')
                        chart = outputs.get('chart')
                        
                        if main_result is None:
                            response = "No result was generated."
                        else:
                            response = generate_natural_response(prompt, main_result, full_code, client)
                        
                        st.markdown(response)
                        
                        if chart is not None:
                            try:
                                st.altair_chart(chart, use_container_width=True)
                            except Exception as viz_err:
                                st.warning(f"Could not display chart: {viz_err}")
                        
                        if isinstance(main_result, pd.DataFrame) and not main_result.empty:
                            with st.expander("📋 View Data"):
                                st.dataframe(main_result)
                        
                        with st.expander("📊 View Query"):
                            st.code(display_code, language="python")
                        
                        message_data = {
                            "role": "assistant",
                            "content": response,
                            "code": display_code
                        }
                        if isinstance(main_result, pd.DataFrame):
                            message_data["result_data"] = main_result
                        st.session_state.messages.append(message_data)
                else:
                    response = "Sorry, I couldn't generate a query for that question."
                    st.error(response)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response
                    })
    
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

else:
    st.error("Failed to load data or data is empty.")