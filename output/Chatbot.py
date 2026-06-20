import streamlit as st
import pandas as pd
import json
import os
from groq import Groq
import re
from datetime import datetime, timedelta
import httpx

# Page configuration
st.set_page_config(
    page_title="Transaction AI Assistant",
    page_icon="💰",
    layout="wide"
)

# Initialize Groq client
@st.cache_resource
def get_groq_client(api_key):
    return Groq(api_key=api_key, http_client=httpx.Client(proxies=None))

def get_api_key():
    """Get API key from multiple sources"""
    # Try session state first
    if 'groq_api_key' in st.session_state and st.session_state.groq_api_key:
        return st.session_state.groq_api_key
    
    # Try environment variable
    api_key = os.environ.get("GROQ_API_KEY")
    if api_key:
        return api_key
    
    return None

def validate_api_key(api_key):
    """Validate API key by making a test API call"""
    try:
        client = Groq(api_key=api_key, http_client=httpx.Client(proxies=None))
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
    """Load transaction data from JSON or CSV with robust error handling"""
    try:
        if file_path.endswith('.json'):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            df = pd.DataFrame(data)
        elif file_path.endswith('.csv'):
            df = pd.read_csv(file_path, encoding='utf-8')
            # Handle CSV without headers
            if df.columns[0] != 'Date':
                df.columns = ['Date', 'Merchant', 'Transaction ID', 'UTR No.', 'Paid by', 'Type', 'Amount']
        else:
            st.error("Unsupported file format. Use JSON or CSV.")
            return None
        
        # Data preprocessing
        df = preprocess_dataframe(df)
        return df
    
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def preprocess_dataframe(df):
    """Clean and standardize the dataframe"""
    # Remove currency symbols and commas from Amount
    df['Amount'] = df['Amount'].astype(str).str.replace('₹', '').str.replace(',', '').str.strip()
    df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce').fillna(0)
    
    # Parse dates flexibly
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%y', errors='coerce')
    
    # Standardize Type column
    df['Type'] = df['Type'].str.upper().str.strip()
    
    # Fill missing Merchant names
    df['Merchant'] = df['Merchant'].fillna('Unknown')
    
    # Add derived columns for better analysis
    df['Month'] = df['Date'].dt.month_name()
    df['Year'] = df['Date'].dt.year
    df['Day'] = df['Date'].dt.day
    df['Weekday'] = df['Date'].dt.day_name()
    
    return df

def get_dataframe_schema(df):
    """Generate schema description for the AI"""
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
- Amount is numeric (already cleaned from currency format)
- Date is datetime format
- Use df['Type'] == 'DEBIT' for expenses, 'CREDIT' for income
"""
    return schema

def clean_generated_code(code):
    """Clean the generated code to extract only the core pandas query for display"""
    # Remove imports
    code = re.sub(r'^import\s+.*$', '', code, flags=re.MULTILINE)
    code = re.sub(r'^from\s+.*$', '', code, flags=re.MULTILINE)
    
    # Remove try-except blocks
    code = re.sub(r'try:.*?(?=except|result\s*=|$)', '', code, flags=re.DOTALL | re.MULTILINE)
    code = re.sub(r'except\s+.*?:.*?(?=print|\n\n|$)', '', code, flags=re.DOTALL | re.MULTILINE)
    
    # Remove print statements
    code = re.sub(r'print\([^)]*\)', '', code)
    
    # Extract lines that lead to 'result ='
    lines = code.split('\n')
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        if line.startswith('result =') or (line and not line.startswith('#')):
            cleaned_lines.append(line)
    
    # Join and clean up
    cleaned_code = '\n'.join(cleaned_lines).strip()
    if not cleaned_code:
        cleaned_code = code.strip()  # Fallback
    
    return cleaned_code

def generate_pandas_query(user_question, df, client):
    """Use Groq AI to generate pandas query"""
    
    schema = get_dataframe_schema(df)
    
    system_prompt = f"""You are an expert Python pandas data analyst. Generate SAFE and EFFICIENT pandas queries.

{schema}

AVAILABLE: pd (pandas), datetime, timedelta. NO IMPORTS REQUIRED!

CRITICAL RULES:
1. Return ONLY executable Python code, no explanations or markdown
2. Use 'df' as the DataFrame variable name
3. Store final result in a variable called 'result'
4. NO try-except blocks or print statements - assume data is clean
5. For aggregations, return the final numeric value or DataFrame
6. Use vectorized operations, avoid loops
7. Be case-insensitive for merchant names (use .str.contains with case=False)
8. For date queries, use datetime.now().date() - timedelta(days=1) for yesterday
9. Consider both DEBIT and CREDIT when calculating totals unless specified
10. NO IMPORT STATEMENTS AT ALL!

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

User: "Average daily spending in November"
Code:
result = df[(df['Date'].dt.month == 11) & (df['Type'] == 'DEBIT')].groupby('Day')['Amount'].sum().mean()

User: "Show transactions from yesterday"
Code:
yesterday = datetime.now().date() - timedelta(days=1)
result = df[(df['Date'].dt.date == yesterday) & (df['Type'] == 'DEBIT')][['Date', 'Merchant', 'Amount', 'Type']]
"""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Generate pandas query for: {user_question}"}
            ],
            temperature=0.1,
            max_tokens=500
        )
        
        code = response.choices[0].message.content.strip()
        # Clean the code
        code = re.sub(r'^```python\n?', '', code)
        code = re.sub(r'^```\n?', '', code)
        code = re.sub(r'\n?```$', '', code)
        
        return code.strip()
    
    except Exception as e:
        st.error(f"Error generating query: {str(e)}")
        return None

def execute_query_safely(code, df):
    """Execute pandas query with safety checks"""
    # Security: Block dangerous operations
    forbidden_patterns = [
        'import', 'exec', 'eval', '__', 'open', 'file', 
        'os.', 'sys.', 'subprocess', 'rm ', 'del '
    ]
    
    code_lower = code.lower()
    for pattern in forbidden_patterns:
        if pattern in code_lower:
            return None, f"Security Error: Forbidden operation '{pattern}' detected"
    
    try:
        # Create safe execution environment
        local_vars = {'df': df.copy(), 'pd': pd, 'datetime': datetime, 'timedelta': timedelta}
        exec(code, {"__builtins__": {}}, local_vars)
        result = local_vars.get('result', None)
        return result, None
    
    except Exception as e:
        return None, f"Execution Error: {str(e)}"

def generate_natural_response(user_question, result, code, client):
    """Generate natural language response using Groq"""
    
    result_str = str(result)
    if isinstance(result, pd.DataFrame):
        result_str = result.to_string()
    
    prompt = f"""Given this user question and query result, provide a clear, concise natural language answer.

User Question: {user_question}
Query Result: {result_str}

Provide a helpful, conversational response. Include numbers with currency symbol (₹) where appropriate.
Keep it brief and direct."""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a helpful financial assistant. Provide clear, concise answers."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=300
        )
        
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        # Fallback response
        if isinstance(result, (int, float)):
            return f"The result is ₹{result:,.2f}"
        return str(result)

# Streamlit UI
st.title("💰 Transaction AI Assistant")
st.markdown("Ask questions about your expenses in natural language!")

# Sidebar for file selection
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
    
    # Test API Key Button
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
            # Auto-validate env key
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
    """)

# Load data
if 'file_path' not in locals():
    file_path = default_path  # Fallback if not defined

if os.path.exists(file_path):
    df = load_transaction_data(file_path)
    
    if df is not None and not df.empty:
        # Check API key before proceeding
        api_key = get_api_key()
        if not api_key:
            st.error("🔑 Please enter your Groq API Key in the sidebar to use the chatbot.")
            st.info("Get your free API key from: https://console.groq.com")
            st.stop()
        
        if not st.session_state.get('api_key_valid', False):
            st.error("🔑 Please validate your API Key using the 'Test & Save API Key' button in the sidebar.")
            st.stop()
        
        # Initialize client with API key
        try:
            client = get_groq_client(api_key)
        except Exception as e:
            st.error(f"❌ Failed to initialize Groq client: {str(e)}")
            st.stop()
        
        # Display data summary
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
            st.metric("Avg Transaction", f"₹{avg_transaction:,.2f}")
        
        st.markdown("---")
        
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if "code" in message:
                    with st.expander("📊 View Query"):
                        st.code(message["code"], language="python")
                if "result_data" in message and isinstance(message["result_data"], pd.DataFrame):
                    with st.expander("📋 View Data"):
                        st.dataframe(message["result_data"])
        
        # Chat input
        if prompt := st.chat_input("Ask about your transactions..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate and execute query
            with st.chat_message("assistant"):
                with st.spinner("🤔 Analyzing..."):
                    # Generate query
                    full_code = generate_pandas_query(prompt, df, client)
                    
                    if full_code:
                        # Clean for display
                        display_code = clean_generated_code(full_code)
                        
                        # Execute full code
                        result, error = execute_query_safely(full_code, df)
                        
                        if error:
                            response = f"⚠️ {error}"
                            st.error(response)
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": response,
                                "code": display_code
                            })
                        else:
                            # Generate natural response
                            response = generate_natural_response(prompt, result, full_code, client)
                            st.markdown(response)
                            
                            # Show cleaned query in expander
                            with st.expander("📊 View Query"):
                                st.code(display_code, language="python")
                            
                            # Show data if DataFrame
                            if isinstance(result, pd.DataFrame) and not result.empty:
                                with st.expander("📋 View Data"):
                                    st.dataframe(result)
                            
                            # Save to history with cleaned code
                            message_data = {
                                "role": "assistant",
                                "content": response,
                                "code": display_code
                            }
                            if isinstance(result, pd.DataFrame):
                                message_data["result_data"] = result
                            
                            st.session_state.messages.append(message_data)
                    else:
                        response = "Sorry, I couldn't generate a query for that question."
                        st.error(response)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response
                        })
        
        # Clear chat button
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()
    else:
        st.error("Failed to load data or data is empty.")
else:
    st.warning(f"⚠️ File not found: {file_path}")
    st.info("Please check the file path in the sidebar.")