# 💰 Expense AI Assistant

An AI-powered personal finance assistant that lets you analyze your expenses using natural language.

Instead of writing SQL or Pandas queries, simply ask questions like:

* "How much did I spend on food last month?"
* "Show all transactions above ₹500"
* "Which merchant received the highest payment?"
* "What is my average daily spending?"
* "Show spending trends over time"

The system converts natural language into executable Pandas queries using an LLM and returns results in a conversational format.

---

## 🚀 Features

### 📄 PDF Statement Processing

* Extracts transactions from bank/UPI statement PDFs
* Robust PDF text extraction using pdfplumber
* Automatic transaction parsing
* Outputs structured CSV and JSON datasets
* Supports transaction metadata extraction:

  * Date
  * Merchant
  * Transaction ID
  * UTR Number
  * Payment Method
  * Transaction Type
  * Amount

### 🤖 Natural Language Expense Analytics

* Ask questions in plain English
* LLM converts questions into Pandas queries
* Safe execution environment
* Automatic response generation
* No SQL knowledge required

### 📊 Interactive Dashboard

* Built with Streamlit
* Expense and income summaries
* Transaction statistics
* Chat-style interface
* Query transparency (generated code can be viewed)

### 📈 Visual Analytics

* Interactive charts using Altair
* Merchant-wise spending analysis
* Monthly expense trends
* Spending distribution visualization
* Dynamic chart generation based on user queries

---

## 🏗️ Architecture

```text
PDF Statement
      │
      ▼
PDF Parser
      │
      ▼
Structured CSV / JSON
      │
      ▼
Natural Language Query
      │
      ▼
Groq LLM
      │
      ▼
Generated Pandas Query
      │
      ▼
Secure Execution Engine
      │
      ▼
Results + Visualization
```

---

## 🛠️ Tech Stack

### Backend

* Python
* Pandas
* Groq API
* Regex
* JSON

### Frontend

* Streamlit

### Visualization

* Altair

### PDF Processing

* pdfplumber

---

## 📂 Project Structure

```text
Expense/
│
├── Chatbot.py
├── Chatbot_plot.py
├── pdf to csv.py
│
├── output/
│   ├── expense.csv
│   ├── expense.json
│
├── transactions.db
├── README.md
└── requirements.txt
```

---

## ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/HarshJain006/Expense.git
cd Expense
```

### Create Virtual Environment

```bash
python -m venv .venv
```

### Activate Environment

Windows:

```bash
.venv\Scripts\activate
```

Linux/Mac:

```bash
source .venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🔑 Configure API Key

Create a `.env` file:

```env
GROQ_API_KEY=your_api_key_here
```

Or enter the key directly through the Streamlit sidebar.

---

## ▶️ Run Application

Basic Chatbot:

```bash
streamlit run Chatbot.py
```

Interactive Visualization Version:

```bash
streamlit run Chatbot_plot.py
```

---

## 💬 Example Queries

```text
How much did I spend yesterday?

Show all transactions above ₹1000

Top 5 merchants by spending

Total expenses at BMTC

Average daily spending this month

Show transactions related to cafes

Monthly spending trend
```

---

## 🔒 Security

The application includes:

* Safe query execution sandbox
* Restricted code execution
* Blocked dangerous operations
* No direct filesystem access from generated queries

---

## 📸 Future Enhancements

* Multi-bank statement support
* Voice-based expense queries
* Budget tracking
* Expense categorization
* AI-generated spending insights
* Forecasting future expenses
* Mobile application support

---

## 👨‍💻 Author

**Harsh Jain**

AI-powered Finance Analytics using:

* Streamlit
* Pandas
* Groq LLM
* Altair
* PDF Processing

If you found this project useful, consider giving it a ⭐ on GitHub.
