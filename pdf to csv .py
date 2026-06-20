import pdfplumber
import csv
import json
import re
import os
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

# Configuration
PDF_PATH = "Expense.pdf"  # Change this to your PDF path
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

CSV_PATH = OUTPUT_DIR / "expense.csv"
JSON_PATH = OUTPUT_DIR / "expense.json"

class TransactionParser:
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.transactions = []
        
    def extract_text_blocks(self) -> List[str]:
        """Extract text blocks from PDF with robust parsing"""
        print("📄 Extracting text blocks from PDF...")
        blocks = []
        
        try:
            with pdfplumber.open(self.pdf_path) as pdf:
                total_pages = len(pdf.pages)
                print(f"📑 Found {total_pages} pages")
                
                for page_num, page in enumerate(pdf.pages, 1):
                    print(f"Processing page {page_num}/{total_pages}...", end=" ")
                    
                    # Try multiple extraction methods
                    text = self._extract_page_text(page)
                    if not text:
                        print("⚠️ No text extracted")
                        continue
                    
                    lines = [line.strip() for line in text.split("\n") if line.strip()]
                    print(f"📝 {len(lines)} lines")
                    
                    # Split into transaction blocks
                    current_block = []
                    for line in lines:
                        # Skip junk lines
                        if self._is_junk_line(line):
                            continue
                        
                        # Start new block if we find a date or transaction indicator
                        if self._is_transaction_start(line):
                            if current_block:
                                blocks.append(" ".join(current_block))
                                current_block = []
                        
                        current_block.append(line)
                    
                    # Add last block
                    if current_block:
                        blocks.append(" ".join(current_block))
                        
        except Exception as e:
            print(f"❌ Error reading PDF: {e}")
            return []
        
        print(f"✅ Extracted {len(blocks)} text blocks")
        return blocks
    
    def _extract_page_text(self, page) -> Optional[str]:
        """Extract text from page with fallback methods"""
        try:
            # Method 1: Standard extract_text
            text = page.extract_text()
            if text and text.strip():
                return text
        except Exception as e:
            print(f"  extract_text failed: {e}")
        
        try:
            # Method 2: Extract words and reconstruct
            words = page.extract_words()
            if words:
                # Sort words by position to maintain reading order
                sorted_words = sorted(words, key=lambda w: (w['top'], w['x0']))
                text = ' '.join(word['text'] for word in sorted_words if word['text'].strip())
                if text.strip():
                    return text
        except Exception as e:
            print(f"  extract_words failed: {e}")
        
        return None
    
    def _is_junk_line(self, line: str) -> bool:
        """Check if line is junk (page numbers, headers, etc.)"""
        junk_patterns = [
            r'Page \d+ of \d+',
            r'Statement Period',
            r'Account Summary',
            r'Opening Balance',
            r'Closing Balance',
            r'Total Credits',
            r'Total Debits',
            r'Available Balance',
            r'^[A-Z]{2,}\s*$',  # All caps headers
            r'^\d{1,2}\s*$',     # Just numbers (page numbers)
        ]
        
        for pattern in junk_patterns:
            if re.match(pattern, line, re.IGNORECASE):
                return True
        return False
    
    def _is_transaction_start(self, line: str) -> bool:
        """Check if line indicates start of new transaction"""
        start_indicators = [
            r'^\w{3}\s+\d{1,2},?\s+\d{4}',  # Date: Sep 07, 2025
            r'Paid to',
            r'₹\s*\d+',                     # Amount with rupee symbol
            r'Transaction ID',
        ]
        for indicator in start_indicators:
            if re.search(indicator, line):
                return True
        return False
    
    def parse_transactions(self, blocks: List[str]) -> List[Dict]:
        """Parse transaction details from text blocks"""
        print("🔍 Parsing transactions...")
        parsed_transactions = []
        
        for i, block in enumerate(blocks, 1):
            if len(block.strip()) < 10:  # Skip very short blocks
                continue
                
            transaction = self._parse_single_transaction(block)
            if transaction and transaction.get('Date') and transaction.get('Amount'):
                parsed_transactions.append(transaction)
                print(f"  {i:3d}: {transaction['Date']} - {transaction['Amount']}")
        
        print(f"✅ Parsed {len(parsed_transactions)} valid transactions")
        return parsed_transactions
    
    def _parse_single_transaction(self, block: str) -> Optional[Dict]:
        """Parse a single transaction block"""
        transaction = {
            "Date": "",
            "Merchant": "",
            "Transaction ID": "",
            "UTR No.": "",
            "Paid by": "",
            "Type": "",
            "Amount": ""
        }
        
        # Enhanced regex patterns
        patterns = {
            "Date": r'([A-Za-z]{3}\s+\d{1,2},?\s+\d{4})',
            "Merchant": r'Paid to\s+([^\n\r]+?)(?=\s*(?:Transaction ID|UTR No\.?|Paid by|DEBIT|CREDIT|₹|$))',
            "Transaction ID": r'Transaction ID\s+([A-Z][A-Z0-9]{15,})',
            "UTR No.": r'UTR No\.\s*([0-9]{10,})',
            "Paid by": r'Paid by\s+([A-Z0-9]{4,})',
            "Type": r'\b(DEBIT|CREDIT)\b',
            "Amount": r'(₹\s*\d{1,3}(?:,\d{3})*(?:\.\d{2})?)'
        }
        
        # Extract using patterns
        for field, pattern in patterns.items():
            match = re.search(pattern, block, re.IGNORECASE)
            if match:
                value = match.group(1).strip()
                if field == "Amount":
                    # Clean amount format
                    value = re.sub(r'\s+', ' ', value)
                elif field == "Date":
                    # Convert date format here
                    value = self._convert_date_format(value)
                transaction[field] = value
        
        # Fallback for merchant if regex fails
        if not transaction["Merchant"]:
            # Try to find text between "Paid to" and next field
            if "Paid to" in block:
                after_paid = block.split("Paid to", 1)[1]
                # Take first meaningful text before next field
                parts = re.split(r'(Transaction ID|UTR No\.?|Paid by|DEBIT|CREDIT|₹)', after_paid, 1)
                if parts and parts[0].strip():
                    transaction["Merchant"] = parts[0].strip()
        
        # Fallback for type
        if not transaction["Type"]:
            if "DEBIT" in block.upper():
                transaction["Type"] = "DEBIT"
            elif "CREDIT" in block.upper():
                transaction["Type"] = "CREDIT"
        
        # Fallback for amount
        if not transaction["Amount"]:
            # Look for any ₹ amount
            amount_match = re.search(r'₹\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)', block)
            if amount_match:
                transaction["Amount"] = f"₹ {amount_match.group(1)}"
        
        # Fallback for date conversion
        if transaction["Date"] and not re.match(r'\d{2}-\d{2}-\d{2}', transaction["Date"]):
            transaction["Date"] = self._convert_date_format(transaction["Date"])
        
        # Clean up merchant name (remove extra text)
        if transaction["Merchant"]:
            # Remove common suffixes
            suffixes = ['via UPI', 'UPI', '(UPI)', 'UPI Payment']
            for suffix in suffixes:
                transaction["Merchant"] = transaction["Merchant"].replace(suffix, '').strip()
        
        return transaction if transaction["Date"] and transaction["Amount"] else None
    
    def _convert_date_format(self, date_str: str) -> str:
        """
        Convert date from 'Sep 07, 2025' to '07-09-25' format
        
        Args:
            date_str: Input date string like "Sep 07, 2025"
            
        Returns:
            Date string in dd-mm-yy format
        """
        try:
            # Handle both "Sep 07, 2025" and "Sep 07 2025" formats
            date_str = date_str.replace(",", "")
            
            # Parse the date
            date_obj = datetime.strptime(date_str, "%b %d %Y")
            
            # Convert to dd-mm-yy format
            formatted_date = date_obj.strftime("%d-%m-%y")
            
            return formatted_date
            
        except ValueError as e:
            print(f"⚠️ Date conversion failed for '{date_str}': {e}")
            # Return original if conversion fails
            return date_str
    
    def save_output(self, transactions: List[Dict]):
        """Save transactions to CSV and JSON"""
        if not transactions:
            print("❌ No transactions to save")
            return
        
        # Create DataFrame
        df = pd.DataFrame(transactions)
        
        # Save CSV
        df.to_csv(CSV_PATH, index=False, encoding="utf-8")
        print(f"💾 CSV saved: {CSV_PATH}")
        
        # Save JSON
        with open(JSON_PATH, 'w', encoding='utf-8') as f:
            json.dump(transactions, f, indent=2, ensure_ascii=False)
        print(f"💾 JSON saved: {JSON_PATH}")
        
        # Print summary
        
        print(f"✅ Total transactions: {len(transactions)}")
        
        # Convert date column to datetime for better analysis
        df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%y', errors='coerce')
        if not df['Date'].isna().all():
            print(f"📅 Date range: {df['Date'].min().strftime('%d-%m-%y')} to {df['Date'].max().strftime('%d-%m-%y')}")
        
        # Calculate total amount (remove ₹ and convert to numeric)
        df['Amount_numeric'] = df['Amount'].str.replace('₹', '').str.replace(',', '').astype(float)
        print(f"💰 Total amount: ₹{df['Amount_numeric'].sum():,.2f}")
        
        # Show sample
        
        sample_df = df[['Date', 'Merchant', 'Type', 'Amount']].head()
        print(sample_df.to_string(index=False, max_colwidth=30))
    
    def run(self):
        """Main execution method"""
        print("🚀 Starting PDF Transaction Parser")
        print(f"📁 Input: {self.pdf_path}")
        print(f"📂 Output: {OUTPUT_DIR}")
        print("-" * 50)
        
        # Check if PDF exists
        if not Path(self.pdf_path).exists():
            print(f"❌ PDF file not found: {self.pdf_path}")
            return
        
        # Extract blocks
        blocks = self.extract_text_blocks()
        if not blocks:
            print("❌ No text blocks extracted")
            return
        
        # Parse transactions
        transactions = self.parse_transactions(blocks)
        
        # Save output
        self.save_output(transactions)
        
        print("\n🎉 Processing complete!")
        
        # Print chatbot-friendly query examples
        print("\n🤖 CHATBOT QUERY EXAMPLES:")
        print("```python")
        print("# Load the data")
        print("df = pd.read_csv('output/expense.csv')")
        print("df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%y')")
        print("df['Amount'] = df['Amount'].str.replace('₹', '').str.replace(',', '').astype(float)")
        print()
        print("# Date range queries")
        print("recent_transactions = df[df['Date'] >= '01-09-25']")
        print("september_expenses = df[(df['Date'] >= '01-09-25') & (df['Date'] <= '30-09-25')]")
        print()
        print("# Merchant queries")
        print("cafe_expenses = df[df['Merchant'].str.contains('cafe', case=False, na=False)]")
        print("all_chai = df[df['Merchant'].str.contains('chai', case=False, na=False)]")
        print()
        print("# Amount queries")
        print("big_spends = df[df['Amount'] > 1000]")
        print("daily_average = df['Amount'].mean()")
        print()
        print("# Type queries")
        print("debits_only = df[df['Type'] == 'DEBIT']")
        print("total_debits = debits_only['Amount'].sum()")
        print("```")

def main():
    # You can change the PDF path here
    parser = TransactionParser(PDF_PATH)
    parser.run()

if __name__ == "__main__":
    main()