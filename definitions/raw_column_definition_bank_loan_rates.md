---
source: https://raw.githubusercontent.com/wenchiehlee-investment/TaiwanLoanInterest/refs/heads/main/definitions/raw_column_definition_bank_loan_rates.md
destination: https://raw.githubusercontent.com/wenchiehlee-money/biztrends.TW/refs/heads/main/definitions/raw_column_definition_bank_loan_rates.md
---

# Raw Column Definitions - Bank Loan Rates

**Source repo:** `wenchiehlee-investment/TaiwanLoanInterest`  
**Script:** `src/bank_loan_rates.py`  
**Raw file:** `data/bank_loan_rates/raw_bank_loan_rates.csv`  
**Primary banks:** 台新銀行、兆豐銀行

**Purpose**: 追蹤台新與兆豐的銀行層級放款參考利率、指標利率與相關揭露，作為本專案由市場總體利率延伸到個別銀行貸款成本的資料集。

**Nature**: 從銀行公開頁面擷取當日揭露內容。銀行頁面通常揭露參考利率或計算規則，不一定等同實際核貸利率；實際貸款利率仍取決於產品、客戶條件、擔保品與銀行審核。

| Column | Type | Description | Example |
|---|---|---|---|
| `as_of_date` | string | 擷取日期，格式 `YYYY-MM-DD` | `2026-07-06` |
| `bank_code` | string | 銀行代碼 | `taishin` |
| `bank_name` | string | 銀行中文名稱 | `台新銀行` |
| `product_group` | string | 利率或產品群組 | `ntd_loan_reference_rate` |
| `rate_type` | string | 頁面揭露的利率名稱或規則名稱 | `一個月期定儲利率指數利率` |
| `rate_percent` | float | 年利率百分比；若來源只揭露規則則留空 | `1.740` |
| `applicable_period` | string | 適用期間或生效描述 | `115.06.22~115.07.20` |
| `source_url` | string | 原始公開來源網址 | `https://www.taishinbank.com.tw/TSB/personal/loan/ntd-loan-rate/detail/` |
| `note` | string | 限制、公式或適用範圍備註 | `頁面主要揭露計算規則。` |
| `download_timestamp` | string | 原始頁面下載時間 (CST) | `2026-07-06 09:00:00 CST` |
| `process_timestamp` | string | CSV 產生或清洗完成時間 (CST) | `2026-07-06 09:00:00 CST` |
