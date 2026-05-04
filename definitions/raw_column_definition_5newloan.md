---
source: https://raw.githubusercontent.com/wenchiehlee-investment/TaiwanLoanInterest/refs/heads/main/definitions/raw_column_definition_5newloan.md
destination: https://raw.githubusercontent.com/wenchiehlee-money/biztrends.TW/refs/heads/main/definitions/raw_column_definition_5newloan.md
---

# Raw Column Definitions - 五大銀行新承作放款 (5newloan)

**Source repo:** `wenchiehlee-investment/TaiwanLoanInterest`  
**Script:** `src/5newloan.py`  
**Raw file:** `data/5newloan/raw_5newloan.csv`  
**Synced to:** `biztrends.TW/data/loaninterest_raw/raw_5newloan.csv`  
**CBC source:** 中央銀行金融統計月報 表 12

**Purpose**: 五大銀行新承作放款統計——依貸款用途分類的月度金額與加權平均利率。

**Nature**: 每月由 CI 自動爬取 CBC 網站並累積更新。資料從 2011-01 起至最新月份，共約 183 筆月度資料。

| Column | Type | Description | Example |
|---|---|---|---|
| `年月` | string | 資料年月，格式 `YYYY-MM` | `2026-03` |
| `購屋貸款_金額` | integer | 購屋貸款新承作金額，單位：百萬元 | `59024` |
| `購屋貸款_利率` | float | 購屋貸款加權平均利率，單位：% | `2.306` |
| `資本支出貸款_金額` | integer | 資本支出貸款新承作金額，單位：百萬元 | `80688` |
| `資本支出貸款_利率` | float | 資本支出貸款加權平均利率，單位：% | `2.426` |
| `週轉金貸款_金額` | integer | 週轉金貸款新承作金額，單位：百萬元 | `1066719` |
| `週轉金貸款_利率` | float | 週轉金貸款加權平均利率，單位：% | `2.066` |
| `消費性貸款_金額` | integer | 消費性貸款新承作金額，單位：百萬元 | `8247` |
| `消費性貸款_利率` | float | 消費性貸款加權平均利率，單位：% | `2.587` |
| `合計_金額` | integer | 所有用途合計新承作金額，單位：百萬元 | `1214678` |
| `合計_加權平均利率` | float | 全體新承作放款加權平均利率（含國庫借款），單位：% | `2.105` |
| `不含國庫借款之加權平均利率` | float | 全體新承作放款加權平均利率（不含國庫借款），單位：% | `2.142` |
| `不含國庫借款之加權平均利率_1` | float | 同上欄備用欄位，CBC 原始表格有時有兩欄；通常為空值 | `` |
