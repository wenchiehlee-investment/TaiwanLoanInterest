---
source: https://raw.githubusercontent.com/wenchiehlee-investment/TaiwanLoanInterest/refs/heads/main/definitions/raw_column_definition_averageir.md
destination: https://raw.githubusercontent.com/wenchiehlee-money/biztrends.TW/refs/heads/main/definitions/raw_column_definition_averageir.md
---

# Raw Column Definitions - 各類金融機構存放款平均利率 (averageir)

**Source repo:** `wenchiehlee-investment/TaiwanLoanInterest`  
**Script:** `src/averageir.py`  
**Raw file:** `data/averageir/raw_averageir.csv`  
**Synced to:** `biztrends.TW/data/loaninterest_raw/raw_averageir.csv`  
**CBC source:** 中央銀行金融統計月報 表 13

**Purpose**: 各類金融機構存放款平均利率統計——本國銀行、外商銀行、信用合作社、農漁會信用部、信託投資公司的季度存款利率、放款利率與利差。

**Nature**: 每月由 CI 自動爬取 CBC 網站並累積更新。資料從 1982Q1 起至最新季度，共約 175 筆季度資料。信託投資公司自 2000 年代後已無業務，相關欄位後期為空值。

| Column | Type | Description | Example |
|---|---|---|---|
| `季別` | string | 資料季別，格式 `YYYYQN` | `2025Q4` |
| `本國銀行_存款` | float | 本國銀行平均存款利率，單位：% | `1.15` |
| `本國銀行_放款` | float | 本國銀行平均放款利率，單位：% | `2.52` |
| `外商銀行_存款` | float | 外商銀行平均存款利率，單位：% | `1.02` |
| `外商銀行_放款` | float | 外商銀行平均放款利率，單位：% | `2.44` |
| `信用合作社_存款` | float | 信用合作社平均存款利率，單位：% | `1.12` |
| `信用合作社_放款` | float | 信用合作社平均放款利率，單位：% | `2.76` |
| `農漁會信用部_存款` | float | 農漁會信用部平均存款利率，單位：% | `0.87` |
| `農漁會信用部_放款` | float | 農漁會信用部平均放款利率，單位：% | `2.57` |
| `信託投資公司_存款` | float | 信託投資公司平均存款利率，單位：%；2000 年代後為空值 | `` |
| `信託投資公司_放款` | float | 信託投資公司平均放款利率，單位：%；2000 年代後為空值 | `` |
| `信託投資公司` | float | 信託投資公司其他利率欄位（CBC 原始表格備用欄）；通常為空值 | `` |
| `本國銀行_利差` | float | 本國銀行放款利率 − 存款利率，單位：% | `1.37` |
| `外商銀行_利差` | float | 外商銀行放款利率 − 存款利率，單位：% | `1.42` |
| `信用合作社_利差` | float | 信用合作社放款利率 − 存款利率，單位：% | `1.64` |
| `農漁會信用部_利差` | float | 農漁會信用部放款利率 − 存款利率，單位：% | `1.70` |
| `信託投資公司_利差` | float | 信託投資公司放款利率 − 存款利率，單位：%；2000 年代後為空值 | `` |
