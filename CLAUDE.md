# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TaiwanLoanInterest - 台灣貸款利率監控與資料收集系統。

目標：自動化收集央行及各銀行貸款利率資料，建立歷史資料庫，並與 [Python-Actions.GoodInfo.Analyzer](../Python-Actions.GoodInfo.Analyzer) 整合，提供股票投資決策所需的宏觀經濟數據。

## Key Data Sources

- **央行五大銀行統計**: https://www.cbc.gov.tw/tw/cp-528-1079-B4682-1.html
  - 檔案格式：XLS/ODS/PDF（檔名：5newloan）
  - 更新頻率：每月
  - 資料：購屋貸款、資本支出、週轉金、消費性貸款之金額與利率

- **央行存放款加權平均利率**: https://www.cbc.gov.tw/tw/cp-529-1081-195A7-1.html
  - 檔案格式：XLS/ODS/PDF（檔名：AVERAGEIR）
  - 更新頻率：每季
  - 資料：全體本國銀行存款與放款加權平均利率
  - 用途：無風險報酬基準、銀行獲利能力（存放利差）

- **央行金融統計**: https://www.cbc.gov.tw/tw/np-528-1.html

## Integration with GoodInfo.Analyzer

本專案資料將整合至 GoodInfo.Analyzer 的 Stage 4 進階校準模組：

| 本專案資料 | 整合目標 |
|-----------|---------|
| 央行政策利率 | P/E 估值倍數動態調整 |
| 定存利率指數 | 殖利率利差計算 |
| 存款加權平均利率 | 無風險報酬基準 |
| 房貸利率 | 營建/金融股前瞻指標 |
| 企業貸款利率 | 產業景氣判斷 |
| 放款加權平均利率 | 全市場企業融資成本 |
| 存放利差 | 金融股獲利能力指標 |

## Development Notes

- 資料輸出格式應為 CSV/JSON，便於 GoodInfo.Analyzer 讀取
- 需保留歷史時間序列資料
- 考慮建立標準化的資料 schema 供兩專案共用
