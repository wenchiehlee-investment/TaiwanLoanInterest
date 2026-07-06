#!/usr/bin/env python3
import argparse
import csv
import datetime as dt
import html
import re
from pathlib import Path
from zoneinfo import ZoneInfo

import requests


BANK_SOURCES = {
    "taishin": {
        "bank_name": "台新銀行",
        "url": "https://www.taishinbank.com.tw/TSB/personal/loan/ntd-loan-rate/detail/",
    },
    "mega": {
        "bank_name": "兆豐銀行",
        "url": "https://www.megabank.com.tw/personal/loan/credit-loan/cf-loan-rate",
    },
}


def _today() -> str:
    return dt.datetime.now(ZoneInfo("Asia/Taipei")).strftime("%Y-%m-%d")


def _timestamp() -> str:
    return dt.datetime.now(ZoneInfo("Asia/Taipei")).strftime("%Y-%m-%d %H:%M:%S CST")


def _normalize_text(page_html: str) -> str:
    text = re.sub(r"(?is)<script.*?</script>", " ", page_html)
    text = re.sub(r"(?is)<style.*?</style>", " ", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = html.unescape(text)
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n+", "\n", text)
    return text.strip()


def _parse_rate(value: str) -> float | None:
    value = value.strip().replace("％", "%")
    match = re.search(r"(\d+(?:\.\d+)?)\s*%", value)
    if not match:
        return None
    return float(match.group(1))


def _parse_taishin(page_html: str, source_url: str, as_of_date: str, process_ts: str) -> list[dict[str, str]]:
    text = _normalize_text(page_html)
    records: list[dict[str, str]] = []
    pattern = re.compile(
        r"(?P<rate_type>"
        r"(?:一個月期|三個月期|六個月期)定儲利率指數利率|"
        r"(?:一年期|二年期)之郵政定期儲金利率|"
        r"(?:一個月期|三個月期)之(?:基準利率|基本放款利率)"
        r")\s+"
        r"(?P<period>(?:\d{3}\.\d{2}\.\d{2}\s*~\s*(?:\d{3}\.\d{2}\.\d{2}|迄今))|-)\s+"
        r"(?P<rate>(?:\d+(?:\.\d+)?\s*[%％])|-)"
    )
    for match in pattern.finditer(text):
        rate = _parse_rate(match.group("rate"))
        if rate is None:
            continue
        records.append(
            {
                "as_of_date": as_of_date,
                "bank_code": "taishin",
                "bank_name": "台新銀行",
                "product_group": "ntd_loan_reference_rate",
                "rate_type": match.group("rate_type"),
                "rate_percent": f"{rate:.3f}",
                "applicable_period": re.sub(r"\s+", "", match.group("period")),
                "source_url": source_url,
                "note": "",
                "download_timestamp": process_ts,
                "process_timestamp": process_ts,
            }
        )
    return records


def _parse_mega(page_html: str, source_url: str, as_of_date: str, process_ts: str) -> list[dict[str, str]]:
    text = _normalize_text(page_html)
    records: list[dict[str, str]] = []

    legacy_match = re.search(
        r"原交銀指數型房貸指數.*?(?P<effective>\d{3}年\d{1,2}月\d{1,2}日)"
        r".*?平均數\s*(?P<rate>\d+(?:\.\d+)?)\s*[％%]"
        r"(?:.*?下次變動日\s*(?P<next>\d{3}年\d{1,2}月\d{1,2}日))?",
        text,
    )
    if legacy_match:
        note = "原交銀借戶適用；頁面揭露為特定舊貸戶指數。"
        if legacy_match.group("next"):
            note += f" 下次變動日：{legacy_match.group('next')}。"
        records.append(
            {
                "as_of_date": as_of_date,
                "bank_code": "mega",
                "bank_name": "兆豐銀行",
                "product_group": "legacy_mortgage_index",
                "rate_type": "原交銀指數型房貸指數",
                "rate_percent": f"{float(legacy_match.group('rate')):.3f}",
                "applicable_period": f"{legacy_match.group('effective')}起",
                "source_url": source_url,
                "note": note,
                "download_timestamp": process_ts,
                "process_timestamp": process_ts,
            }
        )

    if "消費金融放款指標利率" in text:
        records.append(
            {
                "as_of_date": as_of_date,
                "bank_code": "mega",
                "bank_name": "兆豐銀行",
                "product_group": "consumer_finance_indicator",
                "rate_type": "消費金融放款指標利率計算規則",
                "rate_percent": "",
                "applicable_period": "每月或每季定期調整，依新舊貸戶規則",
                "source_url": source_url,
                "note": "指標利率以六大行庫一年期定期儲蓄存款機動利率平均數計算；頁面主要揭露計算規則。",
                "download_timestamp": process_ts,
                "process_timestamp": process_ts,
            }
        )

    return records


def fetch_page(url: str, verify: bool = True) -> str:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "zh-TW,zh;q=0.9,en;q=0.8",
    }
    resp = requests.get(url, headers=headers, timeout=30, verify=verify)
    resp.raise_for_status()
    return resp.text


def write_csv(records: list[dict[str, str]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "as_of_date",
        "bank_code",
        "bank_name",
        "product_group",
        "rate_type",
        "rate_percent",
        "applicable_period",
        "source_url",
        "note",
        "download_timestamp",
        "process_timestamp",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect bank-level loan reference rates for 台新 and 兆豐.")
    parser.add_argument("--out-dir", default="data/bank_loan_rates", help="Output directory")
    parser.add_argument("--insecure", action="store_true", help="Disable TLS certificate verification")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    as_of_date = _today()
    process_ts = _timestamp()

    records: list[dict[str, str]] = []
    for bank_code, source in BANK_SOURCES.items():
        page_html = fetch_page(source["url"], verify=not args.insecure)
        (raw_dir / f"{bank_code}_{as_of_date}.html").write_text(page_html, encoding="utf-8")
        if bank_code == "taishin":
            records.extend(_parse_taishin(page_html, source["url"], as_of_date, process_ts))
        elif bank_code == "mega":
            records.extend(_parse_mega(page_html, source["url"], as_of_date, process_ts))

    write_csv(records, out_dir / f"bank_loan_rates_{as_of_date}.csv")
    write_csv(records, out_dir / "raw_bank_loan_rates.csv")


if __name__ == "__main__":
    main()
