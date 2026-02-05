#!/usr/bin/env python3
import argparse
import csv
import datetime as dt
import re
import zipfile
import os
from pathlib import Path

import pandas as pd
import requests

CBC_PAGE = "https://www.cbc.gov.tw/tw/cp-528-1079-B4682-1.html"


def _parse_roc_month(text: str) -> str:
    """Return YYYY-MM from ROC year/month text like 114.12 or 114/12."""
    match = re.search(r"(\d{3})\s*[./-]\s*(\d{1,2})", text)
    if not match:
        return ""
    roc_year = int(match.group(1))
    if roc_year < 100 or roc_year > 200:
        return ""
    month = int(match.group(2))
    year = roc_year + 1911
    return f"{year:04d}-{month:02d}"


def _extract_download_url(page_html: str, prefer_ext: str) -> str:
    candidates = []
    for match in re.finditer(r'<a[^>]+href="([^"]+)"[^>]*>([^<]+)</a>', page_html):
        link = match.group(1)
        text = match.group(2).strip().lower()
        if "dl-" in link and link.lower().endswith(".html"):
            candidates.append((link, text))

    # Prefer anchors whose visible text matches the desired format (ODS/XLS/XLSX)
    for link, text in candidates:
        if prefer_ext in text:
            return _normalize_url(link)

    if candidates:
        return _normalize_url(candidates[0][0])
    raise RuntimeError("No download link found on CBC page.")


def _normalize_url(url: str) -> str:
    if url.startswith("http"):
        return url
    return f"https://www.cbc.gov.tw{url}"


def _resolve_file_url(download_page_url: str, verify: bool) -> str:
    resp = requests.get(download_page_url, timeout=30, verify=verify)
    resp.raise_for_status()

    # If CBC already redirects to the file, honor it.
    content_type = resp.headers.get("Content-Type", "")
    if "application" in content_type or download_page_url.lower().endswith((".ods", ".xls", ".xlsx")):
        return resp.url

    # Otherwise parse HTML for the actual file link.
    links = re.findall(r'href="([^"]+)"', resp.text)
    for link in links:
        if link.lower().endswith((".ods", ".xls", ".xlsx")):
            return _normalize_url(link)
    raise RuntimeError("Could not resolve file URL from CBC download page.")


def _sniff_extension(path: Path) -> str:
    with open(path, "rb") as f:
        header = f.read(4)
    if header.startswith(b"PK\x03\x04"):
        try:
            with zipfile.ZipFile(path, "r") as zf:
                if "mimetype" in zf.namelist():
                    mimetype = zf.read("mimetype").decode("utf-8", errors="ignore")
                    if "opendocument.spreadsheet" in mimetype:
                        return ".ods"
                if "xl/workbook.xml" in zf.namelist():
                    return ".xlsx"
        except zipfile.BadZipFile:
            pass
        return ".ods"
    if header == b"\xD0\xCF\x11\xE0":
        return ".xls"
    return path.suffix.lower() or ".bin"


def download_latest(out_dir: Path, prefer_ext: str = "ods", verify: bool = True) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    page_resp = requests.get(CBC_PAGE, timeout=30, verify=verify)
    page_resp.raise_for_status()
    page_html = page_resp.text

    ym = _parse_roc_month(page_html)
    if not ym:
        # Fallback: use current month when ROC month not found
        now = dt.date.today()
        ym = f"{now.year:04d}-{now.month:02d}"

    download_page_url = _extract_download_url(page_html, prefer_ext)
    file_url = _resolve_file_url(download_page_url, verify=verify)

    ext = Path(file_url).suffix.lower() or f".{prefer_ext}"
    raw_path = out_dir / "raw" / f"5newloan_{ym}{ext}"
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    with requests.get(file_url, stream=True, timeout=60, verify=verify) as r:
        r.raise_for_status()
        with open(raw_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 256):
                if chunk:
                    f.write(chunk)

    detected_ext = _sniff_extension(raw_path)
    if detected_ext and detected_ext != raw_path.suffix.lower():
        new_raw_path = raw_path.with_suffix(detected_ext)
        raw_path = raw_path.replace(new_raw_path)

    return raw_path


def _find_header_row(df: pd.DataFrame) -> int:
    for i, row in df.iterrows():
        for cell in row.tolist():
            if isinstance(cell, str) and ("年月" in cell or "年/月" in cell or "年 月" in cell):
                return i
    raise RuntimeError("Could not find header row containing 年月.")


def _build_columns(df: pd.DataFrame, header_row: int) -> tuple[list[str], int]:
    header = df.iloc[header_row].tolist()
    next_row = df.iloc[header_row + 1].tolist() if header_row + 1 < len(df) else []

    has_metric_row = any(isinstance(c, str) and ("金額" in c or "利率" in c) for c in next_row)
    if not has_metric_row:
        return [str(c).strip() if c is not None else "" for c in header], header_row + 1

    # Forward fill category names
    categories = []
    last = ""
    for c in header:
        if isinstance(c, str) and c.strip():
            last = c.strip()
        categories.append(last)

    columns = []
    for cat, metric in zip(categories, next_row):
        cat = cat.strip() if isinstance(cat, str) else ""
        metric = metric.strip() if isinstance(metric, str) else ""
        if cat and metric and cat != metric:
            columns.append(f"{cat}_{metric}")
        else:
            columns.append(cat or metric or "")

    return columns, header_row + 2


def _make_unique(columns: list[str]) -> list[str]:
    seen = {}
    unique = []
    for col in columns:
        name = str(col).strip() if col is not None else ""
        if not name:
            name = "欄位"
        if name in seen:
            seen[name] += 1
            unique.append(f"{name}_{seen[name]}")
        else:
            seen[name] = 0
            unique.append(name)
    return unique


def load_5newloan(raw_path: Path) -> pd.DataFrame:
    # Try ODS first, then Excel
    try:
        df = pd.read_excel(raw_path, header=None, engine="odf")
    except Exception:
        df = pd.read_excel(raw_path, header=None)

    df = df.dropna(how="all").dropna(axis=1, how="all")
    header_row = _find_header_row(df)
    columns, data_start = _build_columns(df, header_row)

    data = df.iloc[data_start:].copy()
    data.columns = _make_unique(columns)

    # Identify date column
    date_col = None
    for col in data.columns:
        if isinstance(col, str) and ("年月" in col or "年/月" in col or col.strip() == "年月"):
            date_col = col
            break
    if not date_col:
        date_col = data.columns[0]

    data = data.rename(columns={date_col: "年月"})
    data = data.dropna(subset=["年月"])

    # Normalize month to YYYY-MM
    def parse_month(value) -> str:
        if isinstance(value, (dt.date, dt.datetime)):
            return f"{value.year:04d}-{value.month:02d}"
        text = str(value)
        match = re.search(r"(\d{3})\s*[./-]\s*(\d{1,2})", text)
        if match:
            roc_year = int(match.group(1))
            month = int(match.group(2))
            year = roc_year + 1911
            return f"{year:04d}-{month:02d}"
        match = re.search(r"(\d{4})\s*[./-]\s*(\d{1,2})", text)
        if match:
            return f"{int(match.group(1)):04d}-{int(match.group(2)):02d}"
        return ""

    data["年月"] = data["年月"].apply(parse_month)
    data = data[data["年月"].str.match(r"\d{4}-\d{2}")]

    # Coerce numeric columns
    for col in data.columns:
        if col == "年月":
            continue
        data[col] = pd.to_numeric(data[col], errors="coerce")

    return data


def _to_long(data: pd.DataFrame) -> pd.DataFrame:
    records = []
    for col in data.columns:
        if col == "年月":
            continue
        metric = ""
        category = col
        if isinstance(col, str) and "_" in col:
            category, metric = col.rsplit("_", 1)
        records.append((col, category, metric))

    lookup = {col: (cat, metric) for col, cat, metric in records}
    long = data.melt(id_vars=["年月"], var_name="欄位", value_name="值")
    long["類別"] = long["欄位"].map(lambda c: lookup.get(c, (c, ""))[0])
    long["指標"] = long["欄位"].map(lambda c: lookup.get(c, (c, ""))[1])
    return long


def plot_series(long: pd.DataFrame, out_dir: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.sans-serif"] = [
        "Noto Sans CJK TC",
        "PingFang TC",
        "Heiti TC",
        "Microsoft JhengHei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False

    for metric, filename in [("金額", "5newloan_amount"), ("利率", "5newloan_rate")]:
        subset = long[long["指標"].str.contains(metric)]
        if subset.empty:
            continue

        plt.figure(figsize=(15, 9))

        if metric == "金額":
            # Stacked area for amounts
            pivot = subset.pivot_table(index="年月", columns="類別", values="值", aggfunc="sum").sort_index()
            categories = [c for c in pivot.columns if c != "合計"]
            values = [pivot[cat].values for cat in categories]
            plt.stackplot(pivot.index, values, labels=categories)
        else:
            # Line chart for rates
            for cat in sorted(subset["類別"].unique()):
                series = subset[subset["類別"] == cat].sort_values("年月")
                plt.plot(series["年月"], series["值"], label=cat)

        # Reduce x-axis label density (show every 6th month)
        months = sorted(subset["年月"].unique())
        if months:
            step = 6
            tick_positions = [m for i, m in enumerate(months) if i % step == 0]
            plt.xticks(tick_positions, rotation=45, ha="right")
        else:
            plt.xticks(rotation=45, ha="right")
        plt.title(f"五大銀行新承做放款{metric} (月)")
        plt.xlabel("月份")
        plt.ylabel(metric)
        plt.tight_layout()
        plt.legend(fontsize=8)
        plt.savefig(out_dir / f"{filename}.svg", format="svg")
        plt.close()


def write_csv(data: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(out_path, index=False, quoting=csv.QUOTE_MINIMAL)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and plot CBC 5newloan data.")
    parser.add_argument("--out-dir", default="data/5newloan", help="Output directory")
    parser.add_argument("--format", default="ods", choices=["ods", "xls", "xlsx"], help="Preferred source format")
    parser.add_argument("--plot", action="store_true", help="Generate SVG plots")
    parser.add_argument("--no-download", action="store_true", help="Skip download and use latest file in data/5newloan/raw")
    parser.add_argument("--insecure", action="store_true", help="Disable TLS certificate verification")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)

    if args.no_download:
        raw_dir = out_dir / "raw"
        raw_files = sorted(raw_dir.glob("5newloan_*.ods")) + sorted(raw_dir.glob("5newloan_*.xls")) + sorted(raw_dir.glob("5newloan_*.xlsx"))
        if not raw_files:
            raise RuntimeError("No raw files found. Remove --no-download or place a raw file in data/5newloan/raw.")
        raw_path = raw_files[-1]
    else:
        raw_path = download_latest(out_dir, prefer_ext=args.format, verify=not args.insecure)

    data = load_5newloan(raw_path)
    ym = data["年月"].max()
    if not isinstance(ym, str) or not re.match(r"\d{4}-\d{2}", ym):
        ym = "latest"

    if ym != "latest" and not re.search(rf"5newloan_{ym}", raw_path.name):
        new_raw_path = raw_path.with_name(f"5newloan_{ym}{raw_path.suffix}")
        raw_path = raw_path.replace(new_raw_path)

    write_csv(data, out_dir / f"5newloan_{ym}.csv")
    write_csv(data, out_dir / "5newloan_latest.csv")

    if args.plot:
        long = _to_long(data)
        plot_series(long, out_dir / "plots")


if __name__ == "__main__":
    main()
