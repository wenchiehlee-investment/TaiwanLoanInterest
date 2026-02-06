#!/usr/bin/env python3
import argparse
import csv
import datetime as dt
import os
import re
import zipfile
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests

CBC_PAGE = "https://www.cbc.gov.tw/tw/cp-529-1081-195A7-1.html"


def _parse_roc_quarter(text: str) -> str:
    """
    Return YYYYQn from ROC year/quarter text like 114年第3季.
    """
    match = re.search(r"(\d{3})\s*年\s*第?\s*(\d)\s*季", text)
    if not match:
        # looser fallback
        match = re.search(r"(\d{3}).*?(\d)\s*季", text)
    if not match:
        return ""
    roc_year = int(match.group(1))
    quarter = int(match.group(2))
    if roc_year < 100 or roc_year > 200 or quarter not in (1, 2, 3, 4):
        return ""
    year = roc_year + 1911
    return f"{year}Q{quarter}"


def _normalize_url(url: str) -> str:
    if url.startswith("http"):
        return url
    return f"https://www.cbc.gov.tw{url}"


def _extract_download_url(page_html: str, prefer_ext: str) -> str:
    """
    CBC stats pages typically link to an intermediate dl-*.html page per format.
    """
    candidates: list[tuple[str, str]] = []
    for match in re.finditer(r'<a[^>]+href="([^"]+)"[^>]*>([^<]+)</a>', page_html):
        link = match.group(1)
        text = match.group(2).strip().lower()
        if "dl-" in link and link.lower().endswith(".html"):
            candidates.append((link, text))

    # Prefer anchors whose visible text matches the desired format (ODS/XLS/PDF)
    prefer_ext = prefer_ext.lower()
    for link, text in candidates:
        if prefer_ext in text:
            return _normalize_url(link)

    if candidates:
        return _normalize_url(candidates[0][0])
    raise RuntimeError("No download link found on CBC page.")


def _resolve_file_url(download_page_url: str, verify: bool) -> str:
    resp = requests.get(download_page_url, timeout=30, verify=verify)
    resp.raise_for_status()

    # If CBC already redirects to the file, honor it.
    content_type = resp.headers.get("Content-Type", "")
    if "application" in content_type or download_page_url.lower().endswith((".ods", ".xls", ".xlsx")):
        return resp.url

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

    yq = _parse_roc_quarter(page_html)
    if not yq:
        now = dt.date.today()
        q = (now.month - 1) // 3 + 1
        yq = f"{now.year}Q{q}"

    download_page_url = _extract_download_url(page_html, prefer_ext)
    file_url = _resolve_file_url(download_page_url, verify=verify)

    ext = Path(file_url).suffix.lower() or f".{prefer_ext}"
    raw_path = out_dir / "raw" / f"averageir_{yq}{ext}"
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    with requests.get(file_url, stream=True, timeout=60, verify=verify) as r:
        r.raise_for_status()
        with open(raw_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 256):
                if chunk:
                    f.write(chunk)

    detected_ext = _sniff_extension(raw_path)
    if detected_ext and detected_ext != raw_path.suffix.lower():
        raw_path = raw_path.replace(raw_path.with_suffix(detected_ext))

    return raw_path


def _find_header_row(df: pd.DataFrame) -> int:
    for i, row in df.iterrows():
        values = row.tolist()
        has_year = any(isinstance(c, str) and "年" in c for c in values)
        has_quarter = any(isinstance(c, str) and "季" in c for c in values)
        if has_year and has_quarter:
            return i
    raise RuntimeError("Could not find header row containing 年 and 季.")


def _build_columns(df: pd.DataFrame, header_row: int) -> tuple[list[str], int]:
    """
    Expect 2-level headers:
    - row N: 年 / 季 / (institution names spanning two columns)
    - row N+1: (blank) / (blank) / 存款 / 放款 ...
    """
    header = df.iloc[header_row].tolist()
    next_row = df.iloc[header_row + 1].tolist() if header_row + 1 < len(df) else []

    has_metric_row = any(isinstance(c, str) and ("存" in c or "放" in c) for c in next_row)
    if not has_metric_row:
        return [str(c).strip() if c is not None else "" for c in header], header_row + 1

    categories: list[str] = []
    last = ""
    for c in header:
        if isinstance(c, str) and c.strip():
            last = c.strip()
        categories.append(last)

    columns: list[str] = []
    for cat, metric in zip(categories, next_row):
        cat = cat.strip() if isinstance(cat, str) else ""
        metric = metric.strip() if isinstance(metric, str) else ""

        # Year/quarter columns
        if metric == "" and cat in ("年", "季", "年季", "年 季"):
            columns.append(cat)
            continue

        if cat and metric:
            columns.append(f"{cat}_{metric}")
        else:
            columns.append(cat or metric or "")

    return columns, header_row + 2


def _make_unique(columns: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    unique: list[str] = []
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


def load_averageir(raw_path: Path) -> pd.DataFrame:
    try:
        df = pd.read_excel(raw_path, header=None, engine="odf")
    except Exception:
        df = pd.read_excel(raw_path, header=None)

    df = df.dropna(how="all").dropna(axis=1, how="all")
    header_row = _find_header_row(df)
    columns, data_start = _build_columns(df, header_row)

    data = df.iloc[data_start:].copy()
    data.columns = _make_unique(columns)

    # Find year / quarter columns
    year_col = None
    quarter_col = None
    for col in data.columns:
        if isinstance(col, str) and col.strip() == "年":
            year_col = col
        if isinstance(col, str) and col.strip() == "季":
            quarter_col = col
    if year_col is None:
        year_col = data.columns[0]
    if quarter_col is None:
        quarter_col = data.columns[1] if len(data.columns) > 1 else data.columns[0]

    data = data.rename(columns={year_col: "年", quarter_col: "季"})
    data = data.dropna(subset=["年", "季"])

    def parse_roc_year(value) -> int | None:
        if pd.isna(value):
            return None
        text = str(value).strip()
        m = re.search(r"(\d{2,3})", text)
        if not m:
            return None
        y = int(m.group(1))
        if y < 80:  # probably already Gregorian
            return y
        return y + 1911

    def parse_quarter(value) -> int | None:
        if pd.isna(value):
            return None
        text = str(value).strip()
        m = re.search(r"(\d)", text)
        if not m:
            return None
        q = int(m.group(1))
        return q if q in (1, 2, 3, 4) else None

    data["_year"] = data["年"].apply(parse_roc_year)
    data["_q"] = data["季"].apply(parse_quarter)
    data = data.dropna(subset=["_year", "_q"])
    data["_year"] = data["_year"].astype(int)
    data["_q"] = data["_q"].astype(int)
    data["季別"] = data.apply(lambda r: f"{int(r['_year'])}Q{int(r['_q'])}", axis=1)

    # Drop original 年/季 to reduce confusion, keep 季別 only.
    data = data.drop(columns=["年", "季", "_year", "_q"])

    # Coerce numeric columns
    for col in data.columns:
        if col == "季別":
            continue
        data[col] = pd.to_numeric(data[col], errors="coerce")

    # Compute spreads per institution where we can find deposit/loan pairs.
    spread_cols = {}
    for col in data.columns:
        if not isinstance(col, str) or col == "季別":
            continue
        if col.endswith("_存款"):
            base = col[: -len("_存款")]
            loan_col = f"{base}_放款"
            if loan_col in data.columns:
                spread_cols[f"{base}_利差"] = data[loan_col] - data[col]
    for name, series in spread_cols.items():
        data[name] = series

    # Keep 季別 first.
    ordered = ["季別"] + [c for c in data.columns if c != "季別"]
    return data[ordered].sort_values("季別").reset_index(drop=True)


def _to_long(data: pd.DataFrame) -> pd.DataFrame:
    records: list[tuple[str, str, str]] = []
    for col in data.columns:
        if col == "季別":
            continue
        metric = ""
        inst = col
        if isinstance(col, str) and "_" in col:
            inst, metric = col.rsplit("_", 1)
        records.append((col, inst, metric))

    lookup = {col: (inst, metric) for col, inst, metric in records}
    long = data.melt(id_vars=["季別"], var_name="欄位", value_name="值")
    long["機構"] = long["欄位"].map(lambda c: lookup.get(c, (c, ""))[0])
    long["指標"] = long["欄位"].map(lambda c: lookup.get(c, (c, ""))[1])
    return long


def plot_series(long: pd.DataFrame, out_dir: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)

    font_bump = 4
    plt.rcParams["font.sans-serif"] = [
        "Noto Sans CJK TC",
        "PingFang TC",
        "Heiti TC",
        "Microsoft JhengHei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.size"] = 14 + font_bump
    plt.rcParams["axes.titlesize"] = 16 + font_bump
    plt.rcParams["axes.labelsize"] = 14 + font_bump
    plt.rcParams["xtick.labelsize"] = 12 + font_bump
    plt.rcParams["ytick.labelsize"] = 12 + font_bump
    plt.rcParams["legend.fontsize"] = 12 + font_bump

    for metric, filename in [("存款", "averageir_deposit"), ("放款", "averageir_loan"), ("利差", "averageir_spread")]:
        subset = long[long["指標"] == metric]
        if subset.empty:
            continue

        plt.figure(figsize=(15, 9))
        for inst in sorted(subset["機構"].unique()):
            series = subset[subset["機構"] == inst].sort_values("季別")
            plt.plot(series["季別"], series["值"], label=inst, linewidth=2.0)

        ticks = sorted(subset["季別"].unique())
        if ticks:
            step = 4  # show yearly ticks for quarterly data
            tick_positions = [t for i, t in enumerate(ticks) if i % step == 0]
            plt.xticks(tick_positions, rotation=45, ha="right")
        else:
            plt.xticks(rotation=45, ha="right")

        plt.title(f"存放款加權平均利率（{metric}，季）")
        plt.xlabel("季別")
        plt.ylabel("年息百分比率")
        plt.tight_layout()
        plt.legend(fontsize=10 + font_bump, ncol=2)
        plt.savefig(out_dir / f"{filename}.svg", format="svg")
        plt.close()


def write_csv(data: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(out_path, index=False, quoting=csv.QUOTE_MINIMAL)


def _pick_value(data: pd.DataFrame, yq: str, col_contains: list[str]) -> float | None:
    if data.empty or "季別" not in data.columns:
        return None
    row = data.loc[data["季別"] == yq]
    if row.empty:
        return None
    for col in data.columns:
        if not isinstance(col, str):
            continue
        if all(term in col for term in col_contains):
            val = row.iloc[-1][col]
            try:
                return float(val)
            except Exception:
                return None
    return None


def update_readme_preview(readme_path: Path, timestamp: str, data: pd.DataFrame) -> None:
    """
    Insert/update an AVERAGEIR preview section in README.md.
    """
    if not readme_path.exists():
        return

    yq = None
    if not data.empty and "季別" in data.columns:
        last = data["季別"].max()
        if isinstance(last, str) and re.match(r"\d{4}Q[1-4]", last):
            yq = last

    # Latest headline values (focus on 本國銀行, but data includes all series).
    parts: list[str] = []
    if yq:
        dep = _pick_value(data, yq, ["本國銀行", "存款"])
        loan = _pick_value(data, yq, ["本國銀行", "放款"])
        spread = _pick_value(data, yq, ["本國銀行", "利差"])
        if dep is not None:
            parts.append(f"**本國銀行存款 {dep:.2f}%**")
        if loan is not None:
            parts.append(f"**本國銀行放款 {loan:.2f}%**")
        if spread is not None:
            parts.append(f"**利差 {spread:.2f}%**")

    headline = ""
    if yq and parts:
        headline = f"最新季度（{yq}）：{'、'.join(parts)}"

    marker = "### 圖表預覽（AVERAGEIR）"

    # Only update README when the plots exist; keeps README stable unless SVGs change.
    plot_dir = Path("data/averageir/plots")
    plot_files = [
        plot_dir / "averageir_deposit.svg",
        plot_dir / "averageir_loan.svg",
        plot_dir / "averageir_spread.svg",
    ]
    if not all(p.exists() and p.stat().st_size > 0 for p in plot_files):
        return

    section: list[str] = [
        marker,
        "",
        f"Update time: {timestamp}",
        "",
        "**存放款加權平均利率（存款，折線圖）**  ",
    ]
    if headline:
        section.append(headline + "  ")
    section += [
        "",
        "![存放款加權平均利率（存款）](data/averageir/plots/averageir_deposit.svg)",
        "",
        "**存放款加權平均利率（放款，折線圖）**  ",
        "",
        "![存放款加權平均利率（放款）](data/averageir/plots/averageir_loan.svg)",
        "",
        "**存放款加權平均利率（利差，折線圖）**  ",
        "",
        "![存放款加權平均利率（利差）](data/averageir/plots/averageir_spread.svg)",
        "",
        "---",
        "",
    ]

    content = readme_path.read_text(encoding="utf-8")
    lines = content.splitlines()
    if marker not in lines:
        # Insert right after the 5newloan preview separator if possible.
        insert_after = "### 圖表預覽（5newloan）"
        if insert_after in lines:
            idx = lines.index(insert_after)
            # Find the first separator after that section.
            sep_idx = None
            for j in range(idx + 1, len(lines)):
                if lines[j].strip() == "---":
                    sep_idx = j
                    break
            if sep_idx is not None:
                new_lines = lines[: sep_idx + 1] + [""] + section + lines[sep_idx + 1 :]
            else:
                new_lines = lines + [""] + section
        else:
            new_lines = lines + [""] + section
    else:
        start = lines.index(marker)
        end = None
        for j in range(start + 1, len(lines)):
            if lines[j].strip() == "---":
                end = j
                break
        if end is None:
            new_lines = lines[:start] + section
        else:
            # Replace marker...separator (inclusive) with our canonical section.
            new_lines = lines[:start] + section + lines[end + 1 :]

    new_content = "\n".join(new_lines) + ("\n" if content.endswith("\n") else "")
    readme_path.write_text(new_content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and plot CBC AVERAGEIR data.")
    parser.add_argument("--out-dir", default="data/averageir", help="Output directory")
    parser.add_argument("--format", default="ods", choices=["ods", "xls", "xlsx"], help="Preferred source format")
    parser.add_argument("--plot", action="store_true", help="Generate SVG plots")
    parser.add_argument("--no-download", action="store_true", help="Skip download and use latest file in data/averageir/raw")
    parser.add_argument("--insecure", action="store_true", help="Disable TLS certificate verification")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    raw_path: Path
    if args.no_download:
        raw_dir = out_dir / "raw"
        raw_files = (
            sorted(raw_dir.glob("averageir_*.ods"))
            + sorted(raw_dir.glob("averageir_*.xls"))
            + sorted(raw_dir.glob("averageir_*.xlsx"))
        )
        if not raw_files:
            raise RuntimeError("No raw files found. Remove --no-download or place a raw file in data/averageir/raw.")
        raw_path = raw_files[-1]
    else:
        raw_path = download_latest(out_dir, prefer_ext=args.format, verify=not args.insecure)

    data = load_averageir(raw_path)
    yq = data["季別"].max() if (not data.empty and "季別" in data.columns) else "latest"
    if not isinstance(yq, str) or not re.match(r"\d{4}Q[1-4]", yq):
        yq = "latest"

    if yq != "latest" and yq not in raw_path.name:
        raw_path = raw_path.replace(raw_path.with_name(f"averageir_{yq}{raw_path.suffix}"))

    write_csv(data, out_dir / f"averageir_{yq}.csv")
    write_csv(data, out_dir / "averageir_latest.csv")

    if args.plot:
        long = _to_long(data)
        plot_series(long, out_dir / "plots")

    # Only bump the README preview when we actually re-generate the SVGs.
    if args.plot:
        timestamp = dt.datetime.now(ZoneInfo("Asia/Taipei")).strftime("%Y-%m-%d %H:%M:%S CST")
        update_readme_preview(Path("README.md"), timestamp, data)


if __name__ == "__main__":
    main()
