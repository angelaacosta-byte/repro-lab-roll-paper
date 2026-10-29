"""Build the anonymised modelling table from the private cleaned Veritrade file.

Run (only the author can, because the source file is not public):
    python src/make_dataset.py path/to/bobinas_limpio.xlsx

Output: data/bobinas_model_data.csv  (no firm names, no description text; importer replaced by a code).
The source file is checked against its registered MD5 before anything is written.
"""
import hashlib, re, sys, unicodedata
import numpy as np
import pandas as pd

SOURCE_MD5 = "f2d7e5e54cc455749e50e57bb96dbfea"   # cleaned file, see the thesis Datasheet
OUT = "data/bobinas_model_data.csv"
DESC = ["Descripcion Comercial"] + [f"Descripcion{i}" for i in range(1, 6)]


def norm(s):
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().upper()


def gsm(t):
    m = re.search(r"(?:GSM|G/M2|GRAMAJE|SUB(?:STANCE)?|BASIS(?:\s+GSM)?)\s*[:=]?\s*(\d{2}(?:[.,]\d+)?)", t) or \
        re.search(r"(\d{2}(?:[.,]\d+)?)\s*(?:GSM|G/M2|GRS?\b|GRAMOS|G\b)", t)
    v = float(m.group(1).replace(",", ".")) if m else np.nan
    return v if 10 <= v <= 60 else np.nan


def build(path):
    md5 = hashlib.md5(open(path, "rb").read()).hexdigest()
    assert md5 == SOURCE_MD5, f"Source file is not the registered cleaned version (MD5 {md5})"
    df = pd.read_excel(path).sort_values(["Fecha", "DUA / DAM"]).reset_index(drop=True)
    txt = df[DESC].fillna("").astype(str).agg(" ".join, axis=1).map(norm)

    imp = df["Importador"].fillna("NO DISPONIBLE").astype(str).str.strip()
    exp = df["Embarcador / Exportador"].replace("NO DISPONIBLE", np.nan)
    top_c = df["Pais de Origen"].value_counts().index[:8]
    out = pd.DataFrame({
        "record_id": [f"R{i+1:04d}" for i in range(len(df))],
        "year": df.Fecha.dt.year, "month": df.Fecha.dt.month,
        "importer_code": imp.map({v: f"IMP_{i+1:03d}" for i, v in enumerate(pd.unique(imp))}),
        "hs": df["Partida Aduanera"].astype(str).str[:6],
        "origin": df["Pais de Origen"].where(df["Pais de Origen"].isin(top_c), "OTHER"),
        "customs_office": df["Aduana"].fillna("NA"),
        "transport": df["Via"].fillna("NA"),
        "channel": df["Canal"].fillna("NOT RECORDED"),
        "product": np.select([txt.str.contains(r"TOALLA|TOWEL"), txt.str.contains(r"HIGIENICO|TOILET|BATH"),
                              txt.str.contains(r"SERVILLETA|NAPKIN")], ["towel", "toilet", "napkin"], "other"),
        "gsm": txt.map(gsm),
        "net_kg": df["Kg Neto"].round(2),
        "fob_usd": df["U$ FOB Tot"].round(2),
        "supplier_known": exp.notna().astype(int),
        # label: proxy-indicated sustainable (recycled-fibre or certification wording)
        "sust": (txt.str.contains(r"\bFSC\b|\bPEFC\b|ISO\s*14001|ECO.?LABEL") |
                 txt.str.contains(r"RECICLAD|RECYCL")).astype(int),
    })
    out.to_csv(OUT, index=False, lineterminator="\n")
    print(f"{OUT}: {len(out)} rows, {out.sust.sum()} positive, MD5 {hashlib.md5(open(OUT,'rb').read()).hexdigest()}")


if __name__ == "__main__":
    build(sys.argv[1])
