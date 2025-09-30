import cv2 as cv                         
import numpy as np                        
import pandas as pd                       
import re                                
import difflib                            
import easyocr
from datetime import datetime

from pdf2image import convert_from_path  
from difflib import SequenceMatcher  
from typing import Tuple, Union, List

class MandiriExtractor: 
    def __init__(self, reader):
        self._TIME_SEP    = r"[:\.\uFF1A]"
        self.TIME_RE      = re.compile(
            rf"(?i)(?<!\d)(?:[01]?\d|2[0-3]){self._TIME_SEP}[0-5]\d(?:{self._TIME_SEP}[0-5]\d)?\s*(?:am|pm)?(?!\d)"
        )
        self.DATE_FORMAT  = "dd/MM/yyyy"
        self.reader       = reader
        self.DATE_RE      = re.compile(self.to_regex(self.DATE_FORMAT))
        self.header_input = {
            "Posting Date": (50, 50),
            "Remark": (0, 150),
            "Reference No": (0, 0),
            "Debit": (0, 155),
            "Kredit": (0, 155),
            "Balance": (0, 155)
        }
        
        self.keterangan  = "Remark"
        self.kolom_kode  = "DB/CR"
        self.target_kode = "Amount"
        self.debit_code  = "D"
        self.kredit_code = "K"
        self.header_per_page = False

    # ------------------------
    # Preprocessing
    # ------------------------
    def noise_removal(self, image):
        if len(image.shape) == 3:
            if image.shape[2] == 4:
                gray = cv.cvtColor(image, cv.COLOR_RGBA2GRAY)
            else:
                gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        else:
            gray = image

        clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)

        blur = cv.GaussianBlur(gray, (3,3), 0)
        return blur

    # ------------------------
    # Utility: fuzzy match
    # ------------------------
    def fuzzy_match(self, a, b, cutoff=0.8):
        a_clean = re.sub(r'[^A-Za-z0-9]', '', a)
        b_clean = re.sub(r'[^A-Za-z0-9]', '', b)
        return difflib.SequenceMatcher(None, a_clean.upper(), b_clean.upper()).ratio() >= cutoff

    # ------------------------
    # Cari header di PDF
    # ------------------------
    def find_header_pdf(self, img, y_tolerance=25, default_margin_x=75, min_ratio=0.6):
        HEADER_WORDS = set(self.header_input.keys())
        results = self.reader.readtext(img, detail=1, paragraph=False)

        lines = []
        for (bbox, word, conf) in results:
            if not word.strip():
                continue
            y_min = int(float(bbox[0][1]))
            matched_line = None
            for line in lines:
                if abs(int(line["y"]) - y_min) <= y_tolerance:
                    matched_line = line
                    break
            if matched_line:
                matched_line['words'].append((word, bbox))
            else:
                lines.append({"y": y_min, "words": [(word, bbox)]})

        headers_found = {}
        for line in lines:
            detected_headers = {}
            for word, bbox in line["words"]:
                for header in HEADER_WORDS:
                    if self.fuzzy_match(word, header):
                        margin_left, margin_right = self.header_input.get(header, (default_margin_x, default_margin_x))
                        (x_min, y_min) = bbox[0]
                        (x_max, y_max) = bbox[2]
                        detected_headers[header] = {
                            "x_min": int(x_min) - margin_left,
                            "x_max": int(x_max) + margin_right,
                            "y_min": int(y_min),
                            "y_max": int(y_max)
                        }
            match_ratio = len(detected_headers) / len(HEADER_WORDS)
            if match_ratio >= min_ratio:
                headers_found = detected_headers
                break

        headers_sorted = dict(sorted(headers_found.items(), key=lambda item: item[1]['x_min']))
        return headers_sorted, results

    # ------------------------
    # Date regex helper
    # ------------------------
    def to_regex(self, fmt: str) -> str:
        mapping = {
            "dd": r"\d{1,2}",
            "MM": r"\d{1,2}",
            "MMM": r"[A-Za-z]{3}",
            "MMMM": r"[A-Za-z]{3,9}",
            "yyyy": r"\d{4}",
            "yy": r"\d{2}",
        }

        regex = re.escape(fmt)
        for key in sorted(mapping.keys(), key=len, reverse=True):
            regex = regex.replace(re.escape(key), mapping[key])
        return rf"(?<!\d){regex}(?!\d)"
    
    def find_date_coords(self, hasil_ocr, header_bottom_y=0, y_tol=0, y_shift = 25):
        date_coords = []
        seen = []

        for (bbox, word, conf) in hasil_ocr:
            if self.DATE_RE.match(word.strip()):
                x1, y1 = bbox[0]
                x2, y2 = bbox[2]

                if y1 > header_bottom_y:
                    duplicate = False
                    for sy1, sy2 in seen:
                        if abs(sy1 - int(y1)) <= y_tol and abs(sy2 - int(y2)) <= y_tol:
                            duplicate = True
                            break

                    if not duplicate:
                        clean_text = word.strip()
                        if not re.search(r"\d{1,2}:\d{2}(:\d{2})?", clean_text):
                            seen.append((int(y1), int(y2)))
                            date_coords.append({
                                "text": clean_text,
                                "x_min": int(x1),
                                "x_max": int(x2),
                                "y_min": int(y1) - y_shift,
                                "y_max": int(y2) + y_shift
                            })
        return date_coords

    # ------------------------
    # Normalisasi & Date cleaning
    # ------------------------
    def normalisasi_token(self, tok: str) -> str:
        s = tok
        s = s.replace('：', ':').replace('．', '.')
        s = s.replace('\u2009', ' ')
        if re.search(r"[:\.]", s) or re.search(r"\d+[OoIlI]\d", s):
            s = re.sub(r'[Oo]', '0', s)
            s = re.sub(r'[IlI]', '1', s)
        return s

    def clean_date_text(self, raw: str) -> str:
        if not raw:
            return ""
        tokens = re.split(r'\s+', raw)
        tokens_norm = [self.normalisasi_token(t) for t in tokens]
        joined_norm = " ".join(tokens_norm)

        if self.TIME_RE.search(joined_norm) and self.DATE_RE.search(joined_norm):
            without_time = self.TIME_RE.sub('', joined_norm).strip(" -,:;·•")
            m = self.DATE_RE.search(without_time)
            if m:
                return m.group()

        m = self.DATE_RE.search(raw)
        if m:
            return m.group()
        return raw.strip()

    # ------------------------
    # Extracting Table
    # ------------------------
    def extracting_table(self, hasil_ocr, coords, kolom_coords, page, header_y_min=1000):
        YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")
        detected_year = None

        for c in coords:
            match = YEAR_RE.search(c.get("text", ""))
            if match:
                detected_year = int(match.group())
                break

        if not detected_year:
            candidates = []
            for (bbox, word, conf) in hasil_ocr:
                match = YEAR_RE.search(word)
                if match:
                    year_val = int(match.group())
                    y = int(sum([p[1] for p in bbox]) / 4)
                    if y < header_y_min:
                        candidates.append((y, year_val))
            if candidates:
                detected_year = max(candidates, key=lambda c: c[0])[1]

        if not detected_year:
            detected_year = datetime.now().year

        results = []
        rows_coords = sorted(coords, key=lambda x: x["y_min"])
        rows_coords = [{"y_min": r["y_min"], "y_max": r["y_max"]} for r in rows_coords]

        for row_idx, row in enumerate(rows_coords, start=1):
            row_data = {}
            for col_name, col in kolom_coords.items():
                y_min, y_max = row["y_min"], row["y_max"]
                x_min, x_max = col["x_min"], col["x_max"]

                cell_texts = []
                for (bbox, text, conf) in hasil_ocr:
                    cx = int(sum([p[0] for p in bbox]) / 4)
                    cy = int(sum([p[1] for p in bbox]) / 4)
                    if x_min <= cx <= x_max and y_min <= cy <= y_max:
                        cell_texts.append(text.strip())

                text = " ".join(cell_texts).strip()
                cleaned = self.clean_date_text(text)

                if re.match(r"^\d{1,2}[/-]\d{1,2}$", cleaned):
                    cleaned = f"{cleaned}/{detected_year}"

                row_data[col_name] = cleaned
            results.append(row_data)
        df = pd.DataFrame(results)
        print(f"✅ Berhasil mengekstrak {len(df)} baris transaksi pada halaman ke - {page}.")
        return df

    # ------------------------
    # Formating to Float (Decimal) Format
    # ------------------------
    def to_number(self, x):
        if pd.isna(x) or str(x).strip() == "":
            return 0.0

        s = str(x).strip()
        s = re.sub(r"[^\d\-,.]", "", s)
        last_dot = s.rfind(".")
        last_comma = s.rfind(",")

        if last_comma > last_dot:
            s = s.replace(".", "").replace(",", ".")
        elif last_dot > last_comma:
            s = s.replace(",", "")
        else:
            s = s.replace(",", ".")

        try:
            return float(s)
        except ValueError:
            return 0.0
    
    # ------------------------
    # Find Debit & Kredit
    # ------------------------
    def debit_and_kredit(self, df: pd.DataFrame, kolom_keterangan: str = None) -> pd.DataFrame:
        df = df.copy()
        df.columns       = [c.lower() for c in df.columns]
        self.kolom_kode  = self.kolom_kode.lower()
        self.target_kode = self.target_kode.lower()
        if kolom_keterangan:
            kolom_keterangan = kolom_keterangan.lower()

        if self.target_kode in df.columns:
            df[self.target_kode] = df[self.target_kode].apply(self.to_number)

        if "balance" in df.columns:
            df["saldo"] = df["balance"].apply(self.to_number)
            df = df.drop(columns=["balance"])
        elif "saldo" in df.columns:
            df["saldo"] = df["saldo"].apply(self.to_number)

        debit_aliases  = ["debit", "debet"]
        kredit_aliases = ["kredit", "credit"]

        self.debit_col  = next((c for c in df.columns if c in debit_aliases), None)
        self.kredit_col = next((c for c in df.columns if c in kredit_aliases), None)

        if self.debit_col and self.kredit_col:
            df["debit"]  = df[self.debit_col].apply(self.to_number)
            df["kredit"] = df[self.kredit_col].apply(self.to_number)

            for c in [self.debit_col, self.kredit_col]:
                if c not in ["debit", "kredit"]:
                    df = df.drop(columns=[c])
        else:
            if self.kolom_kode not in df.columns or self.target_kode not in df.columns:
                raise ValueError(
                    "Kolom debit/kredit belum ada. Harap berikan kolom_kode dan target_kode untuk diproses."
                )

            df["debit"] = 0.0
            df["kredit"] = 0.0
            for i, row in df.iterrows():
                kode = str(row.get(self.kolom_kode, "")).strip().upper()
                amount = row.get(self.target_kode, 0.0)
                if kode == self.debit_code.upper():
                    df.at[i, "debit"] = amount
                elif kode == self.kredit_code.upper():
                    df.at[i, "kredit"] = amount

            df = df.drop(columns=[self.kolom_kode, self.target_kode], errors="ignore")

        if kolom_keterangan and kolom_keterangan in df.columns:
            df = df.rename(columns={kolom_keterangan: "keterangan"})
        cols = [c for c in df.columns if c not in ["debit", "kredit", "saldo"]]
        if "saldo" in df.columns:
            df = df[cols + ["debit", "kredit", "saldo"]]
        else:
            df = df[cols + ["debit", "kredit"]]
        return df
    
    # --------------------------------
    # Add Saldo Awal (Opening Balance)
    # --------------------------------
    def add_saldo_awal(self, df: pd.DataFrame) -> pd.DataFrame:
        df_copy    = df.copy()
        saldo_awal = 0.0
        first_row  = df_copy.loc[0]

        debit  = first_row.get("debit", 0)
        kredit = first_row.get("kredit", 0)
        saldo  = first_row.get("saldo", 0)

        if debit != 0:
            saldo_awal = saldo + debit
        elif kredit != 0:
            saldo_awal = saldo - kredit
        else:
            saldo_awal = saldo

        saldo_awal_row = {
            col: 0.0 if col in ["debit", "kredit", "saldo"] else "" 
            for col in df.columns
        }
        
        saldo_awal_row["keterangan"] = "SALDO AWAL"
        saldo_awal_row["saldo"] = saldo_awal
        df_copy = pd.concat(
            [pd.DataFrame([saldo_awal_row]), df_copy], ignore_index=True
        )
        return df_copy

    # ---------------------------------
    # Add Saldo Akhir (Closing Balance)
    # ---------------------------------
    def add_saldo_akhir(self, df: pd.DataFrame) -> pd.DataFrame:
        df_copy = df.copy()

        saldo_awal = df_copy.loc[0, "saldo"]
        debit_temp, kredit_temp = 0.0, 0.0
        for i in range(1, len(df_copy)):
            debit_temp  += df_copy.loc[i, "debit"]
            kredit_temp += df_copy.loc[i, "kredit"]

        saldo_akhir_ = saldo_awal - debit_temp + kredit_temp
        saldo_akhir_row = {
            col: 0.0 if col in ["debit", "kredit", "saldo"] else "" 
            for col in df.columns
        }

        saldo_akhir_row["keterangan"] = "SALDO AKHIR"
        saldo_akhir_row["saldo"] = saldo_akhir_
        df_copy = pd.concat([df_copy, pd.DataFrame([saldo_akhir_row])], ignore_index=True)
        return df_copy

    # ---------------------------------
    # Hapus Duplikasi Baris (Jika Ada)
    # ---------------------------------
    def drop_next_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        df_copy = df.copy()        
        main_cols = ["keterangan", "debit", "kredit", "saldo"]
        mask = df_copy[main_cols].shift().eq(df_copy[main_cols]).all(axis=1)
        df_cleaned = df_copy[~mask].reset_index(drop=True)
        return df_cleaned
    
    # --------------------------------------------
    # Menghapus Baris jika hanya satu kolom terisi
    # --------------------------------------------
    def drop_incomplete(self, df: pd.DataFrame) -> pd.DataFrame:
        df_copy = df.copy()
        main_cols = ["tanggal", "keterangan", "debit", "kredit", "saldo"]
        numeric_cols = ["debit", "kredit", "saldo"]
        for c in numeric_cols:
            df_copy[c] = (
                pd.to_numeric(df_copy[c].astype(str).str.replace(",", ""), errors="coerce")
                .fillna(0)
            )

        mask_non_empty = df_copy[main_cols].apply(
            lambda x: x.notna() & (x.astype(str).str.strip() != ""))
        for c in numeric_cols:
            mask_non_empty[c] = df_copy[c].notna() & (df_copy[c] != 0)

        count_non_empty = mask_non_empty.sum(axis=1)
        cond_incomplete = count_non_empty <= 1
        cond_all_zero = (df_copy[numeric_cols] == 0).all(axis=1)
        df_cleaned = df_copy[~(cond_incomplete | cond_all_zero)].reset_index(drop=True)
        return df_cleaned
    
    # ---------------------------------
    # Mengkapitalisasi Kata
    # ---------------------------------
    def capitalize_and_date(self, df: pd.DataFrame, output_excel=None) -> pd.DataFrame:
        df = df.copy()
        df.columns = [col.title() for col in df.columns]
        if output_excel:
            df.to_excel(output_excel, index=False, engine="openpyxl")
            print(f"📊 Data berhasil disimpan ke Excel: {output_excel}")
        return df

    # ---------------------------------
    # Converting Many Functions
    # ---------------------------------
    def convert(self, pdf_path: str, pages: Union[str, List[int]] = "all", output_excel=None):
        images = convert_from_path(pdf_path, dpi=200)
        if pages != "all":
            images = [images[i-1] for i in pages if 0 < i <= len(images)]
            
        all_pages_df = []
        header_cache = None
        for page_idx, image in enumerate(images):
            img_array = np.array(image)
            img_array = self.noise_removal(img_array)
            
            if self.header_per_page or header_cache is None:
                kolom_coords, hasil_ocr = self.find_header_pdf(img_array)
                if kolom_coords:
                    header_cache = kolom_coords
            else:
                kolom_coords = header_cache
                hasil_ocr    = self.reader.readtext(img_array, detail=1, paragraph=False)
                    
            if not hasil_ocr:
                print(f"⚠️ Halaman {page_idx + 1} kosong, dilewati...") # ------ STAGE 1
                continue
            
            if self.header_per_page or page_idx == 0:
                minimum_header = 500
            else:
                minimum_header = 0
            coords    = self.find_date_coords(hasil_ocr, header_bottom_y=minimum_header)
            if not coords:
                print(f"⚠️ Tidak ada halaman yang berhasil diekstrak.") # ------ STAGE 2
                continue
            df        = self.extracting_table(hasil_ocr, coords, kolom_coords, page = page_idx + 1, header_y_min=minimum_header)
            if df.empty:
                print(f"⚠️ Dataframe kosong di halaman {page_idx + 1}, dilewati...") # ----- STAGE 3
                continue
                
            df_knd    = self.debit_and_kredit(df, kolom_keterangan=self.keterangan)
            all_pages_df.append(df_knd)
            
        if not all_pages_df:
            print("❌ Tidak ada halaman yang berhasil diekstrak.")
            return pd.DataFrame()
        
        df_combined = pd.concat(all_pages_df, ignore_index=True)
        df_dp       = self.drop_next_duplicates(df_combined)
        df_open     = self.add_saldo_awal(df_dp)
        df_close    = self.add_saldo_akhir(df_open)
        df_close    = self.drop_incomplete(df_close)
        df_final    = self.capitalize_and_date(df_close, output_excel)
        return df_final
