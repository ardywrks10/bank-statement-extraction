import cv2 as cv                         
import numpy as np                        
import pandas as pd                       
import re                                
import difflib                            
import easyocr
from datetime import datetime

from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from pdf2image import convert_from_path  
from difflib import SequenceMatcher  
from typing import Tuple, Union, List, Optional

class BCAExtractor: 
    def __init__(self, reader):
        self._TIME_SEP    = r"[:\.\uFF1A]"
        self.TIME_RE      = re.compile(
            rf"(?i)(?<!\d)(?:[01]?\d|2[0-3]){self._TIME_SEP}[0-5]\d(?:{self._TIME_SEP}[0-5]\d)?\s*(?:am|pm)?(?!\d)"
        )
        self.DATE_FORMAT = "dd/MM"
        self.reader      = reader
        self.DATE_RE     = re.compile(self.to_regex(self.DATE_FORMAT))
        self.HEADERS     = ["TANGGAL", "KETERANGAN", "CBG", "MUTASI", "SALDO"]
        self.keterangan  = "KETERANGAN"
        self.kolom_kode  = "DB/CR"
        self.target_kode = "Amount"
        self.debit_code  = "D"
        self.kredit_code = "K"
        self.header_per_page = True

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

        thresh = cv.adaptiveThreshold(
            gray, 255,
            cv.ADAPTIVE_THRESH_MEAN_C,
            cv.THRESH_BINARY,
            21,  
            9    
        )

        kernel = cv.getStructuringElement(cv.MORPH_RECT, (1,1))
        dilated = cv.dilate(thresh, kernel, iterations=1)
        denoised = cv.fastNlMeansDenoising(dilated, h=15)
        kernel_sharp = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
        sharp = cv.filter2D(denoised, -1, kernel_sharp)
        return sharp

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
    def find_header_pdf(self, img, y_tolerance=25, min_ratio=0.6):
        if len(img.shape) == 3: 
            height, width, _ = img.shape
        else:
            height, width = img.shape
            
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
                for header in self.HEADERS:
                    if self.fuzzy_match(word, header):
                        (x_min, y_min) = bbox[0]
                        (x_max, y_max) = bbox[2]
                        detected_headers[header] = {
                            "x_min": max(0, int(x_min)),
                            "x_max": min(width, int(x_max)),
                            "y_min": int(y_min),
                            "y_max": int(y_max)
                        }
            match_ratio = len(detected_headers) / len(self.HEADERS)
            if match_ratio >= min_ratio:
                headers_found = detected_headers
                break

        headers_sorted = dict(sorted(headers_found.items(), key=lambda item: item[1]['x_min']))
        return headers_sorted, results

    # ------------------------
    # Date Regex Helper
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

    # ------------------------------------------------
    # Klastering berdasarkan center koordinat x Header
    # ------------------------------------------------
    def cluster_table(self, hasil_ocr, koordinat_kolom, init_kmeans: Optional[KMeans] = None):
        daftar_header = list(koordinat_kolom.keys())
        header_centers = [ (v["x_min"] + v["x_max"]) / 2.0 for v in koordinat_kolom.values() ]
        header_y_max   = {h: v["y_max"] for h, v in koordinat_kolom.items()}

        words = []
        X = []
        Y = []
        for (bbox, word, conf) in hasil_ocr:
            if not isinstance(word, str) or not word.strip():
                continue
            x_min, y_min = bbox[0]
            x_max, _ = bbox[2]
            x_center = (x_min + x_max) / 2.0
            X.append([x_center])
            Y.append(float(y_min))
            words.append(word)

        if len(X) == 0:
            return [], None

        X = np.array(X)
        Y = np.array(Y)

        k_means = None
        labels  = None
        if init_kmeans is not None:
            try:
                labels  = init_kmeans.predict(X)
                k_means = init_kmeans
            except Exception:
                labels = []
                for x in X:
                    dists = [abs(x[0] - hc) for hc in header_centers]
                    labels.append(int(np.argmin(dists)))
                labels = np.array(labels)
        else:
            if len(X) >= len(header_centers) and len(header_centers) > 0:
                center_header = np.array([[c] for c in header_centers])
                n_clusters = len(center_header)
                try:
                    k_means = KMeans(n_clusters=n_clusters, init=center_header, n_init=1, random_state=42)
                    labels  = k_means.fit_predict(X)
                except Exception:
                    k_means = None
                    labels = []
                    for x in X:
                        dists = [abs(x[0] - hc) for hc in header_centers]
                        labels.append(int(np.argmin(dists)))
                    labels = np.array(labels)
            else:
                labels = []
                for x in X:
                    dists = [abs(x[0] - hc) for hc in header_centers]
                    labels.append(int(np.argmin(dists)))
                labels = np.array(labels)
                    
        cluster_to_header = {}
        if k_means is not None:
            for i, center in enumerate(k_means.cluster_centers_):
                nearest_idx = int(np.argmin([abs(center[0] - hc) for hc in header_centers]))
                cluster_to_header[i] = daftar_header[nearest_idx]
        else:
            unique_labels = sorted(set(labels.tolist()))
            for lbl in unique_labels:
                idx = int(lbl) if lbl < len(daftar_header) else 0
                cluster_to_header[lbl] = daftar_header[idx]
                
        hasil_klaster = []
        for word, label, x_arr, y_val in zip(words, labels, X, Y):
            header_name = cluster_to_header.get(int(label), daftar_header[0] if daftar_header else "col")
            if y_val > header_y_max.get(header_name, -1):
                hasil_klaster.append({"word": word, "header": header_name, "x_center": float(x_arr[0]), 
                                      "y_min": float(y_val)})
        return hasil_klaster, k_means

    # -------------------------------------------------
    # Klastering berdasarkan min koordinat y Header
    # ------------------------------------------------
    def build_table(self, hasil_klaster, eps=15, min_samples=1, header_order=None):
        y_coords = np.array([[c["y_min"]] for c in hasil_klaster])
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(y_coords)
        labels = db.labels_
        for c, lbl in zip(hasil_klaster, labels):
            c["row"] = lbl

        rows = {}
        for c in hasil_klaster:
            row = rows.setdefault(c["row"], {})
            if c["header"] not in row:
                row[c["header"]] = []
            row[c["header"]].append((c["x_center"], c["word"]))

        table = []
        for row_id in sorted(rows.keys()):
            row_data = {}
            for header, words in rows[row_id].items():
                words_sorted = [w for _, w in sorted(words, key=lambda x: x[0])]
                row_data[header] = " ".join(words_sorted)
            table.append(row_data)

        df = pd.DataFrame(table)
        if header_order is not None:
            df = df.reindex(columns=header_order)

        return df.reset_index(drop=True)
    
    # ----------------------------
    # Normalisasi & Date cleaning
    # ----------------------------
    def normalisasi_token(self, tok: str) -> str:
        s = tok
        s = s.replace('：', ':').replace('．', '.')
        s = s.replace('\u2009', ' ')
        if re.search(r"[:\.]", s) or re.search(r"\d+[OoIlI]\d", s):
            s = re.sub(r'[Oo]', '0', s)
            s = re.sub(r'[IlI]', '1', s)
        return s

    def extract_first_date_only(self, raw: str) -> str:
        if not raw:
            return ""
        tokens = re.split(r'\s+', raw)
        tokens_norm = [self.normalisasi_token(t) for t in tokens]
        joined_norm = " ".join(tokens_norm)

        without_time = self.TIME_RE.sub('', joined_norm).strip(" -,:;·•")
        matches = [m.group() for m in self.DATE_RE.finditer(without_time)]
        if matches:
            return matches[0].strip()
        matches = [m.group() for m in self.DATE_RE.finditer(raw)]
        if matches:
            return matches[0].strip()
        return ""

    # ------------------------
    # Extracting Table
    # ------------------------
    def extracting_table(self, df_table, hasil_ocr, header_y_min=0, page=0):
        YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")

        detected_year = None
        if not df_table.empty:
            first_col = df_table.columns[0]
            for val in df_table[first_col]:
                if pd.isna(val):
                    continue
                m = YEAR_RE.search(str(val))
                if m:
                    detected_year = int(m.group())
                    break

        if not detected_year and hasil_ocr is not None:
            candidates = []
            for (bbox, word, conf) in hasil_ocr:
                if not isinstance(word, str):
                    word = str(word)
                m = YEAR_RE.search(word)
                if m:
                    year_val = int(m.group())
                    y = int(sum([p[1] for p in bbox]) / 4)
                    if y < header_y_min:
                        candidates.append((y, year_val))
            if candidates:
                detected_year = min(candidates, key=lambda c: c[0])[1]

        if not detected_year:
            detected_year = datetime.now().year

        cols = list(df_table.columns)
        if not cols:
            raise ValueError("Input DataFrame kosong (tidak ada kolom).")
        first_col = cols[0]

        results = []
        for _, r in df_table.iterrows():
            first_text = "" if pd.isna(r[first_col]) else str(r[first_col]).strip()
            date_only = self.extract_first_date_only(first_text)
            if not date_only:
                continue

            row_data = {}
            if re.match(r"^\d{1,2}[/-]\d{1,2}$", date_only):
                row_data[first_col] = f"{date_only}/{detected_year}"
            else:
                row_data[first_col] = date_only

            for col_name in cols[1:]:
                raw = "" if pd.isna(r[col_name]) else str(r[col_name]).strip()
                row_data[col_name] = raw
            results.append(row_data)

        if not results:
            return pd.DataFrame(columns=cols)

        df_out = pd.DataFrame(results)
        df_out = df_out.reindex(columns=cols).fillna("").reset_index(drop=True)
        print(f"✅ Berhasil mengekstrak {len(df_out)} baris transaksi pada halaman ke - {page}.")
        return df_out


    # ------------------------
    # Formating to Float (Decimal) Format
    # ------------------------
    def clean_number(self, val: str) -> str:
        if not isinstance(val, str):
            return val
        val = val.replace("/", "7")
        return val
    
    def to_number_sal(self, x):
        if pd.isna(x) or str(x).strip() == "":
            return 0.0

        s = str(x).strip()
        s = s.replace("~", "-").replace("–", "-").replace("−", "-")
        s = s.replace(" ", "")
        s = s.replace(",", "")

        import re
        match = re.search(r"-?\d+(\.\d+)?", s)
        if match:
            try:
                return float(match.group(0))
            except ValueError:
                pass

        print("Gagal konversi:", repr(x))
        return 0.0


    def normalize_table(self, df: pd.DataFrame) -> pd.DataFrame:
        num_cols = ["MUTASI", "SALDO"]
        for col in num_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).apply(self.clean_number)
        return df
    
    def enforce_thousands_groups(self, s: str, group_len: int = 3) -> str:
        s = re.sub(r"[^0-9.,]", "", str(s))  
        if "," in s:
            main, dec = s.split(",", 1)
        else:
            main, dec = s, None
        parts = main.split(".")

        fixed_parts = []
        for i, p in enumerate(parts):
            if p == "":
                continue
            if i == 0:
                fixed_parts.append(p)
            else:
                while len(p) > group_len:
                    p = p[:-1]
                fixed_parts.append(p)

        fixed_main = ".".join(fixed_parts)
        return f"{fixed_main},{dec}" if dec is not None else fixed_main

    def to_number(self, x: str) -> Tuple[float, str]:
        if pd.isna(x) or x == "":
            return 0.0, ""
        
        is_debit = "DB" in x.upper()
        x = x.replace("DB", "").replace("CR", "").strip()
        x = self.enforce_thousands_groups(x, group_len=3)
        x = x.replace(",", "").replace(".", ".")
        return float(x), ("Debit" if is_debit else "Kredit")
    
    # ------------------------
    # Find Debit & Kredit
    # ------------------------
    def debit_and_kredit(self, df: pd.DataFrame) -> pd.DataFrame:
        df["Debit"]  = 0.0
        df["Kredit"] = 0.0

        for i in range(len(df)):
            mutasi_raw = str(df.loc[i, "MUTASI"])
            if pd.isna(mutasi_raw) or mutasi_raw.strip() == "":
                continue

            angka, tipe = self.to_number(mutasi_raw) 
            if tipe == "Debit":
                df.loc[i, "Debit"] = angka
            elif tipe == "Kredit":
                df.loc[i, "Kredit"] = angka
        df_final = df.drop(columns=["MUTASI", "CBG"], errors="ignore")

        cols = list(df_final.columns)
        cols.remove("Debit")
        cols.remove("Kredit")
        saldo_idx = cols.index("SALDO")
        cols      = cols[:saldo_idx] + ["Debit", "Kredit"] + cols[saldo_idx:]
        df_final  = df_final[cols]
        return df_final

    # ---------------------------------
    # Add Saldo Awal (Opening Balance)
    # ---------------------------------
    def add_saldo_awal(self, df: pd.DataFrame) -> pd.DataFrame:
        df_copy = df.copy()
        for i in range (len(df_copy)):
            df_copy.loc[i, "SALDO"] = self.to_number_sal(df_copy.loc[i, "SALDO"])
            
        if not df_copy.empty:
            first_row = df_copy.iloc[0]
            df_copy = df_copy.drop(index=0).reset_index(drop=True)
        else:
            raise ValueError("DataFrame kosong, tidak bisa menambahkan saldo awal.")

        debit  = float(first_row.get("Debit", 0) or 0)
        kredit = float(first_row.get("Kredit", 0) or 0)
        saldo  = float(first_row.get("SALDO", 0) or 0)

        if debit != 0:
            saldo_awal = saldo + debit
        elif kredit != 0:
            saldo_awal = saldo - kredit
        else:
            saldo_awal = saldo

        saldo_awal_row = {
            col: 0.0 if col in ["Debit", "Kredit", "SALDO"] else "" 
            for col in df.columns
        }
        saldo_awal_row["KETERANGAN"] = "SALDO AWAL"
        saldo_awal_row["SALDO"] = saldo_awal

        df_copy = pd.concat(
            [pd.DataFrame([saldo_awal_row]), df_copy], ignore_index=True
        )

        return df_copy
    
    # ---------------------------------
    # Add Saldo Akhir (Closing Balance)
    # ---------------------------------
    def add_saldo_akhir(self, df):
        df_copy = df.copy()
        saldo_awal = df_copy.loc[0, "SALDO"]

        debit_temp, kredit_temp = 0.0, 0.0
        for i in range(1, len(df_copy)):
            debit_temp  += df_copy.loc[i, "Debit"]
            kredit_temp += df_copy.loc[i, "Kredit"]

        saldo_akhir_ = saldo_awal - debit_temp + kredit_temp
        saldo_akhir = {
            "TANGGAL": "",
            "KETERANGAN": "SALDO AKHIR",
            "SALDO": saldo_akhir_
        }
        df_copy = pd.concat([df_copy, pd.DataFrame([saldo_akhir])], ignore_index=True)
        return df_copy

    # ---------------------------------
    # Hapus Duplikasi Baris (Jika Ada)
    # ---------------------------------
    def drop_next_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        df_copy = df.copy()     
        df_copy.columns = [str(c).strip().lower() for c in df_copy.columns]   
        main_cols = ["keterangan", "debit", "kredit", "saldo"]
        mask = df_copy[main_cols].shift().eq(df_copy[main_cols]).all(axis=1)
        df_cleaned = df_copy[~mask].reset_index(drop=True)
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
    
    # --------------------------------------------
    # Menghapus Baris jika hanya satu kolom terisi
    # --------------------------------------------
    def drop_incomplete(self, df: pd.DataFrame) -> pd.DataFrame:
        df_copy = df.copy()
        df_copy.columns = [str(c).strip().lower() for c in df_copy.columns]
        main_cols = ["tanggal", "keterangan", "debit", "kredit", "saldo"]

        for col in main_cols:
            if col not in df_copy.columns:
                df_copy[col] = np.nan

        to_drop = []
        for idx, row in df_copy.iterrows():
            values = []
            for col in main_cols:
                val = row[col]
                if pd.isna(val):
                    values.append("")
                    continue
                sval = str(val).strip().lower()
                try:
                    sval_clean = sval.replace(",", "").replace("rp", "").strip()
                    num = float(sval_clean)
                    if num == 0:
                        sval = ""
                except Exception:
                    pass
                values.append(sval)
            non_empty = [v for v in values if v not in ("", "nan", "none")]
            if len(non_empty) <= 1:
                to_drop.append(idx)

        df_cleaned = df_copy.drop(index=to_drop).reset_index(drop=True)
        return df_cleaned

    # ---------------------------------
    # Converting Many Functions
    # ---------------------------------
    def convert(self, pdf_path: str, pages: Union[str, List[int]] = "all", output_excel=None):
        images = convert_from_path(pdf_path, dpi=170)
        if pages != "all":
            images = [images[i-1] for i in pages if 0 < i <= len(images)]
            
        all_pages_df = []
        header_cache = None
        scale_percent = 100
        for page_idx, image in enumerate(images):
            img_array = np.array(image)
            img_array = self.noise_removal(img_array)

            h, w = img_array.shape[:2]
            new_w = int(w * scale_percent / 100)
            new_h = int(h * scale_percent / 100)
            img_array = cv.resize(img_array, (new_w, new_h), interpolation=cv.INTER_AREA)
            
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
            if page_idx == 0:
                hasil_klaster, k_means = self.cluster_table(hasil_ocr, kolom_coords, init_kmeans=None)
                if k_means is not None:
                    kmeans_cache = k_means
            else:
                hasil_klaster, _ = self.cluster_table(hasil_ocr, kolom_coords, init_kmeans=kmeans_cache)            
            df_table         = self.build_table(hasil_klaster, header_order=self.HEADERS)
            if self.header_per_page or page_idx == 0:
                minimum_header = 1400
            else:
                minimum_header = 0
            df = self.extracting_table(df_table, hasil_ocr, header_y_min=minimum_header, page=page_idx + 1)
            if df.empty:
                print(f"⚠️ Dataframe kosong di halaman {page_idx + 1}, dilewati...") # ----- STAGE 3
                continue
            df        = self.normalize_table(df)
            df_knd    = self.debit_and_kredit(df)
            all_pages_df.append(df_knd)
            
        if not all_pages_df:
            print("❌ Tidak ada halaman yang berhasil diekstrak.")
            return pd.DataFrame()
        
        df_combined = pd.concat(all_pages_df, ignore_index=True)
        df_combined = self.add_saldo_awal(df_combined)
        df_close    = self.add_saldo_akhir(df_combined)
        df_dp       = self.drop_next_duplicates(df_close)
        df_close    = self.drop_incomplete(df_dp)
        df_final    = self.capitalize_and_date(df_close, output_excel)
        return df_final