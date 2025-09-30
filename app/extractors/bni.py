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

class BNIExtractor: 
    def __init__(self, reader):
        self._TIME_SEP    = r"[:\.\uFF1A]"
        self.TIME_RE      = re.compile(
            rf"(?i)(?<!\d)(?:[01]?\d|2[0-3]){self._TIME_SEP}[0-5]\d(?:{self._TIME_SEP}[0-5]\d)?\s*(?:am|pm)?(?!\d)"
        )
        self.DATE_FORMAT = "dd-MMM-yyyy"
        self.reader      = reader
        self.DATE_RE     = re.compile(self.to_regex(self.DATE_FORMAT))
        self.HEADERS     = ["Tanggal", "Uraian Transaksi", "Kategori", "Tipe", "Jumlah Pembayaran", 
                            "Saldo", "Pecah"]
        self.keterangan  = "Transaksi"
        self.kolom_kode  = "Tipe"
        self.target_kode = "Jumlah Pembayaran"
        self.debit_code  = "Db."
        self.kredit_code = "Cr."
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
        headers_found.update({
            "Uraian Transaksi": {
                "x_min": 600,
                "x_max": 1198,
                "y_min": 1211,
                "y_max": 1284
            },
            "Tanggal": {
                "x_min": 150,
                "x_max": 550,
                "y_min": 1211,
                "y_max": 1284
            }
        })
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
    # Parsing Mutation
    # ------------------------
    def parse_mutasi(self, cell: str)-> Tuple[Optional[str], Optional[float]]:
        if pd.isna(cell):
            return (0.0, "")
        s = str(cell).strip()
        if s == "":
            return (0.0, "")
        m      = re.search(r"([A-Za-z]{1,3})\s*$", s)
        code   = m.group(1).upper() if m else ""
        amount = s[:m.start()].strip() if m else s
        
        amount = self.to_number(amount)
        return (amount, code)
    
    # ------------------------
    # Find Debit & Kredit
    # ------------------------
    def debit_and_kredit(self, df: pd.DataFrame, kolom_keterangan: str = None) -> pd.DataFrame:
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]
        self.kolom_kode = self.kolom_kode.lower()
        self.target_kode = self.target_kode.lower()
        if kolom_keterangan:
            kolom_keterangan = kolom_keterangan.lower()

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
            elif self.kolom_kode == self.target_kode and self.kolom_kode in df.columns:
                parsed = df[self.kolom_kode].apply(lambda x: pd.Series(self.parse_mutasi(x)))
                parsed.columns = ["amount_parsed", "code_parsed"]
                df = pd.concat([df, parsed], axis=1)
                self.kolom_kode = "code_parsed"
                self.target_kode = "amount_parsed"

            df["debit"] = 0.0
            df["kredit"] = 0.0
            if self.target_kode in df.columns: 
                df[self.target_kode] = df[self.target_kode].apply(self.to_number)
                
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
        df_copy    = df_copy.iloc[::-1].reset_index(drop=True)
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
        numeric_cols = ["debit", "kredit", "saldo"]
        main_cols = df_copy.columns.tolist()

        for c in numeric_cols:
            if c in df_copy.columns:
                df_copy[c] = (pd.to_numeric(df_copy[c].astype(str).str.replace(",", ""), errors="coerce",).fillna(0)
                )
        mask_non_empty = df_copy[main_cols].apply(lambda x: x.notna() & (x.astype(str).str.strip() != ""))
        for c in numeric_cols:
            if c in df_copy.columns:
                mask_non_empty[c] = df_copy[c].notna() & (df_copy[c] != 0)

        count_non_empty = mask_non_empty.sum(axis=1)
        cond_incomplete = count_non_empty <= 1
        cond_all_zero = (df_copy[numeric_cols] == 0).all(axis=1)
        df_cleaned = df_copy[~(cond_incomplete | cond_all_zero)].reset_index(drop=True)
        return df_cleaned

    # ---------------------------------
    # Converting Many Functions
    # ---------------------------------
    def convert(self, pdf_path: str, pages: Union[str, List[int]] = "all", output_excel=None):
        images = convert_from_path(pdf_path, dpi=300)
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
            if page_idx == 0:
                hasil_klaster, k_means = self.cluster_table(hasil_ocr, kolom_coords, init_kmeans=None)
                if k_means is not None:
                    kmeans_cache = k_means
            else:
                hasil_klaster, _ = self.cluster_table(hasil_ocr, kolom_coords, init_kmeans=kmeans_cache)            
            df_table = self.build_table(hasil_klaster, header_order=self.HEADERS)
            if self.header_per_page or page_idx == 0:
                minimum_header = 1500
            else:
                minimum_header = 0
            df        = self.extracting_table(df_table, hasil_ocr, header_y_min=minimum_header, page = page_idx + 1)
            if df.empty:
                print(f"⚠️ Dataframe kosong di halaman {page_idx + 1}, dilewati...") # ----- STAGE 3
                continue
            df_knd    = self.debit_and_kredit(df, kolom_keterangan=self.keterangan)
            all_pages_df.append(df_knd)
            
        if not all_pages_df:
            print("❌ Tidak ada halaman yang berhasil diekstrak.")
            return pd.DataFrame()
        
        df_combined = pd.concat(all_pages_df, ignore_index=True)
        df_open     = self.add_saldo_awal(df_combined)
        df_close    = self.add_saldo_akhir(df_open)
        df_close    = self.drop_incomplete(df_close)
        df_close    = self.drop_next_duplicates(df_close)
        df_final    = self.capitalize_and_date(df_close, output_excel)
        return df_final