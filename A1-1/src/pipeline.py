import gzip
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

COLUMNS = {
    "invoice": "invoice", "invoiceno": "invoice",
    "stockcode": "stock_code",
    "description": "description",
    "quantity": "quantity",
    "invoicedate": "order_date",
    "price": "price", "unitprice": "price",
    "customerid": "customer_id",
    "country": "country",
}
KEEP = ["invoice", "stock_code", "description", "quantity",
        "order_date", "price", "customer_id", "country"]


class DataAnalyzer:
    def __init__(self, data_dir=None, seed=42):
        self.data_dir = Path(data_dir) if data_dir else PROJECT_ROOT / "data" / "raw"
        self.seed = seed
        self.df = None
        self.images = None

    def load_data(self, image_file="train-images-idx3-ubyte.gz"):
        self.df = self._read_transactions()
        self.images = self._read_images(self.data_dir / image_file)
        return self.df

    def _read_transactions(self):
        cache = self.data_dir / "online_retail_ii.csv"
        types = {"invoice": str, "stock_code": str, "customer_id": "Int64"}
        if cache.exists():
            return pd.read_csv(cache, parse_dates=["order_date"], dtype=types)

        files = list(self.data_dir.glob("*.xlsx"))
        if not files:
            raise FileNotFoundError(f"{self.data_dir} 안에 .xlsx 파일이 없습니다. README의 데이터 내려받기를 따라 하세요.")
        df = pd.concat(pd.read_excel(files[0], sheet_name=None).values(), ignore_index=True)
        df = df.rename(columns=lambda c: COLUMNS.get(c.lower().replace(" ", "").replace("_", ""), c))
        df = df[KEEP].astype(types)
        df.to_csv(cache, index=False)  # 엑셀은 느려서 CSV로 저장
        return df

    @staticmethod
    def _read_images(path):
        with gzip.open(path, "rb") as f:
            raw = f.read()
        n, rows, cols = np.frombuffer(raw, dtype=">i4", count=3, offset=4)
        return np.frombuffer(raw, dtype=np.uint8, offset=16).reshape(n, rows, cols)

    # 결측치 처리 부분: 상품명(description) 결측치를 상품코드(stock_code)별로 채우기
    def handle_missing_values(self, target_col="description", group_col="stock_code", strategy="mode"):
        known = self.df.dropna(subset=[target_col]).groupby(group_col)[target_col]
        fill = known.agg(lambda s: s.mode().iloc[0]) if strategy == "mode" else known.agg(strategy)
        self.df[target_col] = self.df[target_col].fillna(self.df[group_col].map(fill))

    # 상품과 이미지의 연결은 난수(시드 고정)로 정한 학습용 시뮬레이션
    def add_image_features(self):
        codes = self.df["stock_code"].unique()
        rng = np.random.default_rng(self.seed)
        imgs = self.images[rng.choice(len(self.images), size=len(codes), replace=False)]
        products = pd.DataFrame({
            "stock_code": codes,
            "img_mean": imgs.mean(axis=(1, 2)),
            "img_std": imgs.std(axis=(1, 2)),
        })
        self.df = self.df.merge(products, on="stock_code", how="left")

    def add_text_features(self):
        text = self.df["description"]
        self.df["name_word_count"] = text.str.split().str.len()
        self.df["name_length"] = text.str.len()

    def clean_data(self):
        df = self.df.drop_duplicates()
        is_sale = (df["quantity"] > 0) & (df["price"] > 0) & ~df["invoice"].str.startswith("C")
        df = df[is_sale].reset_index(drop=True)
        df["amount"] = df["quantity"] * df["price"]
        self.df = df

    
    def _iqr_bounds(self, column, threshold):
        q1, q3 = self.df[column].quantile([0.25, 0.75])
        iqr = q3 - q1
        return q1 - threshold * iqr, q3 + threshold * iqr

    def detect_outliers(self, column, threshold=1.5):
        lower, upper = self._iqr_bounds(column, threshold)
        return self.df[(self.df[column] < lower) | (self.df[column] > upper)]

    def clip_outliers(self, column, threshold=1.5):
        lower, upper = self._iqr_bounds(column, threshold)
        self.df[f"{column}_clipped"] = self.df[column].clip(lower, upper)
    # RFM의 Monetary(금액)은 원본 amount로 계산 하는데 VIP는 원래 대량 구매 고객, VIP와 일반 고객 차이의 경계가 모호해짐


    @staticmethod
    # RFM 점수 계산: 1~5점으로 나누어 점수
    # Recency는 최근 구매일이 가까울수록 점수가 높음, Frequency와 Monetary는 값이 클수록 점수가 높음
    def _score(s):
        cuts = s.quantile([0.2, 0.4, 0.6, 0.8]).to_numpy()
        return np.searchsorted(cuts, s.to_numpy()) + 1

    def calculate_rfm(self, customer_col="customer_id", date_col="order_date",
                      amount_col="amount", invoice_col="invoice"):
        last_day = self.df[date_col].max()
        df = self.df.dropna(subset=[customer_col])
        rfm = df.groupby(customer_col).agg(
            last_order=(date_col, "max"),
            frequency=(invoice_col, "nunique"),
            monetary=(amount_col, "sum"),
        )
        rfm["recency"] = (last_day - rfm.pop("last_order")).dt.days

        rfm["R"] = 6 - self._score(rfm["recency"])
        rfm["F"] = self._score(rfm["frequency"])
        rfm["M"] = self._score(rfm["monetary"])

        r, f, m = rfm["R"], rfm["F"], rfm["M"]
        rfm["segment"] = np.select(
            [(r >= 4) & (f >= 4) & (m >= 4), (f >= 4) & (r >= 3), (r >= 4) & (f <= 2), r <= 2],
            ["VIP", "Loyal", "New", "Churned"],
            default="Regular",
        )
        return rfm

    