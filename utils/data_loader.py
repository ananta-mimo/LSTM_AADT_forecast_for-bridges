import pandas as pd

def load_adt_csv(path: str, year_col: str = "year1", adt_col: str = "totadt1") -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[[year_col, adt_col]].dropna()
    df[year_col] = df[year_col].astype(int)
    df[adt_col] = df[adt_col].astype(float)
    df = df.set_index(year_col).sort_index()
    return df[[adt_col]]
