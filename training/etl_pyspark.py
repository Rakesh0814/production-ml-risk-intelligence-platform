from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "south_german_credit.csv"
PROCESSED_PATH = ROOT / "data" / "processed" / "credit_risk.csv"

DROP_SENSITIVE = ["personal_status_sex", "age", "foreign_worker"]


def pandas_fallback():
    print("Running pandas fallback ETL...")
    df = pd.read_csv(RAW_PATH)

    df = df.drop_duplicates().dropna()

    for column in df.columns:
        df[column] = pd.to_numeric(df[column], errors="raise")

    df["default_risk"] = 1 - df["credit_risk"].astype(int)

    df = df.drop(columns=DROP_SENSITIVE + ["credit_risk"])

    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)
    print(f"Saved processed data to {PROCESSED_PATH}")


def spark_etl():
    from pyspark.sql import SparkSession, functions as F

    print("Starting PySpark local ETL...")
    spark = (
        SparkSession.builder
        .appName("RiskLensCreditETL")
        .master("local[*]")
        .getOrCreate()
    )

    try:
        df = (
            spark.read
            .option("header", True)
            .option("inferSchema", True)
            .csv(str(RAW_PATH))
        )

        df = df.dropDuplicates().dropna()

        for column in df.columns:
            df = df.withColumn(column, F.col(column).cast("double"))

        df = df.withColumn(
            "default_risk",
            (F.lit(1) - F.col("credit_risk")).cast("int"),
        )

        df = df.drop(*DROP_SENSITIVE, "credit_risk")

        PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)

        # Coalesce because this portfolio dataset is only 1,000 rows.
        temp_dir = PROCESSED_PATH.parent / "_spark_output"
        if temp_dir.exists():
            import shutil
            shutil.rmtree(temp_dir)

        df.coalesce(1).write.mode("overwrite").option("header", True).csv(
            str(temp_dir)
        )

        part_file = next(temp_dir.glob("part-*.csv"))
        PROCESSED_PATH.write_bytes(part_file.read_bytes())

        import shutil
        shutil.rmtree(temp_dir)

        print(f"Saved processed data to {PROCESSED_PATH}")
    finally:
        spark.stop()


def main():
    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"{RAW_PATH} does not exist. Run python -m training.download_data first."
        )

    force_pandas = os.getenv("RISK_ETL_ENGINE", "").lower() == "pandas"

    if force_pandas:
        pandas_fallback()
        return

    try:
        spark_etl()
    except Exception as exc:
        print(f"PySpark ETL unavailable: {exc}")
        print("Falling back to pandas so the project can still run locally.")
        pandas_fallback()


if __name__ == "__main__":
    main()
