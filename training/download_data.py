from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "south_german_credit.csv"


def main():
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)

    try:
        from ucimlrepo import fetch_ucirepo

        print("Downloading South German Credit from UCI...")
        dataset = fetch_ucirepo(id=573)

        X = dataset.data.features.copy()
        y = dataset.data.targets.copy()

        y_col = y.columns[0]
        frame = pd.concat([X, y], axis=1)
        frame = frame.rename(columns={y_col: "credit_risk"})

    except Exception as exc:
        print(f"UCI client failed: {exc}")
        print("Trying raw GitHub mirror...")

        url = (
            "https://raw.githubusercontent.com/"
            "bbalint642/South_German_Credit_Dataset/"
            "main/SouthGermanCredit.asc"
        )

        frame = pd.read_csv(url, sep=r"\s+")

    frame.columns = [str(c).strip() for c in frame.columns]

    expected = {
        "status", "duration", "credit_history", "purpose", "amount",
        "savings", "employment_duration", "installment_rate",
        "personal_status_sex", "other_debtors", "present_residence",
        "property", "age", "other_installment_plans", "housing",
        "number_credits", "job", "people_liable", "telephone",
        "foreign_worker", "credit_risk",
    }

    missing = expected.difference(frame.columns)
    if missing:
        raise RuntimeError(f"Dataset missing columns: {sorted(missing)}")

    frame.to_csv(RAW_PATH, index=False)
    print(f"Saved {len(frame)} rows to {RAW_PATH}")


if __name__ == "__main__":
    main()
