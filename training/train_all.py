from training.download_data import main as download
from training.etl_pyspark import main as etl
from training.train_model import main as train


def main():
    print("\n=== 1/3 DOWNLOAD DATA ===")
    download()

    print("\n=== 2/3 ETL ===")
    etl()

    print("\n=== 3/3 TRAIN MODEL ===")
    train()

    print("\nAll steps complete.")


if __name__ == "__main__":
    main()
