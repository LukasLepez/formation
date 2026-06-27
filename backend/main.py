from indusense.processing.ingestion import (
    GoldDatasetConfig,
    configure_logging,
    run_layer_pipeline,
)


def main():
    configure_logging("INFO")
    result = run_layer_pipeline(GoldDatasetConfig(layer="all"))
    if result is not None:
        print(f"Gold Dataset construit: {result.shape[0]:,} lignes x {result.shape[1]} colonnes")


if __name__ == "__main__":
    main()
