#!/usr/bin/env python
"""
Download from W&B the raw dataset and apply some basic data cleaning, exporting the result to a new artifact
"""
import argparse
import logging
import wandb
import pandas as pd
import os


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    artifact_local_path = run.use_artifact(args.input_artifact).file()
    logger.info(
        f"Artifact {args.input_artifact} downloaded to {artifact_local_path}")

    df = pd.read_csv(artifact_local_path)
    logger.info(f"Dataset loaded - shape: {df.shape}")

    # drop duplicated rows
    df.drop_duplicates(inplace=True, ignore_index=True)
    n_duplicates = df.duplicated().sum()
    if n_duplicates > 0:
        logger.info(f"Removed {n_duplicates} duplicated rows")
    else:
        logger.info("No duplicated rows found")

    # fix price range - drop outliers
    idx = df['price'].between(args.min_price, args.max_price)
    df = df[idx].copy()
    n_outliers = ~idx.sum()
    if n_outliers > 0:
        logger.info(f"Removed {n_outliers} outliers from price column")
    else:
        logger.info("No outliers found in price column")

    # drop outliers in minimum nights
    idx_nights = df['minimum_nights'].between(args.min_nights, args.max_nights)
    df = df[idx_nights].copy()
    n_outliers_nights = ~idx.sum()
    if n_outliers_nights > 0:
        logger.info(
            f"Removed {n_outliers_nights} outliers from minimum nights column")
    else:
        logger.info("No outliers found in minimum nights column")

    artifact = wandb.Artifact(
        name=args.output_artifact,
        type=args.output_type,
        description=args.output_description
    )

    df.to_csv(args.output_artifact, index=False)
    artifact.add_file(args.output_artifact)
    run.log_artifact(artifact)
    artifact.wait()
    logger.info(f"Artifact {args.output_artifact} saved and logged")

    run.finish()

    os.remove(args.output_artifact)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Data cleaning")

    parser.add_argument(
        "--input_artifact",
        type=str,
        help="The name of the input artifact to be cleaned",
        required=True
    )

    parser.add_argument(
        "--output_artifact",
        type=str,
        help="The name of the output artifact to be created",
        required=True
    )

    parser.add_argument(
        "--output_type",
        type=str,
        help="The type of the output artifact to be created",
        required=True
    )

    parser.add_argument(
        "--output_description",
        type=str,
        help="The description of the output artifact to be created",
        required=True
    )

    parser.add_argument(
        "--min_price",
        type=int,
        help="Minimum price to filter the dataset",
        required=True
    )

    parser.add_argument(
        "--max_price",
        type=int,
        help="Maximum price to filter the dataset",
        required=True
    )

    parser.add_argument(
        "--min_nights",
        type=int,
        help="Minimum nights to filter the dataset",
        required=True
    )

    parser.add_argument(
        "--max_nights",
        type=int,
        help="Maximum nights to filter the dataset",
        required=True
    )

    args = parser.parse_args()

    go(args)
