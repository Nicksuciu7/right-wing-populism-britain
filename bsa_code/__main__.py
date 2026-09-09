import argparse

from .config import ensure_dirs, load_config
from .attitudes import harmonise_attitudes
from .demographics import harmonise_demographics
from .ingest import ingest_raw_waves
from .inventory import build_inventory
from .model_frame import build_model_frame
from .nlp_item_discovery import discover_question_candidates
from .model_tuning import tune_model
from .train_eval import train_and_evaluate
from .scoring import build_scores
from .trends import predictor_trends, rwp_prevalence
from .utils import set_seed
from .validate_config import validate_config, validate_schema_outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="BSA RWP code")
    parser.add_argument(
        "--config",
        default="config/reproduce.json",
        help="Path to code config JSON",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("inventory")
    subparsers.add_parser("ingest")
    subparsers.add_parser("harmonise_demographics")
    subparsers.add_parser("harmonise_attitudes")
    subparsers.add_parser("validate_config")
    subparsers.add_parser("validate_schema")
    subparsers.add_parser("score")
    subparsers.add_parser("build_model_frame")
    subparsers.add_parser("train_eval")
    subparsers.add_parser("tune_model")
    subparsers.add_parser("trends")
    subparsers.add_parser("nlp_discovery")

    args = parser.parse_args()

    config = load_config(args.config)
    validate_config(args.config)
    if args.command == "validate_config":
        print("Configuration and survey inputs validated.")
        return
    ensure_dirs(config.get("paths", {}))
    set_seed(config["project"]["seed"])

    if args.command == "inventory":
        build_inventory(args.config)
        return
    if args.command == "ingest":
        ingest_raw_waves(args.config)
        return
    if args.command == "harmonise_demographics":
        harmonise_demographics(args.config)
        return
    if args.command == "harmonise_attitudes":
        harmonise_attitudes(args.config)
        return
    if args.command == "validate_schema":
        validate_schema_outputs(args.config)
        return
    if args.command == "score":
        build_scores(args.config)
        return
    if args.command == "build_model_frame":
        build_model_frame(args.config)
        return
    if args.command == "train_eval":
        train_and_evaluate(args.config)
        return
    if args.command == "tune_model":
        tune_model(args.config)
        return
    if args.command == "trends":
        rwp_prevalence(args.config)
        predictor_trends(args.config)
        return
    if args.command == "nlp_discovery":
        discover_question_candidates(args.config)
        return
    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
