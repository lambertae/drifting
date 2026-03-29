"""CLI entry-point for PyTorch Drift training."""
import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Drift PyTorch training.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument(
        "--gen",
        action="store_true",
        help="Run generator training loop.  Default runs MAE training.",
    )
    parser.add_argument(
        "--workdir",
        type=str,
        default="runs",
        help="Local workdir root for checkpoints and logs.",
    )
    args = parser.parse_args()
    args.output_dir = args.workdir

    if args.gen:
        from train import main as train_gen_main
        train_gen_main(args)
    else:
        from train_mae import main as train_mae_main
        train_mae_main(args)


if __name__ == "__main__":
    main()
