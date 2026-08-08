"""Run the Stage 2 production ML pipeline."""

from __future__ import annotations

import argparse

from datasphere.ml.pipeline import run_stage2_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage 2 ML pipeline")
    parser.add_argument("--sample-rows", type=int, default=None, help="Optional row limit for faster runs")
    parser.add_argument("--no-tuning", action="store_true", help="Skip hyperparameter tuning")
    parser.add_argument("--no-class-weight", action="store_true", help="Disable balanced class weights")
    args = parser.parse_args()

    report = run_stage2_pipeline(
        sample_rows=args.sample_rows,
        use_class_weight=not args.no_class_weight,
        run_tuning=not args.no_tuning,
    )
    print("\n" + "=" * 60)
    print("STAGE 2 COMPLETE")
    print("=" * 60)
    print(f"Selected model: {report.selected_model}")
    print(f"Test macro F1: {report.test_metrics['macro_f1']:.4f}")
    print(f"Artifacts: {report.artifact_paths}")


if __name__ == "__main__":
    main()
