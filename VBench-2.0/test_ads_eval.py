"""
Test script for ads video evaluation.

Runs all 8 evaluation dimensions on the ads test videos.
No GPU needed — all VLM calls go through OpenRouter API.

Usage:
    # Run all dimensions on both ads:
    python test_ads_eval.py

    # Run specific dimensions:
    python test_ads_eval.py --dimensions storyboard_adherence,shot_specific_checks

    # Run on a specific ad only:
    python test_ads_eval.py --ad dr_doctor
    python test_ads_eval.py --ad siyi

    # Use a different model:
    python test_ads_eval.py --model anthropic/claude-sonnet-4-6

Requirements:
    - OPENROUTER_API_KEY environment variable set
    - pip install opencv-python scenedetect numpy Pillow requests
"""

import argparse
import importlib.util
import json
import os
import sys


def _import_ads_eval():
    """Import ads_eval without going through vbench2.__init__ (which requires torch)."""
    vbench2_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vbench2")

    # First register openrouter_vlm as a module so relative imports work
    for mod_name in ["openrouter_vlm", "ads_eval"]:
        spec = importlib.util.spec_from_file_location(
            f"vbench2.{mod_name}",
            os.path.join(vbench2_dir, f"{mod_name}.py"),
            submodule_search_locations=[],
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules[f"vbench2.{mod_name}"] = mod

    # Now load them
    for mod_name in ["openrouter_vlm", "ads_eval"]:
        spec = importlib.util.find_spec(f"vbench2.{mod_name}")
        spec.loader.exec_module(sys.modules[f"vbench2.{mod_name}"])

    return sys.modules["vbench2.ads_eval"]


ads_eval_mod = _import_ads_eval()
AdsEvaluator = ads_eval_mod.AdsEvaluator


# Ads config paths (relative to repo root)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ADS_CONFIGS = {
    "dr_doctor": os.path.join(REPO_ROOT, "ads_data", "DR_DOCTOR", "eval_config.json"),
    "siyi": os.path.join(REPO_ROOT, "ads_data", "SIYI", "eval_config.json"),
    # Section-based subtasks
    "siyi_a": os.path.join(REPO_ROOT, "ads_data", "SIYI", "sections", "eval_config_a.json"),
    "siyi_b": os.path.join(REPO_ROOT, "ads_data", "SIYI", "sections", "eval_config_b.json"),
    "siyi_c": os.path.join(REPO_ROOT, "ads_data", "SIYI", "sections", "eval_config_c.json"),
    "siyi_d": os.path.join(REPO_ROOT, "ads_data", "SIYI", "sections", "eval_config_d.json"),
    "dr_a": os.path.join(REPO_ROOT, "ads_data", "DR_DOCTOR", "sections", "eval_config_a.json"),
    "dr_b": os.path.join(REPO_ROOT, "ads_data", "DR_DOCTOR", "sections", "eval_config_b.json"),
    "dr_c": os.path.join(REPO_ROOT, "ads_data", "DR_DOCTOR", "sections", "eval_config_c.json"),
    "dr_d": os.path.join(REPO_ROOT, "ads_data", "DR_DOCTOR", "sections", "eval_config_d.json"),
    "dr_e": os.path.join(REPO_ROOT, "ads_data", "DR_DOCTOR", "sections", "eval_config_e.json"),
}

ALL_DIMENSIONS = [
    "storyboard_adherence",
    "shot_specific_checks",
    "character_consistency",
    "product_consistency",
    "visual_style_unity",
    "shot_continuity",
    "narrative_flow",
    "lens_atmosphere",
]


def main():
    parser = argparse.ArgumentParser(description="Ads Video Evaluation Test")
    parser.add_argument("--ad", type=str, default=None,
                        choices=list(ADS_CONFIGS.keys()),
                        help="Run on specific ad only (default: both full ads)")
    parser.add_argument("--dimensions", type=str, default=None,
                        help="Comma-separated list of dimensions to run (default: all)")
    parser.add_argument("--model", type=str, default="anthropic/claude-sonnet-4-6",
                        help="OpenRouter model to use")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file path (default: ads_eval_results.json)")
    args = parser.parse_args()

    # Check API key
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("ERROR: OPENROUTER_API_KEY environment variable not set.")
        print("Get one at https://openrouter.ai/keys")
        sys.exit(1)

    # Parse dimensions
    dimensions = None
    if args.dimensions:
        dimensions = [d.strip() for d in args.dimensions.split(",")]
        invalid = [d for d in dimensions if d not in ALL_DIMENSIONS]
        if invalid:
            print(f"ERROR: Unknown dimensions: {invalid}")
            print(f"Available: {ALL_DIMENSIONS}")
            sys.exit(1)

    # Select ads to evaluate
    if args.ad:
        configs = {args.ad: ADS_CONFIGS[args.ad]}
    else:
        configs = ADS_CONFIGS

    all_results = {}

    for ad_name, config_path in configs.items():
        print(f"\n{'#'*70}")
        print(f"#  Evaluating: {ad_name.upper()}")
        print(f"#  Config: {config_path}")
        print(f"{'#'*70}")

        if not os.path.exists(config_path):
            print(f"ERROR: Config not found: {config_path}")
            continue

        # Check video exists
        with open(config_path) as f:
            config = json.load(f)
        video_rel = config["video_path"]
        video_abs = os.path.join(os.path.dirname(config_path), os.path.basename(video_rel))
        if not os.path.exists(video_abs):
            video_abs = os.path.join(REPO_ROOT, video_rel)
        if not os.path.exists(video_abs):
            print(f"ERROR: Video not found: {video_rel}")
            print(f"  Tried: {video_abs}")
            continue

        evaluator = AdsEvaluator(config_path, model=args.model)
        results = evaluator.evaluate_all(dimensions=dimensions)
        all_results[ad_name] = results

        # Print summary
        print(f"\n{'='*60}")
        print(f"  RESULTS: {ad_name.upper()}")
        print(f"{'='*60}")
        print(f"  Overall Score: {results['overall_score']:.4f}")
        print(f"  {'Dimension':<30} {'Score':>8}")
        print(f"  {'-'*38}")
        for dim_name, dim_result in results["dimensions"].items():
            score = dim_result.get("score", "ERROR")
            if isinstance(score, float):
                print(f"  {dim_name:<30} {score:>8.4f}")
            else:
                print(f"  {dim_name:<30} {score:>8}")

    # Save results
    default_output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "debug")
    os.makedirs(default_output_dir, exist_ok=True)
    output_path = args.output or os.path.join(default_output_dir, "ads_eval_results.json")
    # Strip detailed responses to keep output manageable
    save_results = {}
    for ad_name, results in all_results.items():
        save_results[ad_name] = {
            "ad_id": results["ad_id"],
            "video_path": results["video_path"],
            "overall_score": results["overall_score"],
            "dimensions": {},
        }
        for dim_name, dim_result in results["dimensions"].items():
            save_results[ad_name]["dimensions"][dim_name] = {
                "score": dim_result.get("score"),
                "error": dim_result.get("error"),
            }
            # Include counts where available
            for key in ["passed", "total", "consistent_pairs", "total_pairs",
                        "completeness", "ordering", "coherence"]:
                if key in dim_result:
                    save_results[ad_name]["dimensions"][dim_name][key] = dim_result[key]

    with open(output_path, "w") as f:
        json.dump(save_results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
