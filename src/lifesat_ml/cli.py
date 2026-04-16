import argparse
import sys

from lifesat_ml.config import CONFIG
from lifesat_ml.data import load_data, preprocess_data, prepare_features
from lifesat_ml.model import (
    get_model,
    save_model,
    load_model,
    save_results_to_json,
    load_results_from_json
)
from lifesat_ml.visualization import plot_lifesat_distribution

from sklearn.metrics import mean_absolute_error





def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Life Satisfaction ML CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--file", type=str, default=CONFIG["default_csv"],
                        help="Path to input dataset")

    parser.add_argument("--model-type",
                        choices=CONFIG["available_models"],
                        default="rf",
                        help="Model to use")

    parser.add_argument("--load-model", action="store_true",
                        help="Load existing model instead of training")

    parser.add_argument("--save-model", action="store_true",
                        help="Save trained model")

    parser.add_argument("--model-path",
                        default=CONFIG["default_model_path"],
                        help="Path to save/load model")

    parser.add_argument("--save-json", action="store_true",
                        help="Save results as JSON")

    parser.add_argument("--load-json", action="store_true",
                        help="Load results from JSON")

    parser.add_argument("--json-path",
                        default=CONFIG["default_json_path"],
                        help="Path for JSON results")

    parser.add_argument("--output-dir",
                        default=CONFIG["default_output_dir"],
                        help="Directory to save plots")

    parser.add_argument("--no-plot", action="store_true",
                        help="Disable plotting")

    return parser.parse_args()

def main():
    args = parse_arguments()

    # 👉 Load from JSON only
    if args.load_json:
        results = load_results_from_json(args.json_path)
        print("Loaded results:")
        print(results)
        return

    # 👉 Load data
    df = load_data(args.file)
    df = preprocess_data(df)
    X, y = prepare_features(df)

    # 👉 Load or train model
    if args.load_model:
        model = load_model(args.model_path)
    else:
        model = get_model(args.model_type)
        model.fit(X, y)

    # 👉 Evaluate
    predictions = model.predict(X)
    mae = mean_absolute_error(y, predictions)

    results = {
        "model_type": args.model_type,
        "mae": float(mae)
    }

    print(results)

    # 👉 Save model
    if args.save_model:
        save_model(model, args.model_path)

    # 👉 Save JSON
    if args.save_json:
        save_results_to_json(results, args.json_path)

    # 👉 Plot
    if not args.no_plot:
        plot_lifesat_distribution(df, args.output_dir)





if __name__ == "__main__":
    sys.exit(main())