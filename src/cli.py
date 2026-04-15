import argparse
import sys

from src.config import CONFIG
from src.data import load_data, preprocess_data, prepare_features
from src.model import (
    get_model,
    save_model,
    load_model,
    save_results_to_json,
    load_results_from_json
)
from src.visualization import plot_lifesat_distribution

from sklearn.metrics import mean_absolute_error





def parse_arguments():
    parser = argparse.ArgumentParser(description="Life Satisfaction ML CLI")

    parser.add_argument("--file", type=str, default=CONFIG["default_csv"])
    parser.add_argument("--model-type", choices=CONFIG["available_models"], default="rf")

    parser.add_argument("--load-model", action="store_true")
    parser.add_argument("--save-model", action="store_true")

    parser.add_argument("--model-path", default=CONFIG["default_model_path"])

    parser.add_argument("--save-json", action="store_true")
    parser.add_argument("--load-json", action="store_true")
    parser.add_argument("--json-path", default=CONFIG["default_json_path"])

    parser.add_argument("--output-dir", default=CONFIG["default_output_dir"])
    parser.add_argument("--no-plot", action="store_true")

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