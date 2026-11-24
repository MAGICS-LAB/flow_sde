from pathlib import Path

from models.classifier_utils import ensure_reward_classifier, freeze_classifier
from models.utils import ensure_dir, get_device, load_json_config, parse_common_args, save_run_metadata, seed_everything


def main() -> None:
    parser = parse_common_args("Prepare or train the reward classifier.")
    args = parser.parse_args()

    config = load_json_config(args.config)
    if args.output_dir is not None:
        config["output_dir"] = args.output_dir

    output_dir = ensure_dir(config["output_dir"])
    device = get_device(args.device or config.get("device"))
    seed_everything(config["seed"])
    save_run_metadata(Path(output_dir), config=config, device=device, seed=config["seed"])

    classifier, report = ensure_reward_classifier(config, device=device)
    freeze_classifier(classifier)
    print(f"Prepared classifier on {device}: {report['mode']}")


if __name__ == "__main__":
    main()
