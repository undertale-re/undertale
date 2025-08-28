import argparse
import os

import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="print out a summary of a given model checkpoint"
    )

    parser.add_argument("checkpoint", help="trained model checkpoint to summarize")

    parser.add_argument(
        "-s", "--shapes", action="store_true", help="print weight vector shapes"
    )
    parser.add_argument(
        "-w", "--weights", action="store_true", help="print weight vectors"
    )

    arguments = parser.parse_args()

    checkpoint = torch.load(arguments.checkpoint, map_location=torch.device("cpu"))

    print(os.path.basename(arguments.checkpoint))
    print(f"Epoch: {checkpoint.get('epoch')}")
    print("Hyperparameters:")

    for name, value in checkpoint["hyper_parameters"].items():
        print(f"  {name}: {value!r}")

    if arguments.shapes or arguments.weights:
        print("Weights:")

        for name, value in checkpoint["state_dict"].items():
            print(f"  {name}: {list(value.shape)!r}")

            if arguments.weights:
                print(f"    {value}")
