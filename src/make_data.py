import argparse
import pandas as pd
from sklearn.datasets import load_iris

def main(out):
    X, y = load_iris(return_X_y=True, as_frame=True)
    df = pd.concat([X, y.rename("target")], axis=1)
    df.to_csv(out, index=False)
    print(f"saved {out} shape={df.shape}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    main(args.out)