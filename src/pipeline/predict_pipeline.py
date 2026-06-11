import os
import sys
import joblib
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

from src.logger import logging
from src.exception import CustomException
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformationConfig, DataTransformation
from src.utils import compute_ge, AES_SBOX

CORRECT_KEY_BYTE = 0xe0
TARGET_BYTE      = 2


@dataclass
class PredictPipelineConfig:
    strategy: str   = "snr"
    k: int          = 50
    model_name: str = "xgboost"
    n_traces_range: List[int] = None
    n_experiments: int = 100

    def __post_init__(self):
        if self.n_traces_range is None:
            self.n_traces_range = [10, 25, 50, 100, 200, 500]

    @property
    def tag(self) -> str:
        return f"{self.strategy}_k{self.k}"

    def model_path(self) -> str:
        return os.path.join(
            "artifacts", "models",
            f"{self.model_name}_{self.tag}.joblib"
        )

    def transformer_path(self) -> str:
        # PCA transformers are saved by n_components (transformer_pca_n{k}),
        # matching DataTransformationConfig and app.py; snr/anova use the k tag.
        transformer_tag = f"pca_n{self.k}" if self.strategy == "pca" else self.tag
        return os.path.join(
            "artifacts", "models",
            f"transformer_{transformer_tag}.joblib"
        )


class PredictPipeline:
    def __init__(self, config: PredictPipelineConfig):
        self.config = config
        self._model       = None
        self._transformer = None

    def _load_artifacts(self) -> None:
        model_path = self.config.model_path()
        transformer_path = self.config.transformer_path()

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found: {model_path}. Run train_pipeline first."
            )
        if not os.path.exists(transformer_path):
            raise FileNotFoundError(
                f"Transformer not found: {transformer_path}. Run train_pipeline first."
            )

        self._model       = joblib.load(model_path)
        self._transformer = joblib.load(transformer_path)
        logging.info(
            f"Loaded model: {self.config.model_name} | "
            f"strategy: {self.config.strategy} | k: {self.config.k}"
        )

    def predict_single(self, trace: np.ndarray) -> Tuple[int, np.ndarray]:
        """
        Predict HW class and class probabilities for a single trace.

        Args:
            trace: raw power trace, shape (n_samples,)

        Returns:
            hw_pred : predicted Hamming Weight class (0–8)
            probs   : class probability array, shape (9,)
        """
        if self._model is None:
            self._load_artifacts()
        try:
            trace_t = self._transformer.transform(trace.reshape(1, -1))
            probs   = self._model.predict_proba(trace_t)[0]
            hw_pred = int(np.argmax(probs))
            return hw_pred, probs
        except Exception as e:
            raise CustomException(e, sys)

    def run_ge_evaluation(
        self,
        X_atk: np.ndarray,
        pt_atk: np.ndarray,
    ) -> Tuple[List[float], int]:
        """
        Run full GE evaluation on attack traces.

        Args:
            X_atk  : raw attack traces, shape (n_atk, n_samples)
            pt_atk : plaintext bytes, shape (n_atk, 16)

        Returns:
            ge_values : GE at each trace count in config.n_traces_range
            ntd       : Number of Traces to Disclosure (first n where GE < 1)
        """
        if self._model is None:
            self._load_artifacts()
        try:
            logging.info("Transforming attack traces for GE evaluation")
            X_atk_t = self._transformer.transform(X_atk)
            probs   = self._model.predict_proba(X_atk_t)

            logging.info(
                f"Running GE — {self.config.n_experiments} experiments × "
                f"{len(self.config.n_traces_range)} trace counts"
            )
            ge_values = compute_ge(
                probs, pt_atk, CORRECT_KEY_BYTE,
                n_traces_range=self.config.n_traces_range,
                n_experiments=self.config.n_experiments
            )

            # Number of Traces to Disclosure
            ntd = None
            for n, ge in zip(self.config.n_traces_range, ge_values):
                if ge < 1.0:
                    ntd = n
                    break

            for n, ge in zip(self.config.n_traces_range, ge_values):
                logging.info(f"  GE @ {n:>4} traces = {ge:.2f}")

            if ntd:
                logging.info(f"NtD (GE<1): {ntd} traces")
            else:
                logging.info("NtD not reached within tested range")

            return ge_values, ntd

        except Exception as e:
            raise CustomException(e, sys)


def run_predict_pipeline(
    strategy: str   = "snr",
    k: int          = 50,
    model_name: str = "xgboost",
    n_experiments: int = 100,
) -> Tuple[List[float], int]:
    """
    Load attack data, run GE evaluation, return results.
    Convenience wrapper for notebooks and app.py.
    """
    try:
        logging.info(
            f"=== Predict pipeline START | "
            f"model={model_name} strategy={strategy} k={k} ==="
        )

        di = DataIngestion()
        _, _, _, X_atk, _, pt_atk = di.initiate_data_ingestion()

        cfg = PredictPipelineConfig(
            strategy=strategy, k=k,
            model_name=model_name,
            n_experiments=n_experiments
        )
        pipeline  = PredictPipeline(cfg)
        ge_values, ntd = pipeline.run_ge_evaluation(X_atk, pt_atk)

        logging.info("=== Predict pipeline COMPLETE ===")
        return ge_values, ntd

    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run GE evaluation")
    parser.add_argument("--strategy",  type=str, default="snr",
                        choices=["snr", "anova", "pca"])
    parser.add_argument("--k",         type=int, default=50)
    parser.add_argument("--model",     type=str, default="xgboost")
    parser.add_argument("--n_exp",     type=int, default=100)
    args = parser.parse_args()

    ge_values, ntd = run_predict_pipeline(
        strategy=args.strategy, k=args.k,
        model_name=args.model, n_experiments=args.n_exp
    )
    print(f"\nGE values: {ge_values}")
    print(f"NtD: {ntd}")