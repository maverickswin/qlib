#  Licensed under the MIT License.
"""
This script demonstrates how to train a Transformer model for stock prediction using QLib.
It is based on the workflow_by_code.py example but uses TransformerModel instead.
"""
import qlib
from qlib.constant import REG_CN
from qlib.utils import init_instance_by_config, flatten_dict
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord, SigAnaRecord
from qlib.tests.data import GetData


if __name__ == "__main__":
    # Initialize QLib
    provider_uri = "~/.qlib/qlib_data/cn_data"  # target directory for data
    GetData().qlib_data(target_dir=provider_uri, region=REG_CN, exists_skip=True)
    qlib.init(provider_uri=provider_uri, region=REG_CN)

    # Define the model configuration
    model_config = {
        "class": "TransformerModel",
        "module_path": "qlib.contrib.model.pytorch_transformer_ts",
        "kwargs": {
            "d_feat": 20,          # Number of features
            "d_model": 64,         # Dimension of the model
            "nhead": 2,            # Number of attention heads
            "num_layers": 2,       # Number of transformer layers
            "dropout": 0.0,        # Dropout rate
            "n_epochs": 100,       # Number of epochs
            "lr": 0.0001,          # Learning rate
            "early_stop": 5,       # Early stopping rounds
            "loss": "mse",         # Loss function
            "optimizer": "adam",   # Optimizer
            "reg": 1e-3,           # Regularization strength
            "batch_size": 8192,    # Batch size
            "n_jobs": 10,          # Number of parallel jobs
            "GPU": 0,              # GPU ID to use (negative for CPU)
            "seed": 0,             # Random seed for reproducibility
        }
    }

    # Define the data handler configuration
    data_handler_config = {
        "start_time": "2008-01-01",
        "end_time": "2020-08-01",
        "fit_start_time": "2008-01-01",
        "fit_end_time": "2014-12-31",
        "instruments": "csi300",
        "infer_processors": [
            {
                "class": "FilterCol",
                "kwargs": {
                    "fields_group": "feature",
                    "col_list": ["RESI5", "WVMA5", "RSQR5", "KLEN", "RSQR10", "CORR5", "CORD5", "CORR10", 
                                "ROC60", "RESI10", "VSTD5", "RSQR60", "CORR60", "WVMA60", "STD5", 
                                "RSQR20", "CORD60", "CORD10", "CORR20", "KLOW"]
                }
            },
            {
                "class": "RobustZScoreNorm",
                "kwargs": {
                    "fields_group": "feature",
                    "clip_outlier": True
                }
            },
            {
                "class": "Fillna",
                "kwargs": {
                    "fields_group": "feature"
                }
            }
        ],
        "learn_processors": [
            {
                "class": "DropnaLabel"
            },
            {
                "class": "CSRankNorm",
                "kwargs": {
                    "fields_group": "label"
                }
            }
        ],
        "label": ["Ref($close, -2) / Ref($close, -1) - 1"]
    }

    # Define the dataset configuration
    dataset_config = {
        "class": "TSDatasetH",
        "module_path": "qlib.data.dataset",
        "kwargs": {
            "handler": {
                "class": "Alpha158",
                "module_path": "qlib.contrib.data.handler",
                "kwargs": data_handler_config
            },
            "segments": {
                "train": ("2008-01-01", "2014-12-31"),
                "valid": ("2015-01-01", "2016-12-31"),
                "test": ("2017-01-01", "2020-08-01")
            },
            "step_len": 20  # Time steps for sequence input
        }
    }

    # Initialize model and dataset
    model = init_instance_by_config(model_config)
    dataset = init_instance_by_config(dataset_config)

    # Define portfolio analysis configuration
    port_analysis_config = {
        "executor": {
            "class": "SimulatorExecutor",
            "module_path": "qlib.backtest.executor",
            "kwargs": {
                "time_per_step": "day",
                "generate_portfolio_metrics": True,
            },
        },
        "strategy": {
            "class": "TopkDropoutStrategy",
            "module_path": "qlib.contrib.strategy",
            "kwargs": {
                "signal": (model, dataset),
                "topk": 50,
                "n_drop": 5,
            },
        },
        "backtest": {
            "start_time": "2017-01-01",
            "end_time": "2020-08-01",
            "account": 100000000,
            "benchmark": "SH000300",
            "exchange_kwargs": {
                "freq": "day",
                "limit_threshold": 0.095,
                "deal_price": "close",
                "open_cost": 0.0005,
                "close_cost": 0.0015,
                "min_cost": 5,
            },
        },
    }

    # Optional: Preview the data
    example_df = dataset.prepare("train")
    # print("Dataset preview:")
    # print(example_df.head())

    # Start the experiment
    with R.start(experiment_name="transformer_stock_prediction"):
        # Log parameters
        R.log_params(**flatten_dict({"model": model_config, "dataset": dataset_config}))
        
        # Train the model
        print("Training the Transformer model...")
        model.fit(dataset)
        
        # Save the trained model
        R.save_objects(**{"params.pkl": model})

        # Generate predictions
        recorder = R.get_recorder()
        sr = SignalRecord(model, dataset, recorder)
        sr.generate()

        # Analyze signals
        sar = SigAnaRecord(recorder)
        sar.generate()

        # Backtest and analyze portfolio performance
        print("Running backtest and portfolio analysis...")
        par = PortAnaRecord(recorder, port_analysis_config, "day")
        par.generate()
        
        print("Workflow completed successfully!")