import polars as pl
from sklearn.metrics import log_loss
import catboost as cb
import gc
from time import time

from dataclasses import dataclass


import GPUtil

def get_available_gpus():
    # Get the list of available GPUs using GPUtil.
    gpus = GPUtil.getGPUs()
    if not gpus:
        print("No GPUs found!")
        return []
    return gpus

def catboost_gpus():
    gpus = get_available_gpus()
    if not gpus:
        return [], []

    # Create a list of GPU IDs (as strings) for CatBoost
    gpu_ids = [int(gpu.id) for gpu in gpus]
    gpu_names = [str(gpu.name) for gpu in gpus]
    
    print("Available GPUs:")
    for gpu in gpus:
        print(f"GPU ID: {gpu.id}, Name: {gpu.name}")
    
    return gpu_ids, gpu_names


@dataclass
class FoldIDs:
    train_ids: list[int]
    valid_id: int
    test_id: int
    
    def __init__(self, test_id: int, kfolds: int) -> None:
        self.test_id = test_id
        self.valid_id = test_id-1
        self.train_ids = [i for i in range(kfolds) if i not in [test_id, self.valid_id]]

def generate_fold_ids(kfolds: int):
    
    assert kfolds % 2 == 0, "Number of folds must be even for this to work"

    res: list[FoldIDs] = []
    for i in range(kfolds-1, 0, -2):

        res.append(FoldIDs(i, kfolds))
        
    return res



def sample_train_dois(df: pl.DataFrame, seed):
    grp_top = (
        df[["time", 'doi']]
        .group_by(pl.col("time").dt.date(), maintain_order=True)
        .agg(
            train=pl.col('doi').value_counts(sort=True).top_k(100).struct.field('doi'),
            all=pl.col('doi').unique()
        )
    )
    
    grp_rnd = df["time", 'doi'].filter(
        ~pl.col("doi").is_in(grp_top["train"].explode())
    ).group_by(pl.col("time").dt.date(), maintain_order=True).agg(
        rng=pl.col('doi').unique().sample(n=5, with_replacement=True, shuffle=True, seed=seed).unique()
    )
    
    return grp_top.join(grp_rnd, on="time", how="left").with_columns(
        pl.col("train").list.concat(pl.col("rng").fill_null(pl.lit([])))
    ).with_columns(
        oos=pl.col("all").list.set_difference(pl.col("train"))
    ).drop("rng").sort("time") 
    
    
    
    
def create_folds(
    df: pl.DataFrame, 
    folds_grp: pl.DataFrame, 
    k_folds: int):   
    
    df = df.select(["doi", "time"]).filter(pl.col("doi").is_in(folds_grp["train"].explode()))
    
    idx = len(df)//k_folds 
    split_dates = []   
    for i in range(1, k_folds):
        oos_date = df["time"].sort()[idx*i].date()
        split_dates.append(oos_date)
        
    print(split_dates)
    
    print(f"{split_dates[-1]} > statisticly correct fold")
    
    folds: list[pl.Series] = []
    prev_idx = 0
    for date in split_dates:
        idx = folds_grp['time'].search_sorted(date)
        
        fold = folds_grp[prev_idx:idx, 'train'].explode()
        folds.append(fold)
        
        prev_idx = idx
        
    fold = folds_grp[prev_idx:, 'train'].explode()
    folds.append(fold)
        
    return folds
        
        

def validate_folds(df: pl.DataFrame, folds: list[pl.Series]):
    for i in range(len(folds)):
        train_set = set(pl.concat([folds[j] for j in range(len(folds)) if j != i]))
        test_set = set(folds[i])
        
        assert len(train_set & test_set) == 0, "Same Doi Leakage"
        
        print(f"Fold {i} len: {len(df.filter(pl.col('doi').is_in(test_set)))}")
        
        


class CatboostBenchmark:
    def __init__(self, data_path: str):
        
        ids, names = catboost_gpus()
                
        self.all_gpus = ids
        self.all_names = names
        
        self.data_path = data_path
        self.df = pl.read_parquet(data_path, use_pyarrow=True)
                    
        self.k = 6
        grp = sample_train_dois(self.df, seed=69)
        self.folds = create_folds(self.df, grp, self.k)
        validate_folds(self.df, self.folds)
        
                
        
    def create_datapools(self, f_ids: FoldIDs):
        
        train_dois = pl.concat([f for i, f in enumerate(self.folds) if i in f_ids.train_ids])
        valid_dois = self.folds[f_ids.valid_id]
        test_dois = self.folds[f_ids.test_id]
        
        features = self.df.columns[:-5]
        t = "target"
        w = "weight"
        
        subset = self.df.filter(pl.col("doi").is_in(train_dois)).filter(pl.col(t).is_not_null())
        
        if self.n_samples > len(subset):
            subset = subset.sample(n=self.n_samples, with_replacement=True, shuffle=True).rechunk()
        else:
            subset = subset.sample(n=self.n_samples, shuffle=True).rechunk()
        
        
        X_train = subset.select(features).cast(pl.Float32).to_numpy()
        y_train = subset[t].cast(pl.Float32).to_numpy()
        w_train = subset[w].cast(pl.Float32).to_numpy() if w else None

        print("train shape:", X_train.shape)

        subset = self.df.filter(pl.col("doi").is_in(valid_dois)).filter(pl.col(t).is_not_null())
        X_valid = subset.select(features).cast(pl.Float32).to_numpy()
        y_valid = subset[t].cast(pl.Float32).to_numpy()
        w_valid = subset[w].cast(pl.Float32).to_numpy() if w else None

        subset = self.df.filter(pl.col("doi").is_in(test_dois)).filter(pl.col(t).is_not_null())
        X_test = subset.select(features).cast(pl.Float32).to_numpy()
        y_test = subset[t].cast(pl.Float32).to_numpy()
        w_test = subset[w].cast(pl.Float32).to_numpy() if w else None
        
        train_tup = (X_train, y_train, w_train)
        valid_pool = cb.Pool(X_valid, y_valid, weight=w_valid)
        test_tup = (X_test, y_test, w_test)
        
        return train_tup, valid_pool, test_tup
        
        
    def run(self, n_gpus: int, n_samples: int):
        
        self.n_samples = n_samples
        
        assert n_gpus > 0, "We need adleast one GPU for Training!"
        
        assert len(self.all_gpus) >= n_gpus, f"{n_gpus} GPUs not available!"
        
        gpu_ids = self.all_gpus[:n_gpus]
        gpu_names = self.all_names[:n_gpus]

        print("Gpus to use:", gpu_ids)
        
        
        params = {
            "task_type": "GPU",
            "devices": gpu_ids,
            
            "logging_level": "Verbose",
            "allow_writing_files": False,
            "metric_period": 250,
            
            "loss_function": "Logloss",
            "eval_metric": "Logloss",
            
            "iterations": 250,
            "learning_rate": 0.1,
            "depth": 12,
            "l2_leaf_reg": 10,
            "random_strength": 0.7,
            "bagging_temperature": 0.85,
            "min_data_in_leaf": 30,
            "grow_policy": "Depthwise",
            
            
            "early_stopping_rounds": None,
        }
        
        if not hasattr(self, "df"):
            self.df = pl.read_parquet(self.data_path, use_pyarrow=True)

        
        for f_ids in generate_fold_ids(self.k):
            
            train_tup, valid_pool, test_tup = self.create_datapools(f_ids)
            
            del self.df
            gc.collect()
            
            model = cb.CatBoostClassifier(**params)
            
            try:
                t0 = time()
                model.fit(X=train_tup[0], y=train_tup[1], sample_weight=train_tup[2], eval_set=valid_pool)
                train_time = time() - t0
                
                n_iters = model.get_best_iteration()
                
                train_score = model.get_best_score()['learn']["Logloss"]
                valid_score = model.get_best_score()['validation']["Logloss"]
                            
                tiny_pred = model.predict_proba(test_tup[0], task_type="CPU")[:, 1]
                test_score = log_loss(test_tup[1], tiny_pred, sample_weight=test_tup[2])
            except Exception as e:
                raise e
                print(f"Exeption! {e}")
                train_time = None
                valid_score = None
                test_score = None
            
            break
        
        
        return {
            "gpu_ids": gpu_ids,
            "n_samples": self.n_samples,
            "train_time": train_time,
            "n_iters": n_iters,
            "train_score": train_score,
            "valid_score": valid_score,
            "test_score": test_score,
            "gpu_names": gpu_names,
        }
        
        
        
            
import multiprocessing
import traceback

# Assume CatboostBenchmark is defined/imported from your code base.
# from your_module import CatboostBenchmark

def run_benchmark(n_gpus, n_samples):
    """
    Run a single benchmark. If any Python-level error occurs, it will be caught.
    """
    bencher = CatboostBenchmark("dataset.parquet")
    try:
        res = bencher.run(n_gpus=n_gpus, n_samples=n_samples)
    except Exception as e:
        # Log the exception details if needed.
        print(f"Caught exception in run_benchmark: {e}")
        # Create a placeholder result dictionary.
        res = {
            'gpu_ids': list(range(n_gpus)),
            'n_samples': n_samples,
            'error': str(e)
        }
    return res

def run_in_subprocess(n_gpus, n_samples, queue):
    """
    Run the benchmark in a subprocess. Any uncatchable error (like a C++ segfault)
    will cause the process to exit with a non-zero status.
    """
    try:
        res = run_benchmark(n_gpus, n_samples)
    except Exception as e:
        # This block might not catch C++ errors causing process crashes.
        res = {
            'gpu_ids': list(range(n_gpus)),
            'n_samples': n_samples,
            'error': f"Exception in subprocess: {e}"
        }
    queue.put(res)

def main():
    results = []
    # Iterate over different numbers of GPUs and samples.
    for n_gpus in [1, 2, 4]:
        for n_samples in [1_000_000, 10_000_000, 20_000_000, 120_000_000]:
            # Create a queue to get the result back from the process.
            queue = multiprocessing.Queue()
            # Start a subprocess for this benchmark run.
            p = multiprocessing.Process(target=run_in_subprocess, args=(n_gpus, n_samples, queue))
            p.start()
            p.join()  # Wait for the subprocess to finish.
            
            # If the subprocess crashes (e.g., due to a C++ error), its exit code will be nonzero.
            if p.exitcode != 0:
                print(f"Process for n_gpus={n_gpus}, n_samples={n_samples} crashed with exit code {p.exitcode}.")
                res = {
                    'gpu_ids': list(range(n_gpus)),
                    'n_samples': n_samples,
                    'error': f"Process crashed with exit code {p.exitcode}"
                }
            else:
                # Retrieve the result from the queue.
                res = queue.get()
            
            print(res)
            results.append(res)
            # Save the log after each run.
            pl.DataFrame(results).write_json("benchmark.json")

if __name__ == "__main__":
    main()
