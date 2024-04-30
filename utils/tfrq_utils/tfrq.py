import concurrent.futures
import os
from typing import Callable, List
from typing import Generator

from tqdm import tqdm

config_default_values = {"return_errors": False, "print_errors": True}


def param_list(exec_data):
    func = exec_data[0]
    chunk_id = exec_data[1]
    params = exec_data[2]
    operator = exec_data[3]
    config = exec_data[4]

    if config.get("print_off", False):
        results = []
        errors = []
        for param in params:
            try:
                if operator is None:
                    results.append(func(param))

                if operator == "*":
                    results.append(func(*param))

                if operator == "**":
                    results.append(func(**param))


            except Exception as e:
                if config["print_errors"]:
                    print(e)
                results.append(None)
                errors.append(e)
        return results, errors
    else:
        results = []
        errors = []
        for param in tqdm(params, desc=f"processing: - chunk_num[{str(chunk_id)}] pid[{str(os.getpid())}]"):
            try:
                if operator is None:
                    results.append(func(param))

                if operator == "*":
                    results.append(func(*param))

                if operator == "**":
                    results.append(func(**param))


            except Exception as e:
                if config["print_errors"]:
                    print(e)
                results.append(None)
                errors.append(e)
        return results, errors


def tfrq(func: Callable, params: List, operator=None, num_cores=None, config=None, custom_executor=None,
         is_async=False):
    import math
    from concurrent.futures import ProcessPoolExecutor
    import os

    if custom_executor:
        executor = custom_executor
    else:
        executor = ProcessPoolExecutor(max_workers=num_cores)

    if num_cores is None:
        num_cores = os.cpu_count()

    if config is None:
        config = config_default_values

    else:
        for cfg in config_default_values:
            if cfg not in config:
                config[cfg] = config_default_values[cfg]
    try:
        chunk_size = math.ceil(len(params) / num_cores)
        chunks = [params[i:i + chunk_size] for i in range(0, len(params), chunk_size)]
        print("Tfrq into", len(chunks), "Chunks for", num_cores, "cores.")
        if custom_executor:
            if is_async:
                results = list(
                    custom_executor.submit(param_list,
                                           [(func, chunk_num, chunk, operator, config) for chunk_num, chunk in
                                            enumerate(chunks)]))
            else:
                results = list(
                    custom_executor.map(param_list,
                                        [(func, chunk_num, chunk, operator, config) for chunk_num, chunk in
                                         enumerate(chunks)]))
        else:
            if is_async:
                results = list(
                    executor.submit(param_list,
                                    [(func, chunk_num, chunk, operator, config) for chunk_num, chunk in
                                     enumerate(chunks)]))
            else:
                results = list(
                    executor.map(param_list,
                                 [(func, chunk_num, chunk, operator, config) for chunk_num, chunk in
                                  enumerate(chunks)]))

        errors = []
        final_results = []
        for res in results:
            final_results.append(res[0])
            errors.append(res[1])

        final_results_flat = []
        for res_per_core in final_results:
            for res in res_per_core:
                if res is None:
                    final_results_flat.append(None)
                else:
                    final_results_flat.append(res)

        errors_flat = []
        for errs_per_core in errors:
            for err in errs_per_core:
                if err is None:
                    errors_flat.append(None)
                else:
                    errors_flat.append(err)

        if config["return_errors"]:
            return final_results_flat, errors_flat
        else:
            return final_results_flat
    finally:
        if not custom_executor and executor:
            executor.shutdown()


def tfrq_generator(func: Callable, params: List, operator=None, num_cores=None, config=None,
                   custom_executor=None, timeout=None) -> Generator:
    """
    Processes the given function (func) in parallel using multiple processes, yielding results as they are completed.

    Args:
        func (Callable): The function to be executed in parallel.
        params (List): A list of parameters to be passed to 'func'.
        operator (str, optional): Determines how arguments are passed to 'func'.
        num_cores (int, optional): Number of processor cores to use.
        config (dict, optional): Configuration options for error handling and reporting.
        custom_executor (Executor, optional): A custom executor for advanced usage.
        timeout (int, optional): Maximum number of seconds to wait for a single task before raising TimeoutError.

    Yields:
        The result of each function call as it completes. If 'config["return_errors"]' is True,
        yields a tuple (result, error), where 'result' is the output of 'func' or None if an exception occurred,
        and 'error' is the exception object or None.
    """

    num_cores = num_cores or os.cpu_count()
    executor = custom_executor or concurrent.futures.ProcessPoolExecutor(max_workers=num_cores)
    config = config or {"print_off": True}

    with tqdm(total=len(params), desc="Processing", smoothing=0.1) as pbar:
        future_to_param = {executor.submit(param_list, (func, i, [param], operator, config)): param for i, param in
                           enumerate(params)}
        for future in concurrent.futures.as_completed(future_to_param):
            try:
                result = future.result(timeout=timeout)
                yield result
                pbar.update(1)
            except concurrent.futures.TimeoutError:
                yield (None, "Task exceeded timeout")
                pbar.update(1)
            except Exception as e:
                yield (None, e)
                pbar.update(1)
    if not custom_executor:
        executor.shutdown()