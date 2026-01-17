import time
import gc
import cupy as cp
import cudf

from convert_to_parquet import convert_to_parquet
from add_prec_days import get_precipitation_days
from add_prec_temp import add_precipitation_temperature
from finalize import finalize
from process import clean_data


def clear_memory():
    gc.collect()
    try:
        cp.get_default_memory_pool().free_all_blocks()
        cp.cuda.runtime.deviceSynchronize()
    except Exception:
        pass  # in case GPU is not used


class Pipeline:
    def __init__(self, crop: str, crop_type: str, parquet: bool = True):
        """
        Parameters
        ----------
        crop : str
            Crop name, e.g. "wheat".
        crop_type : {"winter", "spring"}
            Crop type.
        parquet : bool, optional
            Whether to first convert input data to parquet.
        """
        t_start = time.perf_counter()
        if crop_type not in {"winter", "spring"}:
            raise ValueError("crop_type must be 'winter' or 'spring'")

        self.crop = crop
        self.crop_type = crop_type

        # Step 1: convert to parquet
        if not parquet:
            t0 = time.perf_counter()
            print("File is CSV, now converting to PARQUET")
            convert_to_parquet(self.crop)
            t1 = time.perf_counter()
            print(f"[{self.crop}] convert_to_parquet took {t1 - t0:.2f} seconds")
            clear_memory()

        print("File is already PARQUET. Skipped conversion to PARQUET")

        # Step 2: clean data
        t0 = time.perf_counter()
        clean_data(self.crop)
        t1 = time.perf_counter()
        print(f"[{self.crop}] data cleaning took {t1 - t0:.2f} seconds")
        clear_memory()
        

        # Step 3: precipitation days
        t0 = time.perf_counter()
        get_precipitation_days()
        t1 = time.perf_counter()
        print(f"[{self.crop}] get_precipitation_days took {t1 - t0:.2f} seconds")
        clear_memory()

        # Step 4: precipitation + temperature
        t0 = time.perf_counter()
        add_precipitation_temperature(self.crop, self.crop_type)
        t1 = time.perf_counter()
        print(
            f"[{self.crop}] add_precipitation_temperature "
            f"({self.crop_type}) took {t1 - t0:.2f} seconds"
        )
        clear_memory()

        # Step 5: finalize
        t0 = time.perf_counter()
        finalize(self.crop)
        t1 = time.perf_counter()
        print(f"[{self.crop}] finalize took {t1 - t0:.2f} seconds")

        print(f"All Took {t1 - t_start:.2f} seconds")


if __name__ == "__main__":
    pipeline = Pipeline(crop="WBAR", crop_type="winter", parquet=False)






    
