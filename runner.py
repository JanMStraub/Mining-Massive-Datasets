"""Recommendation System Runner

All you need to do is run this file with 'python3.12 main.py'.
No Colabs.
No 'Reaload all cells'.
No issues without internet connectivity.
No other extra junk that comes with the notebook format.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import dataclasses
import rec_sys.data_util as cfd

# After edits of cf_algorithms_to_complete.py:
# 1. Rename the file rec_sys.cf_algorithms_to_complete.py to rec_sys.cf_algorithms.py
# 2. Restart the runtime (Runtime -> Restart the session); possibly not needed
# 3. Swap the comments in the next two lines, so that cf_algorithms is imported as cfa
import rec_sys.cf_algorithms_to_complete as cfa
#import rec_sys.cf_algorithms as cfa
# 4. Re-run all cells
# 5. If your changes are correct, you will see a long
#    printout of recommendations for MovieLens dataset (last cell)


# Load or set the configuration
#from rec_sys.cf_config import config

@dataclasses.dataclass
class Config:
    """Holds some config vars"""
    max_rows: int = int(1e5)
    dowload_url: str = "https://files.grouplens.org/datasets/movielens/ml-25m.zip"
    download_dir: str = "./"
    unzipped_dir: str = download_dir + "ml-25m/"
    file_path: str = download_dir + "ml-25m/ratings.csv"


# Load the MovieLens and Lecture datasets
um_movielens = cfd.get_um_by_name(Config, "movielens")
um_lecture = cfd.get_um_by_name(Config, "lecture_1")

# Rate all items for the lecture toy dataset
all_ratings = cfa.rate_all_items(um_lecture, 4, 2)
print ("all_ratings lecture toy dataset:", all_ratings)

# Rate all items for the MovieLens data
all_ratings_movielens = cfa.rate_all_items(um_movielens, 0, 2)
print("all_ratings_movielens:", all_ratings_movielens)

