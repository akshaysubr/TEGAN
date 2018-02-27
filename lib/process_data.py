import sys

from hitDataTools import get_all_datafile_info_in
from writeTFRecord import binaryToTFRecord

if len(sys.argv) < 2:
    print("Usage: ")
    print("    python {} <data directory>".format(sys.argv[0]) )
    sys.exit()

directory = sys.argv[1]

files_info = get_all_datafile_info_in(directory)
binaryToTFRecord(files_info, verbose=True)
