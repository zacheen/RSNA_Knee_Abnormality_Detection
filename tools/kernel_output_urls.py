"""Print "<file name> <signed url>" for each output file of a kernel whose name matches a regex.

`kaggle kernels output` reads a whole file into memory before writing it and cannot resume. A 10 GB cache
zip was lost at 7.5 GB that way, so tools/fetch_cache.sh hands these URLs to `curl -C -` instead.

Usage
  python tools/kernel_output_urls.py zacheenoao/rsna-knee-build-cache-v2-s0 "cache_train.*\\.zip"
"""
import re
import sys

from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest

owner, slug = sys.argv[1].split("/")
pattern = re.compile(sys.argv[2])
api = KaggleApi()
api.authenticate()
with api.build_kaggle_client() as kaggle:
    req = ApiListKernelSessionOutputRequest()
    req.user_name, req.kernel_slug = owner, slug
    resp = kaggle.kernels.kernels_api_client.list_kernel_session_output(req)
for item in resp.files or []:
    if pattern.search(item.file_name):
        print(item.file_name, item.url)
