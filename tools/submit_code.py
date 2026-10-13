"""Submit a kernel version to the competition and print Kaggle's reason on failure.

The CLI only prints "403 Forbidden". The reason, such as an account needing identity verification,
is in the response body, which this prints.

Usage
  python tools/submit_code.py <kernel version> "<message>" [owner/submit-kernel]
"""
import sys

from kaggle.api.kaggle_api_extended import KaggleApi

COMP = "rsna-knee-abnormality-detection"
KERNEL = sys.argv[3] if len(sys.argv) > 3 else "zacheenoao/rsna-knee-submit"

api = KaggleApi()
api.authenticate()
try:
    print(api.competition_submit_code(file_name="submission.csv", message=sys.argv[2], competition=COMP,
                                      kernel=KERNEL, kernel_version=int(sys.argv[1])))
except Exception as e:
    resp = getattr(e, "response", None)
    print(f"submit failed: {e}\n{resp.text[:1000] if resp is not None else ''}")
    sys.exit(1)
