"""Set user credentials for hf_hydrodata package."""

# pylint: disable=C0413

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
import hf_hydrodata as hf


def main():
    """Use environment variable secrets to set hf_hydrodata credentials
    for GitHub Actions testing."""
    test_email = str(os.environ["TEST_EMAIL_PUBLIC"])
    test_pin = str(os.environ["TEST_PIN_PUBLIC"])

    hf.register_api_pin(test_email, test_pin)


if __name__ == "__main__":
    main()
