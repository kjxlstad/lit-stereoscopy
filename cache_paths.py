#!/usr/bin/env python3
import sys
import logging
from pathlib import Path
from data import cache_all
logging.basicConfig(level=logging.INFO)

cache_all(Path(sys.argv[1]))
