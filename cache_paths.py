#!/usr/bin/env python3
import sys
from pathlib import Path
from data import cache_all

cache_all(Path(sys.argv[1]))
