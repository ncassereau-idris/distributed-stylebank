# /usr/bin/env python3
# -*- coding: utf-8 -*-

from datetime import timedelta

def format_duration(seconds):
    return str(timedelta(seconds=int(seconds)))