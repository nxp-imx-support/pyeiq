# Copyright 2020 NXP Semiconductors
# SPDX-License-Identifier: BSD-3-Clause

def load_labels(label_path):
    r"""Returns a list of labels"""
    with open(label_path, 'r') as f:
        return [line.strip() for line in f.readlines()]
