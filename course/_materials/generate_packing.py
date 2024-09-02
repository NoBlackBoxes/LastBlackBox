#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
 - Generate a packing list from the BOM
  - Specify level limt
  - Group identical items
  - Estimate package volume
"""
import os
import numpy as np
import pandas as pd

# Get user name
username = os.getlogin()

# Specify paths
repo_path = '/home/' + username + '/NoBlackBoxes/LastBlackBox'

# Specify Number of Kits
num_kits = 15

# Specify Level Limit
#level_limt = '01'
level_limt = '10'
#level_limt = '11'

# Load BOM
if level_limt == '01':
    bom_path = repo_path + "/course/_materials/BOM_01.csv"
elif level_limt == '10':
    bom_path = repo_path + "/course/_materials/BOM_10.csv"
else:
    bom_path = repo_path + "/course/_materials/BOM_11.csv"
bom = pd.read_csv(bom_path)

# Remove empty rows and label
filtered = bom[(bom['Quantity'] != 0) & (bom['Quantity'].notna())]

# Combine (aggregate) duplicates
aggregations = {
    'Part': 'first',            # First value
    'Description': 'first',     # First value
    'Quantity': 'sum',          # Sum the quantities
    'Package': 'first',         # First value
    'Supplier': 'first',        # First value
    'x(mm)': 'first',           # First value
    'y(mm)': 'first',           # First value
    'z(mm)': 'first',           # First value
}
combined = filtered.groupby('Part', as_index=False).agg(aggregations)

# Sort by packages
sorted = combined.sort_values('Package')

# Insert additional fields
sorted.insert(4, '#Kits', int(num_kits))
sorted.insert(5, '#Required', num_kits*sorted['Quantity'].astype(int))
sorted.insert(6, '#Available', 0)
sorted.insert(7, '#Order', 0)
sorted.insert(8, '#Ordered', 0)

# Save to packing file
packing_path = repo_path + f"/course/_materials/packing_{level_limt}.csv"
sorted.to_csv(packing_path, index=False)  # Set index=False if you don't want to include the index in the CSV

# Group packages
grouped = sorted.groupby('Package')
packages = {}
for name, package in grouped:
    packages[name] = package

# Estimate package volumes
total_volume = 0.0
for name in packages:
    package = packages[name]
    lengths = package['x(mm)']
    widths = package['y(mm)']
    heights = package['z(mm)']
    volume = np.sum(lengths * widths * heights)
    print(f"{name}: {volume}")
    total_volume = total_volume + volume
print(f"----------------")
print(f"Total: \t\t{total_volume:.0f} mm^3")
print(f"Available: \t{280 * 220 * 110:.0f} mm^3")

#FIN
