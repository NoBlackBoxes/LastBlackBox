# -*- coding: utf-8 -*-
"""
LBB: Packaging : Utility Library

@author: kampff
"""

# Import libraries
import cadquery as cq

# Import modules

# Save named STEP file
def save_STEP(step_path, step_name, model):
    model.val().label = step_name
    cq.exporters.export(model, step_path, exportType=cq.exporters.ExportTypes.STEP)
    with open(step_path, 'r') as file:
        content = file.read()
        content = content.replace("Open CASCADE STEP translator ", step_name)
    with open(step_path, 'w') as file:
        file.write(content)
    return

#FIN