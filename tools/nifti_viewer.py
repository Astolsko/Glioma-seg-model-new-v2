"""Standalone manual tool for inspecting a NIfTI image + segmentation as an
interactive 3D mesh. Not part of the training pipeline — run directly:

    python tools/nifti_viewer.py

and pick the brain image file, then the segmentation file, via the file
dialogs.
"""
import numpy as np
import plotly.graph_objects as go
from skimage import measure
import nibabel as nib
import tkinter as tk
from tkinter import filedialog


def select_file(prompt):
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title=prompt)
    return file_path


def load_nifti_file():
    brain_path = select_file("Select the brain image file")
    seg_path = select_file("Select the segmentation file")

    img = nib.load(brain_path).get_fdata()
    seg = nib.load(seg_path).get_fdata()
    return img, seg


def visualize_brain(img, seg):
    brain_parts = [
        {'img': img, 'color': 'gray', 'level': 0},
        {'img': seg, 'color': 'purple', 'level': 0},
        {'img': seg, 'color': 'red', 'level': 1},  # peritumoral edema
        {'img': seg, 'color': 'yellow', 'level': 2},  # enhancing tumor
        {'img': seg, 'color': 'blue', 'level': 3}  # tumor core
    ]
    meshes = []
    for part in brain_parts:
        verts, faces, normals, values = measure.marching_cubes(part['img'], part['level'])
        x, y, z = verts.T
        i, j, k = faces.T

        mesh = go.Mesh3d(x=x, y=y, z=z, color=part['color'], opacity=0.5, i=i, j=j, k=k)
        meshes.append(mesh)
    bfig = go.Figure(data=meshes)
    bfig.show()


if __name__ == "__main__":
    img, seg = load_nifti_file()
    visualize_brain(img, seg)
