# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt

def set_corporate_matplotlib():
    plt.rcParams.update({
        "figure.figsize": (10.5, 6.2),  # un poco más grande
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "axes.titleweight": "bold",
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "-",
        "lines.linewidth": 2.0,
        "legend.frameon": False,
        "font.size": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })

PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c",
    "#d62728", "#9467bd", "#8c564b",
    "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
]

def set_corp():
    set_corporate_matplotlib()
