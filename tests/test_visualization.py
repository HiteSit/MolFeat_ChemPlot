"""
Tests for visualization functionality.
"""
import pytest
import numpy as np
from matplotlib.axes import Axes
from bokeh.plotting import figure
from bokeh.layouts import Column
from bokeh.models import Tabs

def test_static_scatter_plot(reduced_plotter):
    """Test static scatter plot generation"""
    # Test basic scatter plot
    ax = reduced_plotter.visualize_plot(size=10, kind='scatter')
    assert isinstance(ax, Axes)
    
    # Test with coloring
    ax = reduced_plotter.visualize_plot(size=10, kind='scatter', is_colored=True)
    assert isinstance(ax, Axes)
    
    # Test without coloring
    ax = reduced_plotter.visualize_plot(size=10, kind='scatter', is_colored=False)
    assert isinstance(ax, Axes)

def test_static_hex_plot(reduced_plotter):
    """Test static hexbin plot generation"""
    ax = reduced_plotter.visualize_plot(size=10, kind='hex')
    assert isinstance(ax, Axes)

def test_static_kde_plot(reduced_plotter):
    """Test static KDE plot generation"""
    ax = reduced_plotter.visualize_plot(size=10, kind='kde')
    assert isinstance(ax, Axes)

def test_interactive_scatter_plot(reduced_plotter):
    """Test interactive scatter plot generation"""
    # Test basic scatter plot
    fig = reduced_plotter.interactive_plot(size=500, kind='scatter')
    assert isinstance(fig, (figure, Column, Tabs))
    
    # Test with coloring
    fig = reduced_plotter.interactive_plot(size=500, kind='scatter', is_colored=True)
    assert isinstance(fig, (figure, Column, Tabs))
    
    # Test without coloring
    fig = reduced_plotter.interactive_plot(size=500, kind='scatter', is_colored=False)
    assert isinstance(fig, (figure, Column, Tabs))

def test_interactive_hex_plot(reduced_plotter):
    """Test interactive hexbin plot generation"""
    fig = reduced_plotter.interactive_plot(size=500, kind='hex')
    assert isinstance(fig, (figure, Column, Tabs))

def test_outlier_removal(reduced_plotter):
    """Test outlier removal functionality"""
    # Get plot with outliers removed
    ax = reduced_plotter.visualize_plot(size=10, remove_outliers=True)
    assert isinstance(ax, Axes)
    
    # Compare with plot without outlier removal
    ax_with_outliers = reduced_plotter.visualize_plot(size=10, remove_outliers=False)
    assert isinstance(ax_with_outliers, Axes)

def test_cluster_visualization(clustered_plotter):
    """Test cluster visualization"""
    # Test static plot with clusters
    ax = clustered_plotter.visualize_plot(size=10, clusters=True)
    assert isinstance(ax, Axes)
    
    # Test interactive plot with clusters
    fig = clustered_plotter.interactive_plot(size=500, clusters=True)
    assert isinstance(fig, (figure, Column, Tabs))

def test_invalid_plot_type(reduced_plotter):
    """Test handling of invalid plot type"""
    # Should default to scatter plot
    ax = reduced_plotter.visualize_plot(size=10, kind='invalid_type')
    assert isinstance(ax, Axes)

def test_plot_customization(reduced_plotter):
    """Test plot customization options"""
    # Test title customization
    ax = reduced_plotter.visualize_plot(size=10, title="Custom Title")
    assert isinstance(ax, Axes)
    assert ax.get_title() == "Custom Title"
    
    # Test size customization
    ax = reduced_plotter.visualize_plot(size=20)
    assert isinstance(ax, Axes)
    
    # Test interactive plot customization
    fig = reduced_plotter.interactive_plot(size=1000)
    assert isinstance(fig, (figure, Column, Tabs))

def test_classification_visualization(plotter_classification):
    """Test visualization with classification data"""
    # Reduce dimensions first
    plotter_classification.pca()
    
    # Test static plot
    ax = plotter_classification.visualize_plot(size=10)
    assert isinstance(ax, Axes)
    
    # Test interactive plot
    fig = plotter_classification.interactive_plot(size=500)
    assert isinstance(fig, (figure, Column, Tabs))

def test_regression_visualization(plotter_regression):
    """Test visualization with regression data"""
    # Reduce dimensions first
    plotter_regression.pca()
    
    # Test static plot
    ax = plotter_regression.visualize_plot(size=10)
    assert isinstance(ax, Axes)
    
    # Test interactive plot with colorbar
    fig = plotter_regression.interactive_plot(size=500)
    assert isinstance(fig, (figure, Column, Tabs)) 