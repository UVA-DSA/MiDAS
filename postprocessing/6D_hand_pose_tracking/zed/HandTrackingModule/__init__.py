"""Reusable building blocks for ZED-based 6D hand pose tracking.

Import the submodules directly so that optional dependencies are only pulled
in when they are actually used (``Vis3D`` needs Open3D, for example)::

    from HandTrackingModule.Zed import Zed
    from HandTrackingModule.HandTracking import HandTracking
    from HandTrackingModule.Vis3D import Vis3D
"""
