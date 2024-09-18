# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

block_cipher = None

# Include JSON data files
datas = [
    ('verb_labels.json', '.'),
    ('instrument_labels.json', '.'),
    ('target_labels.json', '.'),
    ('gesture_labels.json', '.'),
    ('phase_labels.json', '.'),
]

# Collect PyQt5 plugins and libraries
datas += collect_data_files('PyQt5', subdir='Qt5', includes=[
    'plugins/mediaservice/*',
    'plugins/platforms/*',
])
binaries = collect_dynamic_libs('PyQt5')

a = Analysis(
    ['main.py'],
    pathex=['.'],
    binaries=binaries,
    datas=datas,
    hiddenimports=[],
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False
)

pyz = PYZ(
    a.pure,
    a.zipped_data,
    cipher=None
)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='VideoAnnotationTool',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False
)
