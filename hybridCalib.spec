# -*- mode: python ; coding: utf-8 -*-

block_cipher = None


a = Analysis(['hybridCalib.py'],
             pathex=['InitValueCalc.py','RestrictionEquation.py', 'wMPSalgorithm.py', 'C:\\Users\\tju\\PycharmProjects\\wMPSHybridCalib','C:\\Users\\tju\\Anaconda3\\envs\\wMPSHybridCalibPackage\\Lib\\site-packages'],
             binaries=[],
             datas=[('C:\\Users\\tju\\Anaconda3\\envs\\wMPSHybridCalib\\Library\\bin\\libiomp5md.dll','.')],
             hiddenimports=['InitValueCalc','wMPSalgorithm',
             'RestrictionEquation','numpy','scipy','numpy.random.common', 'numpy.random.bounded_integers', 'numpy.random.entropy'],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          [],
          exclude_binaries=True,
          name='hybridCalib',
          debug=True,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=True )
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               upx_exclude=[],
               name='hybridCalib')
