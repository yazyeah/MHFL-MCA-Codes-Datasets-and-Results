@echo off
rem Portable defaults for a repository-local data directory.
rem Override any value in the calling shell when data are stored elsewhere.

set "MHFL_SUITE_ROOT=%~dp0.."
set "MHFL_PROJECT_ROOT=%MHFL_SUITE_ROOT%"
set "MHFL_DATA_ROOT=%MHFL_SUITE_ROOT%\data"
set "MHFL_UO_DATA_ROOT=%MHFL_DATA_ROOT%\3_MatLab_Raw_Data"
set "MHFL_KAIST_VIB_DIR=%MHFL_DATA_ROOT%\vibration\vibration"
set "MHFL_KAIST_CURRENT_DIR=%MHFL_DATA_ROOT%\current"
set "MHFL_MAIN_MANUSCRIPT_SOURCE_ROOT=%MHFL_SUITE_ROOT%\provenance\main_manuscript_sources"

set "MHFL_RUN_TAG=full_20260807_r1"
set "MHFL_TEMP_ROOT=%MHFL_SUITE_ROOT%\tmp\%MHFL_RUN_TAG%"
set "TEMP=%MHFL_TEMP_ROOT%"
set "TMP=%MHFL_TEMP_ROOT%"

set "MHFL_ACCEPT_KAIST_SPEC=1"
set "MHFL_CURRENT_CHANNEL_NAME=cDAQ9185-1F486B5Mod2/ai0"
set "MHFL_KAIST_VIB_COLUMN=0"
set "MHFL_UO_VIB_COLUMN=0"
set "MHFL_UO_ACOUSTIC_COLUMN=1"

set "PYTHONDONTWRITEBYTECODE=1"
