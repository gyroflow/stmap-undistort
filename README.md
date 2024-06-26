# Deprecated!
Latest build of Gyroflow supports exporting STMaps directly from the app

<hr>

## stmap-undistort
Generate stmaps for distortion correction using the Gyroflow lens profile database


### Usage:

Install dependencies:

* imageio: `pip install imageio[pyav]`
* numpy: `pip install numpy`

Run with `python convert.py`, select the Gyroflow json lens profile (available from https://github.com/gyroflow/gyroflow ) and a location to save the .exr file.
