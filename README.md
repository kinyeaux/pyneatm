# pyneatm
Python module to calculate Near-Earth Asteroids' Thermal properties.
How to use:
the code needs several import files to work properly:
  - The "pyneatm.py" file is the module itself, which can be imported in python files.
  - The "parameters.dat" is just a text file, with certain formatting.
    This contains all the needed parameters in the order of usage.
    One can write their own parameter file, but should be careful not to change the formatting.
  - The "fits" and "obj" files contain surface grid information. These are of standard format,
    so one can use any wavefront files. The "fits" files are only for HealPix grids, so it is
    not recommended to try creating new ones.
  - "example.py" shows a script which uses the module. Since the order of steps are important
    during the calculation, it is also recommended to follow the steps given in the example file.
  - The module prints a brief log after each step in the calculation. It warnes the user if they
    skip a step in calculation, and the "<object_name>.help()" command lists all the commands in
    the prompt and a short summary of their usage.
