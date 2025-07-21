## __MLIP MD scripts__
The main script is `mlip_md.py`. It allows for setting up a MD simulation of either water (single molecule or a small bulk water box) and COSAN (void and solvated).

The MD generation process has to be performed via CLI. Check the `commands_examples.txt` and `python mlip_md.py --help` to see its usage with example commands and all the flags available.

To perform simulations, simply clone this github repository and execture the `mlip_md.py` from the same directory. The simulation data will be saved to `../simulations/<SIM_NAME>` directory. 

Happy computing!
