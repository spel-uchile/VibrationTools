# Vibration Tools

Analysis and processing of random and quasi-static vibration tests.

- **Random Vibration Testing (RVT)**

For RVT you need to use two file: RandomTest.py and RandomEvaluation.py.

The RandomTest.py is used to create the acceleration needed for an envelope Power Spectral Density (PSD). Then, the
script save a *.rdf file to be used in Signal Generator RIGOL. The step are:

 - Run RandomTest.py 
 - Select and load the PSD file from "/psd" directory. The files are *.txt, and each file must have two column: Hz & Grms [G^2/Hz]. 
 - Select the number of segment to be used in the evaluation and correction of the random acceleration in [g]
 - Input a name for the output files.
 - Finalize

RandomEvaluation.py is used to transform the acceleration measure from the sensor to PSD. The step are:

 - Load original acceleration.
 - Load measure acceleration in *.tdms file.
 - Select the number of Sample *fsamp*
 - Run RandomEvaluation.py
 - Finalize


