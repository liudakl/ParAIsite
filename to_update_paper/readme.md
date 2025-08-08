# Meaning of the data here

In the folder **data_scan** are results of the scan for my best models over 3 steps; 

File **variances_paraisite.csv** has all the information about the variances, mean, max values of predited TC. File was produced by runnig code **main/src/visualisation_for_paper.py**.

Plots (used the same file  **main/src/visualisation_for_paper.py**): 

- *variances_step_all_TC* :  box plots of variances across the whole mpd for mean predicted TC values; 
- *variances_step_lowTC* : the same as above but for predicted TC values less than 10; 
- *variances_step_lowTC_1* : the same as above but for predicted TC values less than 3; 

- *variances_TC* : I wanted to group the results of variances across the whole mpd for  predicted TC values to see the patterns (a bit different representation); 
- *variances_TC_low_1a* and *variances_TC_low_1b*, or *variances_TC_low_2a* and *variances_TC_low_2b* are same, but different representation. 
