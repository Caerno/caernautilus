# caernautilus
 Caerno utilities

## output.py
* Printing functions
  - informer & informer_print
    - analysis of initial dataset with target feature
  - imperfection
    - missing values, dtypes & value counts
  - multicolumn
    - viewing long data series or df's with one column
* Plotting functions 
  - plot_some_scatters
    - multiple column-pairwise scatterplots
  - plot_conf_map
    - plot confusion matrix and coefficients
* Images transforms
  - img_breakdown + img_set_up / img_breakright + img_set_left
    - dimensionality reduction of color image data by concatenating sideways or downwards
  - img_squeeze
    - image compression and return to original format
  - img_no_ticks & img_framaker
    - images display

## input.py
* Input functions
  - number
    - get a matching number from the user

## classes.py
* SlowPolyLinearReg
  - regression class with poly-feature adding & L2-regularization
* NanFixer
  - nan-filler encoder
* Digitalize
  - automatic representation of categorical data with numbers
