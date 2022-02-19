# PenalizedNR

## Cox Non-proportional Hazards model with penalization
<!-- badges: start -->
<!-- badges: end -->

A penalized Newton's method for the time-varying effects model.

## Installation

You can install the released version of PenalizedNR from [Github](https://github.com/umich-biostatistics/PenalizedNR) with:

``` r
install.packages("devtools") # you need devtools to install packages from Github
devtools::install_github("umich-biostatistics/PenalizedNR")
```

You can install directly from CRAN with:

``` r
install.packages("PenalizedNR")
```

## Example

This tutorial simulates a data set to demonstrate the functions provided by FRprovideR.

```{r example, eval=FALSE}
# load the package
library(PenalizedNR)
# other imports
library(mvtnorm)
library(splines)
library(survival)
```

Load a simple simulation data set with sample size of 2,000.

```{r example.simuate.data, eval=FALSE}
load("simulN2kOP2.RData")
```

<!-- This data is also available in the included data sets that come with the package.
To use the included data, run:
```{r, eval=FALSE}
          # raw data
          # processed data
``` -->


Now, set a sequence of smoothing paramemter to choose from. 
```{r example.fit, eval=FALSE}
#specify smoothing parameter lambda:
lambda_spline = c(1:10)
```

Then set the relevant parameters and fit a model to the prepared data:
```{r example.fit, eval=FALSE}
models <- surtvep(event = delta, z = z, time = time, 
                  lambda_spline = lambda_spline,
                  spline="Smooth-spline", nsplines=8, ties="none", 
                  tol=1e-6, iter.max=20L, method="Newton",
                  btr="dynamic", stop="ratch", 
                  parallel=TRUE, threads=3L, degree=3L)
}
```

![alt text](plots/N5000_p5_timevarying_v1_TIC_smoothcubic.png)
