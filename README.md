# PenalizedNR

## Cox Non-proportional Hazards model with penalization
<!-- badges: start -->
<!-- badges: end -->

PenalizedNR is an R package for fitting penalized Newton's method for the time-varying effects model using mAIC, TIC, GIC as information criteria, in particular we span the parameter using basis functions. Utilities for carrying out post-fitting visualization, summarization, and inference are also provided.

## Installation

You can install the released version of PenalizedNR from [Github](https://github.com/umich-biostatistics/PenalizedNR) with:

``` r
install.packages("devtools") # you need devtools to install packages from Github
devtools::install_github("umich-biostatistics/PenalizedNR")
```

You can import the R package by:

``` r
library(PenalizedNR)
```
  

 
## Example

This tutorial simulates a data set to demonstrate the functions provided by FRprovideR.

```{r example, eval=FALSE}
# load the package
library(PenalizedNR)
```

Load a simple simulation data set with sample size of 2,000 and 2 binary predictors.

```{r example.simuate.data, eval=FALSE}
load("simulN2kOP2.RData")
```

<!-- This data is also available in the included data sets that come with the package.
To use the included data, run:
```{r, eval=FALSE}
          # raw data
          # processed data
``` -->


Now, set a sequence of smoothing paramemter to choose from. The default range is set as
```{r example.fit, eval=FALSE}
#specify smoothing parameter lambda:
lambda_spline = c(1:10)

#if not specified, the default range is set as follows, where n is the sample size
lambda_spline = seq(0, n^(1/3), length.out = 10)
```

Then set the relevant parameters and fit a model to the prepared data:
```{r example.fit, eval=FALSE}
fit <- surtvep(event = delta, z = z, time = time, 
                  lambda_spline = lambda_spline)
}
```

The default penalty term here is the Smooth-spline, but P-spline is also available. 

"nsplines" specifies the number of basis functions used and the default value is 10. 

"ties": when tied data appear, we support "Breslow" which utilizes Breslow approximation to speed up the algorithm.

"tol": the convergence threshold, smaller value requires more iterations to converge. The default value is 1e-6.

"iter.max": maximum number of iterations allowed. The default value is 20.

"method": "Newton" is the default newton's approach. "ProxN" gives the proximal newton algorithm. Refer to Wu et.al (2021)

"btr": backtracking linesearch stratgy. The default is "dynamic". "static" is also supported. Refer to 

"stop": stoppping criteria.

"parallel": When parallel is "TRUE", shared memory parallel computation is used.

"threads": integers indicating the number of threads used for parallel compuatation.

```{r example.fit, eval=FALSE}
fit <- surtvep(event = delta, z = z, time = time, 
                  lambda_spline = lambda_spline),
                  spline="Smooth-spline", nsplines=8, ties="none", 
                  tol=1e-6, iter.max=20L, method="Newton",
                  btr="dynamic", stop="ratch", 
                  parallel=TRUE, threads=3L, degree=3L)
```


Give the estimation plot based on TIC.
```{r example.fit, eval=FALSE}
plot(models, IC = "TIC")
}
```
![alt text](plots/N5000_p5_timevarying_v1_TIC_smoothcubic.png)

