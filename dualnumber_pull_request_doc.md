

## Completed Distributions
"*" means with caveats, notes below

**Continuous**

 - arcsine
 - beta
 - betaprime
 - biweight
 - cauchy
 - cosine
 - epanechnikov
 - erlang
 - exponential
 - fdist
 - frechet
 - gamma
 - generalized extreme value *
 - generalized pareto *
 - gumbel
 - inverse gaussian
 - kolmogorov *
 - ksdist *
 - ksonesided *
 - laplace
 - logistic
 - log normal
 - noncentral beta *
 - noncentral chisq
 - normal
 - normal canon
 - normal inverse gaussian
 - pareto
 - rayleigh
 - symtriangular
 - triweight
 - uniform
 - weibull
 
 
**Discrete**

- bernoulli
- binomial
- categorical
- discrete uniform *
- geometric *
- hypergeometric *
- negative binomial
- wallenius noncentral hypergeometric
- poisson


## Incomplete Distributions
Type instabilities remain

**Continuous**

 - chi
 - chisq
 - inverse gamma
 - levy
 - noncentralf
 - noncentralt
 - tdist
 - triangular
 - vonmises


**Discrete**
 
 - beta binomial
 - fisher noncentral hypergeometric
 - poisson binomial
 - skellam


## Issues

 **Overview**

The issue that currently affect the performance of this pull request is type instability when calling a distribution function.  Most type instabilities are caused by functions outside of the distributions library that do not handle dual numbers well.  Many of them invoke ccalls, which require floating point arguments.


**Rmath (part of StatsFuns pkg)**

The rand (and invcdf for chisq) function for certain distributions calls a function in the package StatsFuns (the file StatsFuns/src/rmath.jl, line 88), which refers to a ccall.

Distributions affected:

 - chi
 - chisq
 - noncentralf
 - noncentralt

**StatsFuns**

Functions in StatsFuns package cast arguments to the type Float64, which would cause errors with dual numbers.  These functions then invoke ccalls.

Distributions affected:

- fisher noncentral hypergeometric (xlogy at StatsFuns/src/basefuns.jl)
- There should be other ones, but I didn't keep track of all of them because I changed most of the functions in basefuns.jl to accept dual numbers

**Base**

Functions in base package cast arguments to the type Float64, which would cause errors with dual numbers.  These functions then invoke ccalls.

Distributions affected:

- tdist (besselk at base/special/bessel.jl)
- vonmises (besseli at base/special/bessel.jl)
- betabinomial (beta at base/special/gamma.jl)
- fisher noncentral hypergeometric (lbeta at base/special/gamma.jl)
- poisson binomial (atan2 at base/math.jl)
- skellam (besseli at base/special/bessel.jl)

**sqrt function**

Stack overflow when taking square root of dual number times im

Distributions affected:

- inverse gamma (mgf)
- levy (cf)

**Miscellaneous**

 - Skipped type instabilites that trace back to quantile_newton function
	 - Distributions affected:
		 - biweight
		 - epanechnikov
		 - triweight
 - cis function in triangular distribution creates possibility of imaginary numbers, resulting in type instability
 - Instability at rand for generalizedextremevalue and generalizedpareto
 - Skipped noncentralbeta because it seemed to be incomplete


## Caveats

- Many distributions have either no parameters or only discrete parameters.  These were skipped.  They include:
	- kolmogorov
	- ksdist
	- konesided
	- discreteuniform
	- hypergeometric
- Some distributions did not have functions that were easy to differentiate, although dual numbers were still tested on them.  They include:
	- vonmises (only kappa parameter)
	- fisher noncentral hypergeometric
	- wallenius noncentral hypergeometric


> Written with [StackEdit](https://stackedit.io/).