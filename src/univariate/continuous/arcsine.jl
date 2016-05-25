doc"""
    Arcsine(a,b)

The *Arcsine distribution* has probability density function

$f(x) = \frac{1}{\pi \sqrt{(x - a) (b - x)}}, \quad x \in [a, b]$

```julia
Arcsine()        # Arcsine distribution with support [0, 1]
Arcsine(b)       # Arcsine distribution with support [0, b]
Arcsine(a, b)    # Arcsine distribution with support [a, b]

params(d)        # Get the parameters, i.e. (a, b)
minimum(d)       # Get the lower bound, i.e. a
maximum(d)       # Get the upper bound, i.e. b
location(d)      # Get the left bound, i.e. a
scale(d)         # Get the span of the support, i.e. b - a
```

External links

* [Arcsine distribution on Wikipedia](http://en.wikipedia.org/wiki/Arcsine_distribution)

"""
immutable Arcsine{T <: Real} <: ContinuousUnivariateDistribution
    a::T
    b::T

    Arcsine(a::T, b::T) = (@check_args(Arcsine, a < b); new(a, b))
end

Arcsine{T <: Real}(a::T, b::T) = Arcsine{T}(a, b)
Arcsine(a::Real, b::Real) = Arcsine(promote(a, b)...)
Arcsine(b::Real) = (@check_args(Arcsine, b > zero(b)); Arcsine(0.0, b))
Arcsine() = Arcsine(0.0, 1.0)

@distr_support Arcsine d.a d.b

#### Conversions
function convert{T <: Real, S <: Real}(::Type{Arcsine{T}}, a::S, b::S)
    Arcsine(T(a), T(b))
end
function convert{T <: Real, S <: Real}(::Type{Arcsine{T}}, d::Arcsine{S})
    Arcsine(T(d.a), T(d.b))
end

### Parameters

params(d::Arcsine) = (d.a, d.b)
location(d::Arcsine) = d.a
scale(d::Arcsine) = d.b - d.a


### Statistics

mean(d::Arcsine) = (d.a + d.b) * 0.5
median(d::Arcsine) = mean(d)
mode(d::Arcsine) = d.a
modes(d::Arcsine) = [d.a, d.b]

var(d::Arcsine) = 0.125 * abs2(d.b - d.a)
skewness(d::Arcsine) = 0.0
kurtosis(d::Arcsine) = -1.5

entropy(d::Arcsine) = -0.24156447527049044469 + log(scale(d))


### Evaluation

pdf(d::Arcsine, x::Real) = insupport(d, x) ? one(d.a) / (π * sqrt((x - d.a) * (d.b - x))) : zero(d.a)

logpdf(d::Arcsine, x::Real) = insupport(d, x) ? -(logπ + 0.5 * log((x - d.a) * (d.b - x))) : -Inf

cdf(d::Arcsine, x::Real) = x < d.a ? 0.0 :
                              x > d.b ? 1.0 :
                              0.636619772367581343 * asin(sqrt((x - d.a) / (d.b - d.a)))

quantile(d::Arcsine, p::Real) = location(d) + abs2(sin(halfπ * p)) * scale(d)


### Sampling

rand(d::Arcsine) = quantile(d, rand())
