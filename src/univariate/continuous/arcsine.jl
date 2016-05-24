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

pdf(d::Arcsine, x::Float64) = insupport(d, x) ? one(d.a) / (π * sqrt((x - d.a) * (d.b - x))) : zero(d.a)

logpdf(d::Arcsine, x::Float64) = insupport(d, x) ? -(logπ + 0.5 * log((x - d.a) * (d.b - x))) : -Inf

cdf(d::Arcsine, x::Float64) = x < d.a ? 0.0 :
                              x > d.b ? 1.0 :
                              0.636619772367581343 * asin(sqrt((x - d.a) / (d.b - d.a)))

quantile(d::Arcsine, p::Float64) = location(d) + abs2(sin(halfπ * p)) * scale(d)


### Sampling

rand(d::Arcsine) = quantile(d, rand())
