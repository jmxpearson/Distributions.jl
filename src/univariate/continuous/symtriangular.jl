doc"""
    SymTriangularDist(μ,σ)

The *Symmetric triangular distribution* with location `μ` and scale `σ` has probability density function

$f(x; \mu, \sigma) = \frac{1}{\sigma} \left( 1 - \left| \frac{x - \mu}{\sigma} \right| \right), \quad \mu - \sigma \le x \le \mu + \sigma$

```julia
SymTriangularDist()         # Symmetric triangular distribution with zero location and unit scale
SymTriangularDist(u)        # Symmetric triangular distribution with location u and unit scale
SymTriangularDist(u, s)     # Symmetric triangular distribution with location u and scale s

params(d)       # Get the parameters, i.e. (u, s)
location(d)     # Get the location parameter, i.e. u
scale(d)        # Get the scale parameter, i.e. s
```
"""
immutable SymTriangularDist{T <: Real} <: ContinuousUnivariateDistribution
    μ::T
    σ::T

    function SymTriangularDist(μ::T, σ::T)
        @check_args(SymTriangularDist, σ > zero(σ))
        new(μ, σ)
    end
end

SymTriangularDist{T <: Real}(μ::T, σ::T) = SymTriangularDist{T}(μ, σ)
SymTriangularDist(μ::Real, σ::Real) = SymTriangularDist(promote(μ, σ)...)
SymTriangularDist(μ::Integer, σ::Integer) = SymTriangularDist(Float64(μ), Float64(σ))
SymTriangularDist(μ::Real) = SymTriangularDist(μ, 1.0)
SymTriangularDist() = SymTriangularDist(0.0, 1.0)

@distr_support SymTriangularDist d.μ - d.σ d.μ + d.σ

#### Conversions

function convert{T <: Real}(::Type{SymTriangularDist{T}}, μ::Real, σ::Real)
    SymTriangularDist(T(μ), T(σ))
end
function convert{T <: Real, S <: Real}(::Type{SymTriangularDist{T}}, d::SymTriangularDist{S})
    SymTriangularDist(T(d.μ), T(d.σ))
end

#### Parameters

location(d::SymTriangularDist) = d.μ
scale(d::SymTriangularDist) = d.σ

params(d::SymTriangularDist) = (d.μ, d.σ)


#### Statistics

mean(d::SymTriangularDist) = d.μ
median(d::SymTriangularDist) = d.μ
mode(d::SymTriangularDist) = d.μ

var(d::SymTriangularDist) = d.σ^2 / 6.0
skewness{T <: Real}(d::SymTriangularDist{T}) = zero(T)
kurtosis{T <: Real}(d::SymTriangularDist{T}) = -0.6*one(T)

entropy(d::SymTriangularDist) = 0.5 + log(d.σ)


#### Evaluation

zval(d::SymTriangularDist, x::Real) = (x - d.μ) / d.σ
xval(d::SymTriangularDist, z::Real) = d.μ + z * d.σ


pdf{T <: Real}(d::SymTriangularDist{T}, x::Real) = insupport(d, x) ? (1.0 - abs(zval(d, x))) / scale(d) : zero(T)

function logpdf{T <: Real}(d::SymTriangularDist{T}, x::Real)
    insupport(d, x) ? log((1.0 - abs(zval(d, x))) / scale(d)) : -convert(T, convert(T, Inf))
end

function cdf{T <: Real}(d::SymTriangularDist{T}, x::Real)
    (μ, σ) = params(d)
    x <= μ - σ ? zero(T) :
    x <= μ ? 0.5 * (1.0 + zval(d, x))^2 :
    x < μ + σ ? 1.0 - 0.5 * (1.0 - zval(d, x))^2 : one(T)
end

function ccdf{T <: Real}(d::SymTriangularDist{T}, x::Real)
    (μ, σ) = params(d)
    x <= μ - σ ? one(T) :
    x <= μ ? 1.0 - 0.5 * (1.0 + zval(d, x))^2 :
    x < μ + σ ? 0.5 * (1.0 - zval(d, x))^2 : zero(T)
end

function logcdf{T <: Real}(d::SymTriangularDist{T}, x::Real)
    (μ, σ) = params(d)
    x <= μ - σ ? -convert(T, Inf) :
    x <= μ ? loghalf + 2.0 * log1p(zval(d, x)) :
    x < μ + σ ? log1p(-0.5 * (1.0 - zval(d, x))^2) : zero(T)
end

function logccdf{T <: Real}(d::SymTriangularDist{T}, x::Real)
    (μ, σ) = params(d)
    x <= μ - σ ? zero(T) :
    x <= μ ? log1p(-0.5 * (1.0 + zval(d, x))^2) :
    x < μ + σ ? loghalf + 2.0 * log1p(-zval(d, x)) : -convert(T, Inf)
end

quantile(d::SymTriangularDist, p::Real) = p < 0.5 ? xval(d, sqrt(2.0 * p) - 1.0) :
                                                       xval(d, 1.0 - sqrt(2.0 * (1.0 - p)))

cquantile(d::SymTriangularDist, p::Real) = p > 0.5 ? xval(d, sqrt(2.0 * (1.0-p)) - 1.0) :
                                                        xval(d, 1.0 - sqrt(2.0 * p))

invlogcdf(d::SymTriangularDist, lp::Real) = lp < loghalf ? xval(d, expm1(0.5*(lp - loghalf))) :
                                                              xval(d, 1.0 - sqrt(-2.0 * expm1(lp)))

invlogccdf(d::SymTriangularDist, lp::Real) = lp > loghalf ? xval(d, sqrt(-2.0 * expm1(lp)) - 1.0) :
                                                               xval(d, -(expm1(0.5 * (lp - loghalf))))


function mgf(d::SymTriangularDist, t::Real)
    (μ, σ) = params(d)
    a = σ * t
    a == zero(a) && return one(a)
    4.0 * exp(μ * t) * (sinh(0.5 * a) / a)^2
end

function cf(d::SymTriangularDist, t::Real)
    (μ, σ) = params(d)
    a = σ * t
    a == zero(a) && return complex(one(a))
    4.0 * cis(μ * t) * (sin(0.5 * a) / a)^2
end


#### Sampling

rand(d::SymTriangularDist) = xval(d, rand() - rand())
