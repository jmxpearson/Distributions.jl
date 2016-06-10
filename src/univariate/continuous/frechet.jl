doc"""
    Frechet(α,θ)

The *Fréchet distribution* with shape `α` and scale `θ` has probability density function

$f(x; \alpha, \theta) = \frac{\alpha}{\theta} \left( \frac{x}{\theta} \right)^{-\alpha-1}
e^{-(x/\theta)^{-\alpha}}, \quad x > 0$

```julia
Frechet()        # Fréchet distribution with unit shape and unit scale, i.e. Frechet(1.0, 1.0)
Frechet(a)       # Fréchet distribution with shape a and unit scale, i.e. Frechet(a, 1.0)
Frechet(a, b)    # Fréchet distribution with shape a and scale b

params(d)        # Get the parameters, i.e. (a, b)
shape(d)         # Get the shape parameter, i.e. a
scale(d)         # Get the scale parameter, i.e. b
```

External links

* [Fréchet_distribution on Wikipedia](http://en.wikipedia.org/wiki/Fréchet_distribution)

"""
immutable Frechet{T <: Real} <: ContinuousUnivariateDistribution
    α::T
    θ::T

    function Frechet(α::T, θ::T)
    	@check_args(Frechet, α > zero(α) && θ > zero(θ))
    	new(α, θ)
    end

end

Frechet{T <: Real}(α::T, θ::T) = Frechet{T}(α, θ)
Frechet(α::Real, θ::Real) = Frechet(promote(α, θ)...)
Frechet(α::Integer, θ::Integer) = Frechet(Float64(α), Float64(θ))
Frechet(α::Real) = Frechet(α, 1.0)
Frechet() = Frechet(1.0, 1.0)

@distr_support Frechet 0.0 Inf

#### Conversions
function convert{T <: Real, S <: Real}(::Type{Frechet{T}}, α::S, θ::S)
    Frechet(T(α), T(θ))
end
function convert{T <: Real, S <: Real}(::Type{Frechet{T}}, d::Frechet{S})
    Frechet(T(d.α), T(d.θ))
end

#### Parameters

shape(d::Frechet) = d.α
scale(d::Frechet) = d.θ
params(d::Frechet) = (d.α, d.θ)


#### Statistics

function mean{T <: Real}(d::Frechet{T})
    (α = d.α; α > 1.0 ? d.θ * gamma(1.0 - 1.0 / α) : convert(T, Inf))
end

median(d::Frechet) = d.θ * logtwo^(-1.0 / d.α)

mode(d::Frechet) = (iα = -1.0/d.α; d.θ * (1.0 - iα) ^ iα)

function var{T <: Real}(d::Frechet{T})
    if d.α > 2.0
        iα = 1.0 / d.α
        return d.θ^2 * (gamma(1.0 - 2.0 * iα) - gamma(1.0 - iα)^2)
    else
        return convert(T, Inf)
    end
end

function skewness{T <: Real}(d::Frechet{T})
    if d.α > 3.0
        iα = 1.0 / d.α
        g1 = gamma(1.0 - iα)
        g2 = gamma(1.0 - 2.0 * iα)
        g3 = gamma(1.0 - 3.0 * iα)
        return (g3 - 3.0 * g2 * g1 + 2 * g1^3) / ((g2 - g1^2)^1.5)
    else
        return convert(T, Inf)
    end
end

function kurtosis{T <: Real}(d::Frechet{T})
    if d.α > 3.0
        iα = 1.0 / d.α
        g1 = gamma(1.0 - iα)
        g2 = gamma(1.0 - 2.0 * iα)
        g3 = gamma(1.0 - 3.0 * iα)
        g4 = gamma(1.0 - 4.0 * iα)
        return (g4 - 4.0 * g3 * g1 + 3 * g2^2) / ((g2 - g1^2)^2) - 6.0
    else
        return convert(T, Inf)
    end
end

function entropy(d::Frechet)
    const γ = 0.57721566490153286060  # γ is the Euler-Mascheroni constant
    1.0 + γ / d.α + γ + log(d.θ / d.α)
end


#### Evaluation

function logpdf{T <: Real}(d::Frechet{T}, x::Real)
    (α, θ) = params(d)
    if x > 0.0
        z = θ / x
        return log(α / θ) + (1.0 + α) * log(z) - z^α
    else
        return -convert(T, Inf)
    end
end

pdf(d::Frechet, x::Real) = exp(logpdf(d, x))

cdf{T <: Real}(d::Frechet{T}, x::Real) = x > 0.0 ? exp(-((d.θ / x) ^ d.α)) : zero(T)
ccdf{T <: Real}(d::Frechet{T}, x::Real) = x > 0.0 ? -expm1(-((d.θ / x) ^ d.α)) : one(T)
logcdf{T <: Real}(d::Frechet{T}, x::Real) = x > 0.0 ? -(d.θ / x) ^ d.α : -convert(T, Inf)
logccdf{T <: Real}(d::Frechet{T}, x::Real) = x > 0.0 ? log1mexp(-((d.θ / x) ^ d.α)) : zero(T)

quantile(d::Frechet, p::Real) = d.θ * (-log(p)) ^ (-1.0 / d.α)
cquantile(d::Frechet, p::Real) = d.θ * (-log1p(-p)) ^ (-1.0 / d.α)
invlogcdf(d::Frechet, lp::Real) = d.θ * (-lp)^(-1.0 / d.α)
invlogccdf(d::Frechet, lp::Real) = d.θ * (-log1mexp(lp))^(-1.0 / d.α)

function gradlogpdf{T <: Real}(d::Frechet{T}, x::Real)
    (α, θ) = params(d)
    insupport(Frechet, x) ? -(α + 1.0) / x + α * (θ^α) * x^(-α-1.0)  : zero(T)
end

## Sampling

rand(d::Frechet) = d.θ * randexp() ^ (-1.0 / d.α)
