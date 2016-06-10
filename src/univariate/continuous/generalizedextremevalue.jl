doc"""
    GeneralizedExtremeValue(μ, σ, ξ)

The *Generalized extreme value distribution* with shape parameter `ξ`, scale `σ` and location `μ` has probability density function

$f(x; \xi, \sigma, \mu) = \begin{cases}
        \frac{1}{\sigma} \left[ 1+\left(\frac{x-\mu}{\sigma}\right)\xi\right]^{-1/\xi-1} \exp\left\{-\left[ 1+ \left(\frac{x-\mu}{\sigma}\right)\xi\right]^{-1/\xi} \right\} & \text{for } \xi \neq 0 \\
        \frac{1}{\sigma} \exp\left\{-\frac{x-\mu}{\sigma}\right\} \exp\left\{-\exp\left[-\frac{x-\mu}{\sigma}\right]\right\} & \text{for } \xi = 0
    \end{cases}$

for

$x \in \begin{cases}
        \left[ \mu - \frac{\sigma}{\xi}, + \infty \right) & \text{for } \xi > 0 \\
        \left( - \infty, + \infty \right) & \text{for } \xi = 0 \\
        \left( - \infty, \mu - \frac{\sigma}{\xi} \right] & \text{for } \xi < 0
    \end{cases}$

```julia
GeneralizedExtremeValue(k, s, m)      # Generalized Pareto distribution with shape k, scale s and location m.

params(d)       # Get the parameters, i.e. (k, s, m)
shape(d)        # Get the shape parameter, i.e. k (sometimes called c)
scale(d)        # Get the scale parameter, i.e. s
location(d)     # Get the location parameter, i.e. m
```

External links

* [Generalized extreme value distribution on Wikipedia](https://en.wikipedia.org/wiki/Generalized_extreme_value_distribution)

"""

immutable GeneralizedExtremeValue{T <: Real} <: ContinuousUnivariateDistribution
    μ::T
    σ::T
    ξ::T

    function GeneralizedExtremeValue(μ::T, σ::T, ξ::T)
        σ > zero(σ) || error("Scale must be positive")
        new(μ, σ, ξ)
    end
end

GeneralizedExtremeValue{T <: Real}(μ::T, σ::T, ξ::T) = GeneralizedExtremeValue{T}(μ, σ, ξ)
GeneralizedExtremeValue(μ::Real, σ::Real, ξ::Real) = GeneralizedExtremeValue(promote(μ, σ, ξ)...)
function GeneralizedExtremeValue(μ::Integer, σ::Integer, ξ::Integer)
    GeneralizedExtremeValue(Float64(μ), Float64(σ), Float64(ξ))
end

#### Conversions
function convert{T <: Real}(::Type{GeneralizedExtremeValue{T}}, μ::Real, σ::Real, ξ::Real)
    GeneralizedExtremeValue(T(μ), T(σ), T(ξ))
end
function convert{T <: Real, S <: Real}(::Type{GeneralizedExtremeValue{T}}, d::GeneralizedExtremeValue{S})
    GeneralizedExtremeValue(T(d.μ), T(d.σ), T(d.ξ))
end

minimum{T <: Real}(d::GeneralizedExtremeValue{T}) =
        d.ξ > 0.0 ? d.μ - d.σ / d.ξ : -convert(T, Inf)
maximum{T <: Real}(d::GeneralizedExtremeValue{T}) =
        d.ξ < 0.0 ? d.μ - d.σ / d.ξ : convert(T, Inf)


#### Parameters

shape(d::GeneralizedExtremeValue) = d.ξ
scale(d::GeneralizedExtremeValue) = d.σ
location(d::GeneralizedExtremeValue) = d.μ
params(d::GeneralizedExtremeValue) = (d.μ, d.σ, d.ξ)


#### Statistics

testfd(d::GeneralizedExtremeValue) = d.ξ^3
g(d::GeneralizedExtremeValue, k::Real) = gamma(1 - k * d.ξ) # This should not be exported.

function median(d::GeneralizedExtremeValue)
    (μ, σ, ξ) = params(d)

    if abs(ξ) < eps() # ξ == 0.0
        return μ - σ * log(log(2.0))
    else
        return μ + σ * (log(2.0) ^ (- ξ) - 1.0) / ξ
    end
end

function mean{T <: Real}(d::GeneralizedExtremeValue{T})
    (μ, σ, ξ) = params(d)

    if abs(ξ) < eps() # ξ == 0.0
        return μ + σ * γ
    elseif ξ < 1.0
        return μ + σ * (gamma(1.0 - ξ) - 1.0) / ξ
    else
        return convert(T, Inf)
    end
end

function mode(d::GeneralizedExtremeValue)
    (μ, σ, ξ) = params(d)

    if abs(ξ) < eps() # ξ == 0.0
        return μ
    else
        return μ + σ * ((1.0 + ξ) ^ (- ξ) - 1.0) / ξ
    end
end

function var{T <: Real}(d::GeneralizedExtremeValue{T})
    (μ, σ, ξ) = params(d)

    if abs(ξ) < eps() # ξ == 0.0
        return σ ^ 2.0 * π ^ 2.0 / 6.0
    elseif ξ < 0.5
        return σ ^ 2.0 * (g(d, 2.0) - g(d, 1.0) ^ 2.0) / ξ ^ 2.0
    else
        return convert(T, Inf)
    end
end

function skewness{T <: Real}(d::GeneralizedExtremeValue{T})
    (μ, σ, ξ) = params(d)

    if abs(ξ) < eps() # ξ == 0.0
        return 12.0 * sqrt(6.0) * zeta(3.0) / pi ^ 3.0
    elseif ξ < 1.0 / 3.0
        g1 = g(d, 1)
        g2 = g(d, 2)
        g3 = g(d, 3)
        return sign(ξ) * (g3 - 3.0 * g1 * g2 + 2.0 * g1 ^ 3.0) / (g2 - g1 ^ 2.0) ^ (3.0 / 2.0)
    else
        return convert(T, Inf)
    end
end

function kurtosis{T <: Real}(d::GeneralizedExtremeValue{T})
    (μ, σ, ξ) = params(d)

    if abs(ξ) < eps() # ξ == 0.0
        return 12.0 / 5.0 * one(T)
    elseif ξ < 1.0 / 4.0
        g1 = g(d, 1)
        g2 = g(d, 2)
        g3 = g(d, 3)
        g4 = g(d, 4)
        return (g4 - 4.0 * g1 * g3 + 6.0 * g2 * g1 ^ 2.0 - 3.0 * g1 ^ 4.0) / (g2 - g1 ^ 2.0) ^ 2.0 - 3.0
    else
        return convert(T, Inf)
    end
end

function entropy(d::GeneralizedExtremeValue)
    (μ, σ, ξ) = params(d)
    return log(σ) + γ * ξ + (1.0 + γ)
end

function quantile(d::GeneralizedExtremeValue, p::Real)
    (μ, σ, ξ) = params(d)

    if abs(ξ) < eps() # ξ == 0.0
        return μ + σ * (- log(- log(p)))
    else
        return μ + σ * ((- log(p)) ^ (- ξ) - 1.0) / ξ
    end
end


#### Support

insupport(d::GeneralizedExtremeValue, x::Real) = minimum(d) <= x <= maximum(d)


#### Evaluation

function logpdf{T <: Real}(d::GeneralizedExtremeValue{T}, x::Real)
    if x == -Inf || x == Inf || ! insupport(d, x)
      return -convert(T, Inf)
    else
        (μ, σ, ξ) = params(d)

        z = (x - μ) / σ # Normalise x.
        if abs(ξ) < eps() # ξ == 0.0
            t = z
            return - log(σ) - t - exp(- t)
        else
            if z * ξ == -1.0 # Otherwise, would compute zero to the power something.
                return -convert(T, Inf)
            else
                t = (1.0 + z * ξ) ^ (- 1.0 / ξ)
                return - log(σ) + (ξ + 1.0) * log(t) - t
            end
        end
    end
end

function pdf{T <: Real}(d::GeneralizedExtremeValue{T}, x::Real)
    if x == -Inf || x == Inf || ! insupport(d, x)
        return zero(T)
    else
        (μ, σ, ξ) = params(d)

        z = (x - μ) / σ # Normalise x.
        if abs(ξ) < eps() # ξ == 0.0
            t = exp(- z)
            return (t * exp(- t)) / σ
        else
            if z * ξ == -1.0 # In this case: zero to the power something.
                return zero(T)
            else
                t = (1.0 + z * ξ) ^ (- 1.0 / ξ)
                return (t ^ (ξ + 1.0) * exp(- t)) / σ
            end
        end
    end
end

function logcdf{T <: Real}(d::GeneralizedExtremeValue{T}, x::Real)
    if insupport(d, x)
        (μ, σ, ξ) = params(d)

        z = (x - μ) / σ # Normalise x.
        if abs(ξ) < eps() # ξ == 0.0
            return - exp(- z)
        else
            return - (1.0 + z * ξ) ^ ( -1.0 / ξ)
        end
    elseif x <= minimum(d)
        return -convert(T, Inf)
    else
        return zero(T)
    end
end

function cdf{T <: Real}(d::GeneralizedExtremeValue{T}, x::Real)
    if insupport(d, x)
        (μ, σ, ξ) = params(d)

        z = (x - μ) / σ # Normalise x.
        if abs(ξ) < eps() # ξ == 0.0
            t = exp(- z)
        else
            t = (1.0 + z * ξ) ^ (- 1.0 / ξ)
        end
        return exp(- t)
    elseif x <= minimum(d)
        return zero(T)
    else
        return one(T)
    end
end

logccdf(d::GeneralizedExtremeValue, x::Real) = log1p(- cdf(d, x))
ccdf(d::GeneralizedExtremeValue, x::Real) = - expm1(logcdf(d, x))


#### Sampling

function rand(d::GeneralizedExtremeValue)
    (μ, σ, ξ) = params(d)

    # Generate a Float64 random number uniformly in (0,1].
    u = 1.0 - rand()

    if abs(ξ) < eps() # ξ == 0.0
        rd = - log(- log(u))
    else
        rd = expm1(- ξ * log(- log(u))) / ξ
    end

    return μ + σ * rd
end
