immutable Cauchy{T <: Real} <: ContinuousUnivariateDistribution
    μ::T
    σ::T

    function Cauchy(μ::T, σ::T)
        @check_args(Cauchy, σ > zero(σ))
        new(μ, σ)
    end
end

Cauchy{T <: Real}(μ::T, σ::T) = Cauchy{T}(μ, σ)
Cauchy(μ::Real, σ::Real) = Cauchy(promote(μ, σ)...)
Cauchy(n::Int, μ::Integer, σ::Integer) = Cauchy(n, Float64(μ), Float64(σ))
Cauchy(n::Int, μ::Integer, σ::Real) = Cauchy(n, Float64(μ), σ)
Cauchy(n::Int, μ::Real, σ::Integer) = Cauchy(n, μ, Float64(σ))
Cauchy(μ::Real) = Cauchy(μ, 1.0)
Cauchy() = Cauchy(0.0, 1.0)

@distr_support Cauchy -Inf Inf

#### Conversions
function convert{T <: Real, S <: Real}(::Type{Cauchy{T}}, μ::S, σ::S)
    Cauchy(T(μ), T(σ))
end
function convert{T <: Real, S <: Real}(::Type{Cauchy{T}}, d::Cauchy{S})
    Cauchy(T(d.μ), T(d.σ))
end

#### Parameters

location(d::Cauchy) = d.μ
scale(d::Cauchy) = d.σ

params(d::Cauchy) = (d.μ, d.σ)


#### Statistics

mean(d::Cauchy) = NaN
median(d::Cauchy) = d.μ
mode(d::Cauchy) = d.μ

var(d::Cauchy) = NaN
skewness(d::Cauchy) = NaN
kurtosis(d::Cauchy) = NaN

entropy(d::Cauchy) = log4π + log(d.σ)


#### Functions

zval(d::Cauchy, x::Real) = (x - d.μ) / d.σ
xval(d::Cauchy, z::Real) = d.μ + z * d.σ

pdf(d::Cauchy, x::Real) = 1.0 / (π * scale(d) * (1.0 + zval(d, x)^2))
logpdf(d::Cauchy, x::Real) = - (log1psq(zval(d, x)) + logπ + log(d.σ))

function cdf(d::Cauchy, x::Real)
    μ, σ = params(d)
    invπ * atan2(x - μ, σ) + 0.5
end

function ccdf(d::Cauchy, x::Real)
    μ, σ = params(d)
    invπ * atan2(μ - x, σ) + 0.5
end

function quantile(d::Cauchy, p::Real)
    μ, σ = params(d)
    μ + σ * tan(π * (p - 0.5))
end

function cquantile(d::Cauchy, p::Real)
    μ, σ = params(d)
    μ + σ * tan(π * (0.5 - p))
end

mgf(d::Cauchy, t::Real) = t == zero(t) ? 1.0 : NaN
cf(d::Cauchy, t::Real) = exp(im * (t * d.μ) - d.σ * abs(t))


#### Fitting

# Note: this is not a Maximum Likelihood estimator
function fit{T<:Real}(::Type{Cauchy}, x::AbstractArray{T})
    l, m, u = quantile(x, [0.25, 0.5, 0.75])
    Cauchy(m, (u - l) / 2.0)
end
