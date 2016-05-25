immutable Rayleigh{T <: Real} <: ContinuousUnivariateDistribution
    σ::T

    Rayleigh(σ::T) = (@check_args(Rayleigh, σ > zero(σ)); new(σ))

end

Rayleigh{T <: Real}(σ::T) = Rayleigh{T}(σ)
Rayleigh{T <: Integer}(σ::T) = Rayleigh(Float64(σ))
Rayleigh() = Rayleigh(1.0)

@distr_support Rayleigh 0.0 Inf

#### Conversions
Rayleigh{T <: Real, S <: Real}(::Type{Rayleigh{T}}, σ::S) = Rayleigh(T(σ))
Rayleigh{T <: Real, S <: Real}(::Type{Rayleigh{T}}, d::Rayleigh{S}) = Rayleigh(T(d.σ))

#### Parameters

scale(d::Rayleigh) = d.σ
params(d::Rayleigh) = (d.σ,)


#### Statistics

mean(d::Rayleigh) = sqrthalfπ * d.σ
median(d::Rayleigh) = 1.177410022515474691 * d.σ   # sqrt(log(4.0)) = 1.177410022515474691
mode(d::Rayleigh) = d.σ

var(d::Rayleigh) = 0.429203673205103381 * d.σ^2   # (2.0 - π / 2) = 0.429203673205103381
std(d::Rayleigh) = 0.655136377562033553 * d.σ

skewness(d::Rayleigh) = 0.631110657818937138
kurtosis(d::Rayleigh) = 0.245089300687638063

entropy(d::Rayleigh) = 0.942034242170793776 + log(d.σ)


#### Evaluation

function pdf(d::Rayleigh, x::Real)
	σ2 = d.σ^2
	x > 0.0 ? (x / σ2) * exp(- (x^2) / (2.0 * σ2)) : zero(d.σ)
end

function logpdf(d::Rayleigh, x::Real)
	σ2 = d.σ^2
	x > 0.0 ? log(x / σ2) - (x^2) / (2.0 * σ2) : -Inf
end

logccdf(d::Rayleigh, x::Real) = - (x^2) / (2.0 * d.σ^2)
ccdf(d::Rayleigh, x::Real) = exp(logccdf(d, x))

cdf(d::Rayleigh, x::Real) = 1.0 - ccdf(d, x)
logcdf(d::Rayleigh, x::Real) = log1mexp(logccdf(d, x))

quantile(d::Rayleigh, p::Real) = sqrt(-2.0 * d.σ^2 * log1p(-p))


#### Sampling

rand(d::Rayleigh) = d.σ * sqrt(2.0 * randexp())
