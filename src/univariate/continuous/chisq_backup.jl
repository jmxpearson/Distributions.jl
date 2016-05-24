immutable Chisq{T <: Real} <: ContinuousUnivariateDistribution
    ν::T

    Chisq(ν::T) = (@check_args(Chisq, ν > zero(ν)); new(ν))
end

Chisq{T <: Real}(ν::T) = Chisq{T}(ν)
Chisq(ν::Int) = Chisq(Float64(ν))

@distr_support Chisq 0.0 Inf

### Conversions
convert{T <: Real, S <: Real}(::Type{Chisq{T}}, ν::S) = Chisq(T(ν))
convert{T <: Real, S <: Real}(::Type{Chisq{T}}, d::Chisq{S}) = Chisq(T(d.ν))


#### Statistics

mean(d::Chisq) = d.ν

var(d::Chisq) = 2.0 * d.ν

skewness(d::Chisq) = sqrt(8.0 / d.ν)

kurtosis(d::Chisq) = 12.0 / d.ν

mode(d::Chisq) = d.ν > 2.0 ? d.ν - 2.0 : 0.0

function median(d::Chisq; approx::Bool=false)
    if approx
        return d.ν * (1.0 - 2.0 / (9.0 * d.ν))^3
    else
        return quantile(d, 0.5)
    end
end

function entropy(d::Chisq)
    hν = 0.5 * d.ν
    hν + logtwo + lgamma(hν) + (1.0 - hν) * digamma(hν)
end


#### Evaluation

@_delegate_statsfuns Chisq chisq ν

mgf(d::Chisq, t::Real) = (1.0 - 2.0 * t)^(-d.ν * 0.5)

cf(d::Chisq, t::Real) = (1.0 - 2.0 * im * t)^(-d.ν * 0.5)

gradlogpdf(d::Chisq, x::Float64) =  x > 0.0 ? (d.ν * 0.5 - 1) / x - 0.5 : 0.0


#### Sampling

_chisq_rand(ν::Float64) = StatsFuns.Rmath.chisqrand(ν)
rand(d::Chisq) = _chisq_rand(d.ν)