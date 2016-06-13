immutable NoncentralT{T <: Real} <: ContinuousUnivariateDistribution
    ν::T
    λ::T

    function NoncentralT(ν::T, λ::T)
    	@check_args(NoncentralT, ν > zero(ν))
        @check_args(NoncentralT, λ >= zero(λ))
        new(ν, λ)
    end
end

NoncentralT{T <: Real}(ν::T, λ::T) = NoncentralT{T}(ν, λ)
NoncentralT(ν::Real, λ::Real) = NoncentralT(promote(ν, λ)...)
NoncentralT(ν::Integer, λ::Integer) = NoncentralT(Float64(ν), Float64(λ))

@distr_support NoncentralT -Inf Inf

### Conversions
convert{T <: Real, S <: Real}(::Type{NoncentralT{T}}, ν::S, λ::S) = NoncentralT(T(ν), T(λ))
convert{T <: Real, S <: Real}(::Type{NoncentralT{T}}, d::NoncentralT{S}) = NoncentralT(T(d.ν), T(d.λ))

### Parameters

params(d::NoncentralT) = (d.ν, d.λ)


### Statistics

function mean{T <: Real}(d::NoncentralT{T})
    if d.ν > 1.0
        isinf(d.ν) ? d.λ :
        sqrt(0.5*d.ν) * d.λ * gamma(0.5*(d.ν-1)) / gamma(0.5*d.ν)
    else
        convert(T, NaN)
    end
end

function var{T <: Real}(d::NoncentralT{T})
    d.ν > 2.0 ? d.ν*(1+d.λ^2)/(d.ν-2.0) - mean(d)^2 : convert(T, NaN)
end

### Evaluation & Sampling

@_delegate_statsfuns NoncentralT ntdist ν λ

function rand(d::NoncentralT)
    z = randn()
    v = rand(Chisq(d.ν))
    (z+d.λ)/sqrt(v/d.ν)
end
