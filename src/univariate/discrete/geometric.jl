immutable Geometric{T <: Real} <: DiscreteUnivariateDistribution
    p::T

    function Geometric(p::T)
        @check_args(Geometric, zero(p) < p < one(p))
    	new(p)
    end
end

Geometric{T <: Real}(p::T) = Geometric{T}(p)
Geometric() = Geometric(0.5)

@distr_support Geometric 0 Inf

### Conversions
convert{T <: Real, S <: Real}(::Type{Geometric{T}}, p::S) = Geometric(T(p))
convert{T <: Real, S <: Real}(::Type{Geometric{T}}, d::Geometric{S}) = Geometric(T(d.p))

### Parameters

succprob(d::Geometric) = d.p
failprob(d::Geometric) = one(d.p) - d.p
params(d::Geometric) = (d.p,)


### Statistics

mean(d::Geometric) = failprob(d) / succprob(d)

median(d::Geometric) = -fld(logtwo, log1p(-d.p)) - 1

mode(d::Geometric) = 0

var(d::Geometric) = (one(d.p) - d.p) / abs2(d.p)

skewness(d::Geometric) = (2*one(d.p) - d.p) / sqrt(one(d.p) - d.p)

kurtosis(d::Geometric) = 6*one(d.p) + abs2(d.p) / (one(d.p) - d.p)

entropy(d::Geometric) = (-xlogx(succprob(d)) - xlogx(failprob(d))) / d.p


### Evaluations

function pdf(d::Geometric, x::Int)
    if x >= 0
        p = d.p
        return p < one(p) / 10 ? p * exp(log1p(-p) * x) : d.p * (one(p) - p)^x
    else
        return zero(p)
    end
end

logpdf(d::Geometric, x::Int) = x >= 0 ? log(d.p) + log1p(-d.p) * x : -Inf

immutable RecursiveGeomProbEvaluator <: RecursiveProbabilityEvaluator
    p0::Float64
end

RecursiveGeomProbEvaluator(d::Geometric) = RecursiveGeomProbEvaluator(failprob(d))
nextpdf(s::RecursiveGeomProbEvaluator, p::Float64, x::Integer) = p * s.p0
_pdf!(r::AbstractArray, d::Geometric, rgn::UnitRange) = _pdf!(r, d, rgn, RecursiveGeomProbEvaluator(d))


function cdf(d::Geometric, x::Int)
    x < 0 && return zero(d.p)
    p = succprob(d)
    n = x + 1
    p < one(d.p)/2 ? -expm1(log1p(-p)*n) : one(d.p)-(one(d.p)-p)^n
end

function ccdf(d::Geometric, x::Int)
    x < 0 && return one(d.p)
    p = succprob(d)
    n = x + 1
    p < one(d.p)/2 ? exp(log1p(-p)*n) : (one(d.p)-p)^n
end

logcdf(d::Geometric, x::Int) = x < 0 ? -Inf : log1mexp(log1p(-d.p) * (x + 1))

logccdf(d::Geometric, x::Int) =  x < 0 ? zero(d.p) : log1p(-d.p) * (x + 1)

quantile(d::Geometric, p::Float64) = invlogccdf(d, log1p(-p))

cquantile(d::Geometric, p::Float64) = invlogccdf(d, log(p))

invlogcdf(d::Geometric, lp::Float64) = invlogccdf(d, log1mexp(lp))

function invlogccdf(d::Geometric, lp::Float64)
    if (lp > zero(d.p)) || isnan(lp)
        return NaN
    elseif isinf(lp)
        return Inf
    elseif lp == zero(d.p)
        return zero(d.p)
    end
    max(ceil(lp/log1p(-d.p))-one(d.p),zero(d.p))
end

function mgf(d::Geometric, t::Real)
    p = succprob(d)
    p / (expm1(-t) + p)
end

function cf(d::Geometric, t::Real)
    p = succprob(d)
    # replace with expm1 when complex version available
    p / (exp(-t*im) - one(d.p) + p)
end


### Sampling

rand(d::Geometric) = floor(Int,-randexp() / log1p(-d.p))


### Model Fitting

immutable GeometricStats <: SufficientStats
    sx::Float64
    tw::Float64

    GeometricStats(sx::Real, tw::Real) = new(sx, tw)
end

suffstats{T<:Integer}(::Type{Geometric}, x::AbstractArray{T}) = GeometricStats(sum(x), length(x))

function suffstats{T<:Integer}(::Type{Geometric}, x::AbstractArray{T}, w::AbstractArray{Float64})
    n = length(x)
    if length(w) != n
        throw(ArgumentError("Inconsistent argument dimensions."))
    end
    sx = 0.
    tw = 0.
    for i = 1:n
        wi = w[i]
        sx += wi * x[i]
        tw += wi
    end
    GeometricStats(sx, tw)
end

fit_mle(::Type{Geometric}, ss::GeometricStats) = Geometric(1.0 / (ss.sx / ss.tw + 1.0))
