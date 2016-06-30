doc"""
    DiscreteUniform(a,b)

A *Discrete uniform distribution* is a uniform distribution over a consecutive sequence of integers between `a` and `b`, inclusive.

$P(X = k) = 1 / (b - a + 1) \quad \text{for } k = a, a+1, \ldots, b.$

```julia
DiscreteUniform(a, b)   # a uniform distribution over {a, a+1, ..., b}

params(d)       # Get the parameters, i.e. (a, b)
span(d)         # Get the span of the support, i.e. (b - a + 1)
probval(d)      # Get the probability value, i.e. 1 / (b - a + 1)
minimum(d)      # Return a
maximum(d)      # Return b
```

External links

* [Discrete uniform distribution on Wikipedia](http://en.wikipedia.org/wiki/Uniform_distribution_(discrete))
"""
immutable DiscreteUniform{T<:Real} <: DiscreteUnivariateDistribution
    a::T
    b::T
    pv::T

    function DiscreteUniform(a::T, b::T)
        @check_args(DiscreteUniform, a <= b)
        new(a, b, 1 / (b - a + 1))
    end

end

DiscreteUniform{T<:Real}(a::T, b::T) = DiscreteUniform{T}(a, b)
DiscreteUniform(a::Real, b::Real) = DiscreteUniform(promote(a, b)...)
DiscreteUniform(a::Integer, b::Integer) = DiscreteUniform(Float64(a), Float64(b))
DiscreteUniform(b::Real) = DiscreteUniform(0.0, b)
DiscreteUniform() = DiscreteUniform(0.0, 1.0)

@distr_support DiscreteUniform d.a d.b

### Parameters

span(d::DiscreteUniform) = d.b - d.a + 1
probval(d::DiscreteUniform) = d.pv
params(d::DiscreteUniform) = (d.a, d.b)


### Show

show(io::IO, d::DiscreteUniform) = show(io, d, (:a, :b))


### Statistics

mean(d::DiscreteUniform) = middle(d.a, d.b)

median(d::DiscreteUniform) = middle(d.a, d.b)

var(d::DiscreteUniform) = (span(d)^2 - 1) / 12

skewness{T<:Real}(d::DiscreteUniform{T}) = zero(T)

function kurtosis(d::DiscreteUniform)
    n2 = span(d)^2
    -1.2 * (n2 + 1) / (n2 - 1)
end

entropy(d::DiscreteUniform) = log(span(d))

mode(d::DiscreteUniform) = d.a
modes(d::DiscreteUniform) = [d.a:d.b]


### Evaluation

cdf{T<:Real}(d::DiscreteUniform{T}, x::Int) = (x < d.a ? zero(T) :
                                   x > d.b ? one(T) :
                                   (floor(Int,x) - d.a + 1) * d.pv)

pdf{T<:Real}(d::DiscreteUniform{T}, x::Int) = insupport(d, x) ? d.pv : zero(T)

logpdf{T<:Real}(d::DiscreteUniform{T}, x::Int) = insupport(d, x) ? log(d.pv) : -T(Inf)

pdf(d::DiscreteUniform) = fill(probval(d), span(d))

function _pdf!(r::AbstractArray, d::DiscreteUniform, rgn::UnitRange)
    vfirst = round(Int, first(rgn))
    vlast = round(Int, last(rgn))
    vl = max(vfirst, d.a)
    vr = min(vlast, d.b)
    if vl > vfirst
        for i = 1:(vl - vfirst)
            r[i] = 0
        end
    end
    fm1 = vfirst - 1
    if vl <= vr
        pv = d.pv
        for v = vl:vr
            r[v - fm1] = pv
        end
    end
    if vr < vlast
        for i = (vr-vfirst+2):length(rgn)
            r[i] = 0
        end
    end
    return r
end

function _logpdf!{T<:Real}(r::AbstractArray, d::DiscreteUniform{T}, x::AbstractArray)
    lpv = log(probval(d))
    for i = 1:length(x)
        @inbounds r[i] = insupport(d, x[i]) ? lpv : -T(Inf)
    end
    return r
end

quantile(d::DiscreteUniform, p::Float64) = d.a + floor(Int,p * span(d))

function mgf{T<:Real}(d::DiscreteUniform{T}, t::Real)
    a, b = d.a, d.b
    u = b - a + 1
    t == 0 ? one(T) : (exp(t*a) * expm1(t*u)) / (u*expm1(t))
end

function cf(d::DiscreteUniform, t::Real)
    a, b = d.a, d.b
    u = b - a + 1
    t == 0 ? complex(1) : (im*cos(t*(a+b)/2) + sin(t*(a-b-1)/2)) / (u*sin(t/2))
end


### Sampling

rand(d::DiscreteUniform) = randi(d.a, d.b)

# Fit model

function fit_mle{T<:Real}(::Type{DiscreteUniform}, x::AbstractArray{T})
    if isempty(x)
        throw(ArgumentError("x cannot be empty."))
    end

    xmin = xmax = x[1]
    for i = 2:length(x)
        @inbounds xi = x[i]
        if xi < xmin
            xmin = xi
        elseif xi > xmax
            xmax = xi
        end
    end

    DiscreteUniform(xmin, xmax)
end
