# Kolmogorov distribution
# defined as the sup_{t \in [0,1]} |B(t)|, where B(t) is a Brownian bridge
# used in the Kolmogorov--Smirnov test for large n.

immutable Kolmogorov <: ContinuousUnivariateDistribution
end

@distr_support Kolmogorov 0 Inf

params(d::Kolmogorov) = ()


#### Statistics

mean(d::Kolmogorov) = sqrt2π*log(2)/2
var(d::Kolmogorov) = pi^2/12 - pi*log(2)^2/2
# TODO: higher-order moments also exist, can be obtained by differentiating series

mode(d::Kolmogorov) = 0.735467907916572
median(d::Kolmogorov) = 0.8275735551899077

#### Evaluation

# cdf and ccdf are based on series truncation.
# two different series are available, e.g. see:
#   N. Smirnov, "Table for Estimating the Goodness of Fit of Empirical Distributions",
#   The Annals of Mathematical Statistics , Vol. 19, No. 2 (Jun., 1948), pp. 279-281
#   http://projecteuclid.org/euclid.aoms/1177730256
# use one series for small x, one for large x
# 5 terms seems to be sufficient for Float64 accuracy
# some divergence from Smirnov's table in 6th decimal near 1 (e.g. 1.04): occurs in
# both series so assume error in table.

function cdf_raw(d::Kolmogorov, x::Real)
    a = -(pi^2)/(x^2)
    f = exp(a)
    u = (1 + f*(1 + f^2))
    sqrt2π*exp(a/8)*u/x
end

function ccdf_raw(d::Kolmogorov, x::Real)
    f = exp(-2x^2)
    u = (1 - f^3*(1 - f^5*(1 - f^7)))
    2f*u
end

function cdf(d::Kolmogorov,x::Real)
    if x <= 0
        0
    elseif x <= 1
        cdf_raw(d,x)
    else
        1-ccdf_raw(d,x)
    end
end
function ccdf(d::Kolmogorov,x::Real)
    if x <= 0
        1
    elseif x <= 1
        1-cdf_raw(d,x)
    else
        ccdf_raw(d,x)
    end
end


# TODO: figure out how best to truncate series
function pdf(d::Kolmogorov,x::Real)
    if x <= 0
        return 0
    elseif x <= 1
        c = π/(2*x)
        s = 0
        for i = 1:20
            k = ((2i - 1)*c)^2
            s += (k - 1)*exp(-k/2)
        end
        return sqrt2π*s/x^2
    else
        s = 0
        for i = 1:20
            s += (iseven(i) ? -1 : 1)*i^2*exp(-2(i*x)^2)
        end
        return 8*x*s
    end
end


@quantile_newton Kolmogorov

#### Sampling

# Alternating series method, from:
#   Devroye, Luc (1986) "Non-Uniform Random Variate Generation"
#   Chapter IV.5, pp. 163-165.
function rand(d::Kolmogorov)
    t = 0.75
    if rand() < 0.3728329582237386 # cdf(d,t)
        # left interval
        while true
            g = rand_trunc_gamma()

            x = pi/sqrt(8g)
            w = 0
            z = 1/(2g)
            p = exp(-g)
            n = 1
            q = 1
            u = rand()
            while u >= w
                w += z*q
                if u >= w
                    return x
                end
                n += 2
                q = p^(n^2 - 1)
                w -= n^2*q
            end
        end
    else
        while true
            e = randexp()
            u = rand()
            x = sqrt(t^2 + e/2)
            w = 0
            n = 1
            z = exp(-2x^2)
            while u > w
                n += 1
                w += n^2*z^(n^2 - 1)
                if u >= w
                    return x
                end
                n += 1
                w -= n^2*z^(n^2 - 1)
            end
        end
    end
end

# equivalent to
# rand(Truncated(Gamma(1.5,1),tp,Inf))
function rand_trunc_gamma()
    tp = 2.193245422464302 #pi^2/(8*t^2)
    while true
        e0 = rand(Exponential(1.2952909208355123))
        e1 = rand(Exponential(2))
        g = tp + e0
        if (e0^2 <= tp*e1*(g+tp)) || (g/tp - 1 - log(g/tp) <= e1)
            return g
        end
    end
end
