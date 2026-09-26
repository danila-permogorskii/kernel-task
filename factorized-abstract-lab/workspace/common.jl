using Printf

"Call f once (compile + warm up), then return the fastest of 'reps ' runs in seconds"
function best_time(f; reps = 30)
    f()
    best = Inf
    for _ in 1:reps
        best = min(best, @elapsed f())
    end
    return best
end


"Allocate bytes by one call of f, measured after a warm-up call"
function alloc_bytes(f)
    f()
    return @allocated f()
end

human(b) = b < 1024   ? @sprintf("%d B", b) :
           b < 1024^2 ? @sprintf("%.1f KiB", b / 1024) :
                        @sprintf("%.1f MiB", b / 1024^2)

us(s) = s < 1e-3 ? @sprintf("%.1f µs", s * 1e6) : @sprintf("%.2f ms", s * 1e3)
