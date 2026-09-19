export LockFreeRingBuffer, write!, read!, available, space_available

"""
    LockFreeRingBuffer

A lock-free ring buffer for audio data.
Capacity must be a power of 2 for fast wrapping via bitwise AND.
"""
mutable struct LockFreeRingBuffer
    buffer::Vector{Float64}
    capacity::Int
    write_pos::Threads.Atomic{Int}
    read_pos::Threads.Atomic{Int}
    
    function LockFreeRingBuffer(capacity::Int)
        if !ispow2(capacity)
            throw(ArgumentError("Capacity must be a power of 2"))
        end
        new(zeros(Float64, capacity), capacity, Threads.Atomic{Int}(0), Threads.Atomic{Int}(0))
    end
end

"""
    available(rb::LockFreeRingBuffer)::Int

Returns the number of readable samples currently in the buffer.
"""
function available(rb::LockFreeRingBuffer)::Int
    w = rb.write_pos[]
    r = rb.read_pos[]
    return w - r
end

"""
    space_available(rb::LockFreeRingBuffer)::Int

Returns the available writable space in the buffer.
"""
function space_available(rb::LockFreeRingBuffer)::Int
    return rb.capacity - available(rb)
end

"""
    Base.isempty(rb::LockFreeRingBuffer)::Bool

Returns true if the buffer is empty.
"""
function Base.isempty(rb::LockFreeRingBuffer)::Bool
    return available(rb) == 0
end

"""
    write!(rb::LockFreeRingBuffer, data::Vector{Float64})::Int

Writes data to the ring buffer (non-blocking).
Returns the actual number of samples written.
"""
function write!(rb::LockFreeRingBuffer, data::Vector{Float64})::Int
    space = space_available(rb)
    to_write = min(length(data), space)
    
    if to_write == 0
        return 0
    end
    
    w_pos = rb.write_pos[]
    mask = rb.capacity - 1
    
    for i in 1:to_write
        rb.buffer[(w_pos & mask) + 1] = data[i]
        w_pos += 1
    end
    
    Threads.atomic_add!(rb.write_pos, to_write)
    
    return to_write
end

"""
    read!(rb::LockFreeRingBuffer, output::Vector{Float64})::Int

Reads data from the ring buffer into the provided output vector (non-blocking).
Returns the actual number of samples read.
"""
function read!(rb::LockFreeRingBuffer, output::Vector{Float64})::Int
    avail = available(rb)
    to_read = min(length(output), avail)
    
    if to_read == 0
        return 0
    end
    
    r_pos = rb.read_pos[]
    mask = rb.capacity - 1
    
    for i in 1:to_read
        output[i] = rb.buffer[(r_pos & mask) + 1]
        r_pos += 1
    end
    
    Threads.atomic_add!(rb.read_pos, to_read)
    
    return to_read
end
