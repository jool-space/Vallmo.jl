module Vallmo

include("safetensors/safetensors.jl")
public safetensors

using Vall: @sizes    # extracted (two consumers: our kernels, Vall's verbs)
public @sizes

include("splitaxis.jl")
public splitaxis

include("kernels/kernels.jl")

include("llm/llm.jl")
public Generation, generate!, generate_captured!, CaptureSession, reset!, qwen35
public snapshot, snapshot!, restore!

end
