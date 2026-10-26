# Viva answers

**1. Model parameter vs prompt parameter?** A model parameter is a learned weight or bias from training. A prompt or inference parameter is a setting or input used at generation time, such as the system prompt, temperature, or maximum output tokens; it does not change learned weights.

**2. Why local instead of cloud?** Local inference can improve privacy, offline availability, and cost control after setup. It requires sufficient hardware, storage, and maintenance and may be slower than hosted services.

**3. What is quantisation?** Quantisation stores or computes model weights using lower numerical precision, such as 4-bit instead of 16-bit. It can reduce RAM/VRAM use and sometimes improve speed, but aggressive quantisation can reduce response quality.
