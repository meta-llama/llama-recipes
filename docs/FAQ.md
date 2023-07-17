# FAQ

Here we discuss frequently asked questions that may occur and we found useful along the way.

1. Does FSDP support mixed precision in one FSDP unit? Meaning, in one FSDP unit some of the parameters are in Fp16/Bf16 and others in FP32.

    FSDP requires each FSDP unit to have consistent precision, so this case is not supported at this point. It might be added in future but no ETA at the moment.

2.  How does FSDP handles mixed grad requirements?

    FSDP does not support mixed `require_grad` in one FSDP unit. This means if you are planning to freeze some layers, you need to do it on the FSDP unit level rather than model layer. For example, let us assume our model has 30 decoder layers and we want to freeze the bottom 28 layers and only train 2 top transformer layers. In this case, we need to make sure `require_grad` for the top two transformer layers are set to `True`.

3. How do PEFT methods work with FSDP in terms of grad requirements/layer freezing?

    We wrap the PEFT modules separate from the transfromer layer in auto_wrapping policy, that would result in PEFT models having `require_grad=True` while the rest of the model is  `require_grad=False`.

4. Can I add custom datasets?

    Yes, you can find more information on how to do that [here](Dataset.md).
